#################################################################################################
# Import Libraries
from __future__ import annotations

import base64
import datetime
import errno
import html
import io
import logging
import os
import re
import stat
import sys
import tempfile
import time
import uuid
import zoneinfo
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path, PurePath
from typing import Any, Literal, NamedTuple, Protocol, TypeVar, cast

import exchange_calendars as xcals  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import polars as pl
import pyarrow as pa  # type: ignore[import-untyped]
from arrow_odbc import read_arrow_batches_from_odbc  # type: ignore[import-untyped]
from filelock import FileLock
from matplotlib.figure import Figure

__all__ = [
    "parse_yyyymmdd",
    "compute_lookback_startdate",
    "resolve_runtime_params",
    "compute_mtd_date_range",
    "lazy_parquet",
    "winsorized_rolling_stats",
    "plot_time_series",
    "save_matplotlib_charts_as_html",
    "save_results",
    "redshift_query",
    "RedshiftIngestError",
]


#################################################################################################
# Module-level utilities used across multiple sections.

def _datetime_tz(dtype: pl.DataType) -> str | None:
    """Return the timezone of a Polars Datetime dtype, or None if naive.

    Polars renamed the attribute from ``tz`` to ``time_zone`` around 0.18;
    check both so older installs don't crash with ``AttributeError`` when
    a caller opts into ``require_tz_aware_timestamps``. Collapse ``""`` to
    ``None`` because a handful of polars minor versions report the absence
    of a zone as an empty string rather than ``None``, which would
    otherwise let a naive column slip through an ``is None`` check.
    """
    tz = getattr(dtype, "time_zone", None)
    if tz is None:
        tz = getattr(dtype, "tz", None)
    return tz or None


# Below this length, treating a secret as a redaction pattern does more
# damage to error diagnosability than it prevents: a 1-3 char secret is
# almost certainly present as an incidental substring in any multi-line
# ODBC error (codes like "08S01", hex bytes, status tokens), and every
# such match turns the message into ``<redacted>``-spam. AWS keys are
# >= 20 chars; any password short enough to hit this gate has much bigger
# problems than a traceback. The boundary password validators in
# ``redshift_query`` rejects < this many chars at input.
_MIN_SECRET_REDACTION_LEN: int = 4


def _redact_secrets(text: str, secrets: Sequence[str]) -> str:
    """Replace known-secret substrings in ``text`` with ``<redacted>``.

    Some ODBC drivers echo the full connection string (or kwargs they
    received) verbatim in their error text. This helper is applied to
    exception messages from ``arrow_odbc`` / the underlying driver
    before they are surfaced to the caller or to logs. Empty and
    under-threshold secrets are skipped (see ``_MIN_SECRET_REDACTION_LEN``).

    Secrets are replaced longest-first: if one secret happens to be a
    prefix/substring of another (e.g. ``password="abc"`` while
    ``aws_session_token="abc1234"``), a shorter-first pass would turn the
    longer secret into ``"<redacted>1234"``, leaking the suffix. Sorting
    descending by length matches the more specific string first.
    """
    out = text
    for s in sorted(filter(None, secrets), key=len, reverse=True):
        if len(s) < _MIN_SECRET_REDACTION_LEN:
            continue
        out = out.replace(s, "<redacted>")
    return out


def _build_dtype_tuple(*names: str) -> tuple[type[pl.DataType], ...]:
    """Look up Polars dtype classes by name, dropping ones the installed
    polars version does not export.

    Some dtypes (``pl.Decimal``, occasionally ``pl.Array``) are absent on
    older polars builds. Building the tuple dynamically keeps module
    import green on those versions; the missing dtype is simply not
    checked. ``isinstance(cls, type) and issubclass(cls, pl.DataType)``
    narrows ``cls`` to ``type[pl.DataType]`` for mypy.
    """
    out: list[type[pl.DataType]] = []
    for name in names:
        cls: object = getattr(pl, name, None)
        if isinstance(cls, type) and issubclass(cls, pl.DataType):
            out.append(cls)
    return tuple(out)


# Numeric dtype classes for the opt-in finiteness checks
# (CLAUDE.md inbound contract item 4). Used by ``redshift_query``.
_NUMERIC_DTYPE_CLASSES: tuple[type[pl.DataType], ...] = _build_dtype_tuple(
    "Int8", "Int16", "Int32", "Int64",
    "UInt8", "UInt16", "UInt32", "UInt64",
    "Float32", "Float64",
    "Decimal",
)
_FLOAT_DTYPE_CLASSES: tuple[type[pl.DataType], ...] = _build_dtype_tuple(
    "Float32", "Float64"
)

# Generous but finite upper bounds for the ODBC ingest path
# (redshift_query) to catch typos without over-constraining.
# 2 GiB covers ODBC wide-buffer limits.
_MAX_BATCH_SIZE = 10_000_000
_MAX_TIMEOUT_SEC = 86_400  # 1 day; beyond this the caller almost certainly meant "no timeout"
_MAX_ROWS_CAP = 10**12  # trillion-row hard ceiling on the caller-supplied max_rows
_MAX_VALUE_BYTES = 2**31 - 1  # 2 GiB; ODBC wide-buffer upper bound


# ─── Shared NAS / SMB resilience primitives ───────────────────────────
# These are used by every retry envelope in this module: the parquet
# writer (save_results), the chart writer (save_matplotlib_charts_as_html),
# and the parquet reader (lazy_parquet).  Defining them once here keeps
# the transient-lock errno set, the chmod-on-EACCES recovery, and the
# best-effort tmp-cleanup pattern from drifting between sites.

# Safe errno access: these POSIX constants may be absent on some Windows
# Python builds.  Using -1 as sentinel ensures they never accidentally
# match a real errno value.
_ESTALE: int = getattr(errno, "ESTALE", -1)  # NFS stale file handle
_ETXTBSY: int = getattr(errno, "ETXTBSY", -1)  # Text file busy


def _is_transient_lock(e: OSError) -> bool:
    """Return True when ``e`` represents a transient NAS / SMB lock.

    Covers WinError 5 (Access Denied), 32 (Sharing Violation), 33 (Lock
    Violation) on Windows; EACCES, EBUSY, EAGAIN, ESTALE, ETXTBSY on
    POSIX.  Used by every NAS-resilient retry envelope in this module.
    """
    win_err = getattr(e, "winerror", 0)
    posix_err = getattr(e, "errno", 0)
    return win_err in (5, 32, 33) or posix_err in (
        errno.EACCES,
        errno.EBUSY,
        errno.EAGAIN,
        _ESTALE,
        _ETXTBSY,
    )


def _is_access_denied(e: OSError) -> bool:
    """True when ``e`` is the cross-platform Access-Denied case.

    Backup / compliance tools on Isilon set the read-only attribute on
    both Windows (``WinError 5``) and Linux (``EACCES`` on NFS/SMB);
    every NAS-mutation retry helper needs to detect this case to clear
    the attribute before the next attempt.
    """
    return getattr(e, "winerror", 0) == 5 or getattr(e, "errno", 0) == errno.EACCES


def _unlink_best_effort(path: str | Path) -> None:
    """Unlink ``path`` if present; swallow ``OSError`` (best-effort cleanup).

    Use only at sites where leaving the file in place is acceptable
    (tmp-file cleanup paths, where a retry will overwrite anyway).
    """
    try:
        Path(path).unlink()
    except OSError:
        pass


def _clear_readonly_attr(path: Path) -> None:
    """If ``path`` exists and is read-only, clear the attribute.

    Best-effort: any error is swallowed because the next retry of the
    surrounding NAS-mutation operation will surface the real error.
    Used by the chmod-on-EACCES branches of ``_makedirs_with_retry``,
    ``_fsync_and_verify_size``, and ``_atomic_replace``.
    """
    try:
        attrs = path.stat().st_mode
        if not (attrs & stat.S_IWRITE):
            path.chmod(attrs | stat.S_IWRITE)
    except OSError:
        pass


_RetryT = TypeVar("_RetryT")


def _retry_on_transient_lock(
    op: Callable[[], _RetryT], *, max_attempts: int = 12
) -> _RetryT:
    """Run ``op`` up to ``max_attempts`` times, retrying on transient
    NAS / SMB locks (see ``_is_transient_lock``).  Backoff: 0.5s … 15s cap.

    Non-transient ``OSError``\\s propagate immediately; the final
    attempt's transient ``OSError`` also propagates so the caller sees a
    real failure rather than a silent loop exit.  Sites needing
    additional recovery (chmod-on-EACCES, partial-tmp cleanup) keep
    their own bespoke loops; this helper is for the simple read-only
    retry case shared by the parquet metadata helpers.
    """
    for attempt in range(max_attempts):
        try:
            return op()
        except OSError as e:
            if not _is_transient_lock(e):
                raise
            if attempt == max_attempts - 1:
                raise
            time.sleep(min(15.0, 0.5 * (2**attempt)))
    raise AssertionError("unreachable")  # for-loop exits via return or raise

# Characters that would let an attacker (or a malformed config value) escape
# a DRIVER={...} / SERVER=... / DATABASE=... segment and inject arbitrary
# ODBC connection-string keys.  Used by redshift_query.
_CONN_STR_FORBIDDEN = (";", "{", "}", "\x00", "\n", "\r")


class _BoundaryChecker:
    """Boundary validators parameterized on the exception class
    (``RedshiftIngestError``)."""

    __slots__ = ("error_cls",)

    def __init__(self, error_cls: type[Exception]) -> None:
        self.error_cls = error_cls

    def conn_token(self, name: str, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise self.error_cls(
                f"{name} must be a non-empty, non-whitespace str, got "
                f"{type(value).__name__}={value!r}"
            )
        if value != value.strip():
            raise self.error_cls(
                f"{name} has leading/trailing whitespace: {value!r}"
            )
        for ch in _CONN_STR_FORBIDDEN:
            if ch in value:
                raise self.error_cls(
                    f"{name} contains forbidden character {ch!r}: {value!r}"
                )
        return value

    def int_(
        self,
        name: str,
        value: int | None,
        *,
        minv: int,
        maxv: int,
        allow_none: bool = False,
    ) -> int | None:
        if value is None:
            if allow_none:
                return None
            raise self.error_cls(f"{name} must not be None")
        # bool is an int subclass; reject so `port=True` doesn't mean 1.
        if isinstance(value, bool) or not isinstance(value, int):
            raise self.error_cls(
                f"{name} must be int, got {type(value).__name__}={value!r}"
            )
        if not (minv <= value <= maxv):
            raise self.error_cls(
                f"{name} must be in [{minv}, {maxv}], got {value}"
            )
        return value

    def bool_(self, name: str, value: bool) -> None:
        if not isinstance(value, bool):
            raise self.error_cls(
                f"{name} must be bool, got {type(value).__name__}={value!r}"
            )

    def optional_bool(self, name: str, value: bool | None) -> None:
        if value is None:
            return
        self.bool_(name, value)

    def optional_str(
        self, name: str, value: str | None, *, sensitive: bool = False
    ) -> None:
        if value is not None and not isinstance(value, str):
            # Credential fields: never echo the rejected value (could be a
            # bytes-wrapped password from un-decoded stdin).
            shown = "<redacted>" if sensitive else repr(value)
            raise self.error_cls(
                f"{name} must be str or None, got {type(value).__name__}={shown}"
            )

    def no_control_chars(self, name: str, value: str) -> None:
        """Reject NUL / LF / CR (usually a stripped-trailing-newline bug)."""
        for bad in ("\x00", "\n", "\r"):
            if bad in value:
                raise self.error_cls(
                    f"{name} contains a control character {bad!r}"
                )

    def list_or_tuple(self, name: str, value: object) -> None:
        """Reject str/bytes/set/dict masquerading as a Sequence — they
        either iterate char-by-char or in non-deterministic order."""
        if isinstance(value, (list, tuple)):
            return
        raise self.error_cls(
            f"{name} must be a list or tuple, not {type(value).__name__}. "
            f"Wrap in a list: {name}=[value] (or list(value))."
        )

    @staticmethod
    def find_duplicates(names: Sequence[str]) -> list[str]:
        seen: set[str] = set()
        dupes_seen: set[str] = set()
        dupes: list[str] = []
        for n in names:
            if n in seen and n not in dupes_seen:
                dupes.append(n)
                dupes_seen.add(n)
            seen.add(n)
        return sorted(dupes)

    def finite_numerics(self, df: pl.DataFrame, *, require: bool) -> None:
        """Reject NULL / NaN / ±Inf in numeric columns when require=True
        (CLAUDE.md inbound contract item 4). MemoryError propagates;
        other polars exceptions are aggregated so the operator sees
        every offending column at once."""
        if not require:
            return
        bad: list[str] = []
        api_errors: list[tuple[str, Exception]] = []
        for name, dtype in df.schema.items():
            if not isinstance(dtype, _NUMERIC_DTYPE_CLASSES):
                continue
            try:
                if df[name].null_count() > 0:
                    bad.append(name)
                    continue
                if isinstance(dtype, _FLOAT_DTYPE_CLASSES):
                    if not bool(df[name].is_finite().all()):
                        bad.append(name)
            except MemoryError:
                raise
            except Exception as exc:  # noqa: BLE001 — aggregate per-col
                api_errors.append((name, exc))
        if bad:
            msg = (
                "require_finite_numerics=True but numeric columns contain "
                f"NULL / NaN / Inf: {bad}"
            )
            if api_errors:
                err_cols = [n for n, _ in api_errors]
                msg += (
                    " (additionally, the check could not be evaluated on "
                    f"columns: {err_cols})"
                )
            raise self.error_cls(msg)
        if api_errors:
            bad_name, api_err = api_errors[0]
            raise self.error_cls(
                f"finiteness check failed on column {bad_name!r}: "
                f"{type(api_err).__name__}: {api_err}"
            ) from api_err


#################################################################################################
# Date parser
def parse_yyyymmdd(value: str | None, label: str) -> datetime.date | None:
    """Parse a YYYYMMDD CLI string; return None if value is None; raise on bad format."""
    if value is None:
        return None
    try:
        return datetime.datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError(
            f"Invalid date format. Expected YYYYMMDD (e.g. 20241231), "
            f"got {label}='{value}'. Original error: {exc}"
        ) from exc

#################################################################################################
# compute_lookback_startdate — resolve the trading session that is N sessions before startdate

def compute_lookback_startdate(
    startdate: datetime.date,
    exchange: str,
    lookback_trading_days: int,
) -> datetime.date:
    """Return the trading session that is ``lookback_trading_days`` sessions before ``startdate`` on the given exchange.

    ``exchange`` is an ``exchange_calendars`` ISO-MIC code (e.g. ``"XNYS"``,
    ``"XNAS"``); it is passed straight to ``xcals.get_calendar`` and an unknown
    code raises there with a clear message. ``lookback_trading_days`` is the
    integer number of sessions to step back; precondition >= 1, guarded at the
    module-level constant's definition site.

    The function asks the calendar to enumerate every session in a generously
    overprovisioned calendar-day window, then slices the last
    ``lookback_trading_days`` to get an exact answer — the calendar-day
    arithmetic is intentionally loose; exactness comes from the slice, not from
    the arithmetic.
    """
    cal = xcals.get_calendar(exchange)
    earliest = startdate - datetime.timedelta(days=lookback_trading_days * 2 + 14)
    sessions_before = cal.sessions_in_range(
        earliest, startdate - datetime.timedelta(days=1)
    ).date
    if len(sessions_before) < lookback_trading_days:
        raise ValueError(
            f"Cannot resolve {lookback_trading_days} trading sessions before "
            f"{startdate}; only {len(sessions_before)} found in "
            f"[{earliest}, {startdate - datetime.timedelta(days=1)}]"
        )
    return cast(datetime.date, sessions_before[-lookback_trading_days])


#################################################################################################
# resolve_runtime_params — apply defaults + validate runtime params for EDA-style scripts

def resolve_runtime_params(
    startdate: datetime.date | None,
    enddate: datetime.date | None,
    write_mode: str | None,
    input_mode: str | None,
    *,
    default_startdate: datetime.date,
    default_enddate: datetime.date,
    default_write_mode: str,
    default_input_mode: str,
    research_folder: str,
    production_folder: str,
) -> tuple[datetime.date, datetime.date, str, str]:
    """Apply default fallbacks; validate write_mode, write_directory, date range.

    Centralizes the resolution rules used by the four invocation modes (CLI
    no-args, CLI with-args, function no-args, function with-args) of
    EDA-style entry-point scripts. All defaults and the
    research/production folder paths are passed in by the caller so this
    helper has no module-level coupling. As a public shared-util boundary,
    caller-supplied defaults and folder paths are themselves validated up
    front so that misattributed error messages can never blame a runtime
    input for a bad default.
    """
    # --- Validate caller-supplied defaults (boundary check) ---------------
    _VALID_WRITE_MODES = {"research", "production"}
    _VALID_INPUT_MODES = {"hot", "cold"}
    if default_write_mode not in _VALID_WRITE_MODES:
        raise ValueError(
            f"Invalid default_write_mode: {default_write_mode!r}. "
            f"Options are {sorted(_VALID_WRITE_MODES)} (exact match)"
        )
    if default_input_mode not in _VALID_INPUT_MODES:
        raise ValueError(
            f"Invalid default_input_mode: {default_input_mode!r}. "
            f"Options are {sorted(_VALID_INPUT_MODES)} (exact match)"
        )
    for _name, _val in (("research_folder", research_folder), ("production_folder", production_folder)):
        if not isinstance(_val, str) or not _val.strip():
            raise ValueError(f"{_name} must be a non-empty string; got {_val!r}")
        if not Path(_val).is_absolute():
            raise ValueError(f"{_name} must be an absolute path; got {_val!r}")

    # --- Narrow datetime → date at entry (CLAUDE.md boundary rule) --------
    # datetime.datetime is a subclass of datetime.date; normalize so downstream
    # date arithmetic doesn't silently mix the two. Applied symmetrically to
    # caller-supplied defaults AND runtime inputs, otherwise a datetime-typed
    # default leaks through into the return tuple declared as date.
    if isinstance(default_startdate, datetime.datetime):
        default_startdate = default_startdate.date()
    if isinstance(default_enddate, datetime.datetime):
        default_enddate = default_enddate.date()
    if not isinstance(default_startdate, datetime.date):
        raise TypeError(f"default_startdate must be datetime.date; got {type(default_startdate).__name__}")
    if not isinstance(default_enddate, datetime.date):
        raise TypeError(f"default_enddate must be datetime.date; got {type(default_enddate).__name__}")
    if startdate is not None and isinstance(startdate, datetime.datetime):
        startdate = startdate.date()
    if enddate is not None and isinstance(enddate, datetime.datetime):
        enddate = enddate.date()
    if startdate is not None and not isinstance(startdate, datetime.date):
        raise TypeError(f"startdate must be datetime.date or None; got {type(startdate).__name__}")
    if enddate is not None and not isinstance(enddate, datetime.date):
        raise TypeError(f"enddate must be datetime.date or None; got {type(enddate).__name__}")

    if default_startdate > default_enddate:
        raise ValueError(
            f"default_startdate ({default_startdate}) must be on or before "
            f"default_enddate ({default_enddate})"
        )

    # --- Apply defaults ---------------------------------------------------
    startdate = startdate if startdate is not None else default_startdate
    enddate = enddate if enddate is not None else default_enddate

    if startdate > enddate:
        raise ValueError(
            f"startdate ({startdate}) must be on or before enddate ({enddate})"
        )

    if write_mode is None:
        write_mode = default_write_mode
    if input_mode is None:
        input_mode = default_input_mode

    if write_mode == "research":
        write_directory = research_folder
    elif write_mode == "production":
        write_directory = production_folder
    else:
        raise ValueError(
            f"Invalid value for write_mode: {write_mode!r}. Options are 'research' or 'production' (exact match; no whitespace or case normalization)"
        )

    if input_mode not in _VALID_INPUT_MODES:
        raise ValueError(
            f"Invalid value for input_mode: {input_mode!r}. Options are 'hot' or 'cold' (exact match; no whitespace or case normalization)"
        )

    write_directory = str(Path(write_directory).resolve())
    if not Path(write_directory).is_dir():
        raise FileNotFoundError(f"Write directory does not exist: {write_directory}")

    return startdate, enddate, write_directory, input_mode


#################################################################################################
# Compute default month-to-date date range from the composite exchange calendar of the given venues.

def _mtd_validate_inputs(
    venues: list[str],
    combine_type: Literal["union", "intersect"],
) -> list[str]:
    """Reject empty venue list / blank venue codes / bad combine_type.
    Return order-preserving dedup."""
    if not venues:
        raise ValueError("compute_mtd_date_range: venues must be non-empty")
    if combine_type not in ("union", "intersect"):
        raise ValueError(
            f"compute_mtd_date_range: combine_type must be 'union' or "
            f"'intersect', got {combine_type!r}"
        )
    seen: set[str] = set()
    unique_venues: list[str] = []
    for v in venues:
        if not v or not v.strip():
            raise ValueError(
                f"compute_mtd_date_range: each venue must be a non-empty "
                f"string, got {v!r}"
            )
        if v not in seen:
            seen.add(v)
            unique_venues.append(v)
    return unique_venues


def _mtd_load_calendars_and_tz(
    unique_venues: list[str],
) -> tuple[list[Any], datetime.tzinfo]:
    """Load each venue's xcals calendar; require all venues share one tz."""
    cals: list[Any] = []
    for v in unique_venues:
        try:
            cal = xcals.get_calendar(v)
        except Exception as e:  # noqa: BLE001 — xcals exceptions drift across versions
            raise ValueError(
                f"compute_mtd_date_range: failed to load calendar for "
                f"venue {v!r}: {e!s}"
            ) from e
        if cal is None:
            raise RuntimeError(
                f"compute_mtd_date_range: xcals returned None for "
                f"venue {v!r}"
            )
        cals.append(cal)

    tzs: list[Any] = []
    for v, c in zip(unique_venues, cals, strict=True):
        try:
            tzs.append(c.tz)
        except Exception as e:  # noqa: BLE001
            raise ValueError(
                f"compute_mtd_date_range: failed to read tz from calendar "
                f"for venue {v!r}: {e!s}"
            ) from e

    tz_missing = [v for v, t in zip(unique_venues, tzs, strict=True) if t is None]
    if tz_missing:
        raise ValueError(
            f"compute_mtd_date_range: calendar(s) for {tz_missing} "
            f"have no timezone"
        )

    tz_labels = {str(t) for t in tzs}
    if len(tz_labels) != 1:
        raise ValueError(
            f"compute_mtd_date_range: venues span multiple timezones "
            f"{tz_labels}"
        )

    return cals, tzs[0]


def _mtd_now_local(tz: datetime.tzinfo) -> datetime.datetime:
    """Return ``datetime.now(tz=tz)``. ``tz`` is isinstance-validated upstream."""
    return datetime.datetime.now(tz=tz)


def _mtd_make_session_probe(
    cals: list[Any],
    combine_type: Literal["union", "intersect"],
) -> tuple[Callable[[datetime.date], bool], list[BaseException]]:
    """Build the session probe (fail-closed on any cal error) and a
    shared error-capture list. Probes EVERY calendar (no short-circuit)
    so a chronic library fault on one venue surfaces in probe_errors."""
    probe_errors: list[BaseException] = []
    combine_fn = all if combine_type == "intersect" else any

    def _is_target_session(d: datetime.date) -> bool:
        results: list[bool] = []
        had_error = False
        for c in cals:
            try:
                results.append(bool(c.is_session(d)))
            except Exception as e:  # noqa: BLE001 — captured for caller hint
                probe_errors.append(e)
                had_error = True
                results.append(False)
        if had_error:
            return False
        return combine_fn(results)

    return _is_target_session, probe_errors


def _mtd_should_include_today(
    *,
    now_local: datetime.datetime,
    today: datetime.date,
    cals: list[Any],
    combine_type: Literal["union", "intersect"],
    is_target_session: Callable[[datetime.date], bool],
    unique_venues: list[str],
) -> bool:
    """Include today iff now_local is at or past the latest session close
    among venues whose calendar has today as a session. Conservative on
    library inconsistency (excludes today + logs)."""
    if not is_target_session(today):
        return False
    try:
        closes: list[Any] = []
        close_errors: list[BaseException] = []
        has_none = False
        for c in cals:
            try:
                cl = c.session_close(today)
            except Exception as e:  # noqa: BLE001 — aggregated below
                close_errors.append(e)
                continue
            if cl is None:
                has_none = True
                continue
            closes.append(cl)

        last_err_repr = (
            f"{type(close_errors[-1]).__name__}: {close_errors[-1]!s}"
            if close_errors
            else "no-exception"
        )
        if combine_type == "intersect":
            # intersect ⇒ every venue must yield a concrete close.
            if close_errors or has_none:
                logging.getLogger(__name__).warning(
                    "compute_mtd_date_range: intersect today-inclusion "
                    "found is_target_session(today)=True but not every "
                    "venue yielded a concrete close (venues=%s, today=%s, "
                    "close_errors=%d, none_closes=%s, last_error=%s)",
                    unique_venues, today, len(close_errors),
                    has_none, last_err_repr,
                )
                return False
        else:  # union — at least one concrete close required
            if not closes:
                logging.getLogger(__name__).warning(
                    "compute_mtd_date_range: union today-inclusion found "
                    "is_target_session(today)=True but zero concrete closes "
                    "(venues=%s, today=%s, close_errors=%d, last=%s)",
                    unique_venues, today, len(close_errors), last_err_repr,
                )
                return False

        latest_close = max(closes)
        return bool(now_local >= latest_close)
    except Exception as e:  # noqa: BLE001 — log + conservative exclude
        logging.getLogger(__name__).warning(
            "compute_mtd_date_range: today-inclusion check failed, "
            "excluding today (venues=%s, combine_type=%s, today=%s): "
            "%s: %s",
            unique_venues, combine_type, today, type(e).__name__, e,
        )
        return False


def _mtd_resolve_lookback_days() -> int:
    """Resolve the lookback window from ``COMPUTE_MTD_MAX_LOOKBACK_DAYS``
    (default 30, capped at 10y to reject env-var typos)."""
    default_days = 30
    max_allowed_days = 3650
    raw_env = os.environ.get("COMPUTE_MTD_MAX_LOOKBACK_DAYS")
    raw = "" if raw_env is None else raw_env.strip()
    if raw == "":
        return default_days
    try:
        days = int(raw)
    except ValueError as e:
        raise ValueError(
            f"compute_mtd_date_range: env var "
            f"COMPUTE_MTD_MAX_LOOKBACK_DAYS must be a positive integer, "
            f"got {raw!r}"
        ) from e
    if days < 1:
        raise ValueError(
            f"compute_mtd_date_range: env var "
            f"COMPUTE_MTD_MAX_LOOKBACK_DAYS must be a positive integer, "
            f"got {raw!r}"
        )
    if days > max_allowed_days:
        raise ValueError(
            f"compute_mtd_date_range: env var "
            f"COMPUTE_MTD_MAX_LOOKBACK_DAYS={raw!r} exceeds safety cap of "
            f"{max_allowed_days} days"
        )
    return days


def _mtd_find_enddate(
    *,
    today: datetime.date,
    include_today: bool,
    max_lookback: int,
    is_target_session: Callable[[datetime.date], bool],
    probe_errors: list[BaseException],
    unique_venues: list[str],
    combine_type: Literal["union", "intersect"],
) -> datetime.date:
    """Walk backwards from today (or yesterday) until a combined session
    is found, up to max_lookback days. On exhaustion surface the last
    probe exception so operators can tell a library fault from a genuine
    long closure."""
    probe = today if include_today else today - datetime.timedelta(days=1)
    for _ in range(max_lookback):
        if is_target_session(probe):
            return probe
        probe -= datetime.timedelta(days=1)

    hint = ""
    if probe_errors:
        last = probe_errors[-1]
        hint = (
            f"; {len(probe_errors)} of {max_lookback} probe(s) raised — "
            f"last was {type(last).__name__}: {last}"
        )
    raise RuntimeError(
        f"compute_mtd_date_range: no {combine_type} session across "
        f"{unique_venues} within {max_lookback} days of {today}"
        f"{hint}"
    )


def _mtd_find_month_start_session(
    *,
    enddate: datetime.date,
    is_target_session: Callable[[datetime.date], bool],
    unique_venues: list[str],
    combine_type: Literal["union", "intersect"],
) -> datetime.date:
    """First combined session in enddate's calendar month."""
    month_first = datetime.date(enddate.year, enddate.month, 1)
    probe = month_first
    while probe <= enddate:
        if is_target_session(probe):
            return probe
        probe += datetime.timedelta(days=1)
    raise RuntimeError(
        f"compute_mtd_date_range: no {combine_type} session in "
        f"[{month_first}, {enddate}] across {unique_venues}"
    )


def compute_mtd_date_range(
    venues: list[str],
    combine_type: Literal["union", "intersect"],
) -> tuple[datetime.date, datetime.date]:
    """
    Return (default_startdate, default_enddate) from the combined trading
    calendars of the given venues.

    combine_type:
        ``"intersect"`` — a date counts iff it is a session on *every*
            venue. Use when a single "all-markets-open" date is required
            (e.g. cross-venue signal construction on shared trading days).
        ``"union"`` — a date counts iff it is a session on *any* venue.
            Use when any contributing venue's data is sufficient for the
            day (e.g. processing per-venue data independently).

    default_enddate: most recent combined session under ``combine_type``.
        Aggressive "today" rule — today counts iff it is a combined
        session AND current local time is at or past the latest session
        close among venues whose calendar has today as a session. For
        ``intersect`` that's all venues; for ``union`` only the subset
        that trades today contributes to the close.
    default_startdate: first combined session in the calendar month
        containing default_enddate.

    The lookback window defaults to 30 days and can be overridden via the
    ``COMPUTE_MTD_MAX_LOOKBACK_DAYS`` environment variable (positive int)
    for markets with historic long closures (e.g. Russian equities 2022).

    Note
    ----
    All venues must share a single timezone for both modes. Cross-tz
    ``union`` is ambiguous ("today" depends on reference tz) and is
    rejected.

    The body of this function is a thin orchestrator over ``_mtd_*``
    helpers (defined above) — each helper encapsulates one phase with
    its own error contract. Extend behavior by editing the relevant
    helper, not by re-inlining logic here.

    Raises
    ------
    TypeError
        ``venues`` is not a list/tuple of strings (bare str rejected so it
        does not silently iterate as characters).
    ValueError
        Empty venue list, non-string or blank venue code, unknown venue,
        any calendar missing a timezone, venues spanning multiple
        timezones or tzinfo implementations, invalid ``combine_type``,
        or a ``COMPUTE_MTD_MAX_LOOKBACK_DAYS`` value that is non-integer,
        non-positive, or exceeds the 10-year safety cap.
    RuntimeError
        Calendar tz fails to produce a current local time; no combined
        session within the configured lookback (error includes last
        probe exception when the exhaustion coincides with library
        faults); post-condition violation (startdate > enddate,
        cross-month range, future enddate, or either endpoint not a
        combined session); or (defensively) no combined session within
        the enddate's calendar month.
    """
    unique_venues = _mtd_validate_inputs(venues, combine_type)
    # Resolve lookback before loading calendars (env-var failures fail fast).
    max_lookback = _mtd_resolve_lookback_days()

    cals, tz = _mtd_load_calendars_and_tz(unique_venues)
    now_local = _mtd_now_local(tz)
    today = now_local.date()

    is_target_session, probe_errors = _mtd_make_session_probe(cals, combine_type)
    include_today = _mtd_should_include_today(
        now_local=now_local,
        today=today,
        cals=cals,
        combine_type=combine_type,
        is_target_session=is_target_session,
        unique_venues=unique_venues,
    )

    enddate = _mtd_find_enddate(
        today=today,
        include_today=include_today,
        max_lookback=max_lookback,
        is_target_session=is_target_session,
        probe_errors=probe_errors,
        unique_venues=unique_venues,
        combine_type=combine_type,
    )
    startdate = _mtd_find_month_start_session(
        enddate=enddate,
        is_target_session=is_target_session,
        unique_venues=unique_venues,
        combine_type=combine_type,
    )
    return startdate, enddate

###################################################################################################
# Define function to lazy scan parquet files, concatenate them into a single polars lazyframe
# and filter by date range
def lazy_parquet(
    folder_path: str,
    date_column: str,
    start_date: datetime.date,
    end_date: datetime.date,
    *,
    require_tz_aware_timestamps: bool = False,
) -> pl.LazyFrame:
    """Lazy-scan every parquet file in folder_path, validate schemas, and
    return a single date-filtered LazyFrame.

    date_column dtype handling:
      * pl.Date — cast is no-op; pure date arithmetic.
      * pl.Datetime tz-aware — stored tz defines the calendar-day boundary.
      * pl.Datetime tz-naive — wall-time date used as-is; correct iff the
        upstream writer and the caller's bounds share one reference frame.
        House convention for naive data in this codebase is America/New_York.

    Set require_tz_aware_timestamps=True to reject tz-naive Datetime at the
    inbound boundary (mirrors redshift_query's flag).

    The NAS transient-lock retry inside this function covers ONLY the
    metadata reads. The actual payload read happens at .collect() time,
    outside this function — callers running concurrently with save_results
    on the same NAS share must wrap .collect() in their own retry.
    """
    if not folder_path or not folder_path.strip():
        raise ValueError("folder_path must not be empty or whitespace")
    if not date_column or not date_column.strip():
        raise ValueError("date_column must not be empty or whitespace")
    # bool is an int subclass; reject truthy int (matches redshift_query).
    if not isinstance(require_tz_aware_timestamps, bool):
        raise TypeError(
            f"require_tz_aware_timestamps must be a bool, got "
            f"{type(require_tz_aware_timestamps).__name__}"
        )

    # Resolve to absolute so scan_parquet paths survive a chdir before .collect().
    folder = Path(folder_path).resolve()
    folder_path = str(folder)

    if not folder.is_dir():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")

    # Reject tz-aware datetime: .date() on a tz-aware value returns the
    # wall-clock date IN THAT TZ, silently shifting the filter boundary.
    for _name, _val in (("start_date", start_date), ("end_date", end_date)):
        if isinstance(_val, datetime.datetime) and _val.tzinfo is not None:
            raise TypeError(
                f"{_name} must be a datetime.date or tz-naive datetime; got "
                f"tz-aware datetime ({_val!r}). Convert at the call site "
                "(``dt.astimezone(target_tz).date()``) and pass a datetime.date."
            )
    # datetime.datetime is a date subclass; normalize so pl.lit() doesn't
    # create a Datetime literal that shifts the filter boundary by hours.
    if isinstance(start_date, datetime.datetime):
        start_date = start_date.date()
    if isinstance(end_date, datetime.datetime):
        end_date = end_date.date()

    if start_date > end_date:
        raise ValueError(
            f"start_date ({start_date}) must be on or before end_date ({end_date})"
        )

    # Retry directory listing on transient NAS errors (3×, 0.5s/1s backoff).
    file_names: list[str] | None = None
    for _listdir_attempt in range(3):
        try:
            file_names = [entry.name for entry in folder.iterdir()]
            break
        except OSError:
            if _listdir_attempt == 2:
                raise
            time.sleep(0.5 * (2**_listdir_attempt))
    if file_names is None:
        raise RuntimeError("lazy_parquet: listdir retry loop exhausted unexpectedly")

    # Filter on extension only — Path.is_file() swallows ENOENT silently.
    parquet_file_names = [
        file for file in file_names if file.lower().endswith(".parquet")
    ]

    if not parquet_file_names:
        raise ValueError(f"No parquet files found in folder: {folder_path}")

    def _read_schema_with_retry(path: str) -> dict[str, pl.DataType]:
        return _retry_on_transient_lock(lambda: pl.read_parquet_schema(path))

    def _scan_with_retry(path: str) -> pl.LazyFrame:
        return _retry_on_transient_lock(lambda: pl.scan_parquet(path))

    def _null_count_with_retry(path: str, col: str) -> int:
        # Polars-written parquet embeds null counts in stats → near-zero IO.
        return _retry_on_transient_lock(
            lambda: int(
                pl.scan_parquet(path)
                .select(pl.col(col).null_count().alias("__lp_nullcount"))
                .collect()
                .item()
            )
        )

    # Files are named with zero-padded ranges; alphabetical sort = chronological.
    file_paths = sorted([str(folder / file) for file in parquet_file_names])

    schemas: dict[str, dict[str, pl.DataType]] = {}
    for path in file_paths:
        # Fail loud on corruption as well as I/O.  Orchestrators (Airflow,
        # Dagster, cron) do not surface stdout warnings, so a warn+continue
        # on ComputeError would silently shrink the dataset while the
        # pipeline reported success.  Corrupt parquet is an inbound-boundary
        # violation and must halt the load.
        schemas[path] = _read_schema_with_retry(path)

    # Every file's schema was read successfully (the loop above raises on
    # any failure), so file_paths and schemas are already in sync.
    reference_path = file_paths[0]
    reference_schema = schemas[reference_path]

    ref_name = Path(reference_path).name
    ref_cols = set(reference_schema.keys())

    # Cap rendered diff to avoid MB-scale exception text on widespread drift.
    _MISMATCH_REPORT_CAP = 20
    mismatches: list[str] = []
    total_mismatches = 0
    for path, schema in schemas.items():
        if schema == reference_schema:
            continue
        total_mismatches += 1
        if len(mismatches) >= _MISMATCH_REPORT_CAP:
            continue
        file_name = Path(path).name
        cur_cols = set(schema.keys())

        missing = ref_cols - cur_cols
        extra = cur_cols - ref_cols
        if missing or extra:
            parts: list[str] = [f"  {file_name} vs {ref_name} — column differences:"]
            if missing:
                parts.append(f"    missing columns: {sorted(missing)}")
            if extra:
                parts.append(f"    extra columns:   {sorted(extra)}")
            mismatches.append("\n".join(parts))
        else:
            dtype_diffs = [
                f"    {col}: {schema[col]} != {reference_schema[col]}"
                for col in reference_schema
                if schema[col] != reference_schema[col]
            ]
            mismatches.append(
                f"  {file_name} vs {ref_name} — dtype differences:\n"
                + "\n".join(dtype_diffs)
            )

    if total_mismatches:
        suffix = ""
        if total_mismatches > _MISMATCH_REPORT_CAP:
            suffix = (
                f"\n  (+{total_mismatches - _MISMATCH_REPORT_CAP} more "
                f"mismatching file(s) suppressed; total={total_mismatches} "
                f"of {len(schemas)} scanned)"
            )
        raise ValueError(
            f"Schema mismatch across parquet files in {folder_path}:\n"
            + "\n".join(mismatches)
            + suffix
        )

    # Validate date_column up front to avoid a deferred collect-time error.
    if date_column not in reference_schema:
        raise ValueError(
            f"date_column '{date_column}' not found in parquet schema. "
            f"Available columns: {list(reference_schema.keys())}"
        )

    col_dtype = reference_schema[date_column]
    base_dtype = col_dtype.base_type()
    if base_dtype not in (pl.Date, pl.Datetime):
        raise TypeError(
            f"date_column '{date_column}' must be Date or Datetime, got {col_dtype}"
        )

    # Opt-in tz-awareness guard (CLAUDE.md inbound contract). pl.Date
    # always passes (no tz concept); only tz-naive pl.Datetime is rejected.
    if require_tz_aware_timestamps and isinstance(col_dtype, pl.Datetime):
        if _datetime_tz(col_dtype) is None:
            raise TypeError(
                f"lazy_parquet: date_column {date_column!r} is tz-naive "
                f"Datetime ({col_dtype}) but require_tz_aware_timestamps=True. "
                f"Attach a tz upstream or convert to pl.Date."
            )

    # Reject nulls in date_column: filter() silently drops null-dated rows.
    null_counts_by_file: dict[str, int] = {}
    for path in file_paths:
        _n = _null_count_with_retry(path, date_column)
        if _n > 0:
            null_counts_by_file[Path(path).name] = _n
    if null_counts_by_file:
        _total_nulls = sum(null_counts_by_file.values())
        _sample = dict(list(null_counts_by_file.items())[:5])
        _more = (
            ""
            if len(null_counts_by_file) <= 5
            else f" (+{len(null_counts_by_file) - 5} more file(s))"
        )
        raise ValueError(
            f"lazy_parquet: date_column {date_column!r} contains "
            f"{_total_nulls} null value(s) across {len(null_counts_by_file)} "
            f"file(s); null-dated rows would be silently dropped by the "
            f"date-range filter. Reject at the inbound boundary instead — "
            f"pre-filter nulls at the call site or fix upstream. "
            f"Sample: {_sample}{_more}"
        )

    # Pre-align columns to reference order so pl.concat(how="vertical")
    # sees identical schema everywhere; real drift raises at collect time.
    reference_columns = list(reference_schema.keys())
    dataframes = []
    for path in file_paths:
        # Cast to Date so the <= filter doesn't drop same-day non-midnight
        # rows (Polars super-casts Date to Datetime midnight).
        scan = _scan_with_retry(path)
        date_expr = pl.col(date_column).cast(pl.Date)
        df = (
            scan
            .select(reference_columns)
            .filter(
                (date_expr >= pl.lit(start_date)) & (date_expr <= pl.lit(end_date))
            )
        )
        dataframes.append(df)

    concatenated_df = pl.concat(dataframes, how="vertical")

    return concatenated_df


###################################################################################################
# Define a function to compute rolling winsorized statistics
class _WinsorPlan(NamedTuple):
    """Expressions and column aliases for the winsorized-rolling pipeline.

    Bundles the input-neutralized ``val_expr`` (fed into ``rolling().agg()``),
    the final ``mean_final`` / ``std_final`` output expressions, and the
    collision-proof internal column aliases used to pass sort / sum /
    sum-of-squares / length through the pipeline.
    """
    val_expr: pl.Expr
    mean_final: pl.Expr
    std_final: pl.Expr
    tag: str
    sorted_c: str
    sum_c: str
    ss_c: str
    len_c: str


def _validate_winsorized_args(
    index_col: str,
    val_col: str,
    group_by: str | None,
    window: str | None,
    window_size: int | None,
    min_samples: int,
) -> None:
    """Reject empty col-name strings, mutually-exclusive window args,
    bool-as-int, non-positive sizes, and min_samples > window_size."""
    for arg_name, arg_val in (("index_col", index_col), ("val_col", val_col)):
        if not arg_val:
            raise TypeError(
                f"{arg_name} must be a non-empty string, got {arg_val!r}"
            )
    if group_by is not None and not group_by:
        raise TypeError(
            f"group_by must be None or a non-empty string, got {group_by!r}"
        )
    if (window is None) == (window_size is None):
        raise ValueError(
            "exactly one of window (duration string) or window_size "
            "(positive int, count-based) must be provided; got "
            f"window={window!r}, window_size={window_size!r}"
        )
    if window is not None and not window:
        raise TypeError(
            f"window must be a non-empty string, got {window!r}"
        )
    # bool is an int subclass; reject so window_size=True doesn't mean 1.
    if window_size is not None and (
        isinstance(window_size, bool) or window_size <= 0
    ):
        raise ValueError(
            f"window_size must be a positive int, got {window_size!r}"
        )
    if isinstance(min_samples, bool) or min_samples < 1:
        raise ValueError(
            f"min_samples must be a positive int, got {min_samples!r}"
        )
    if window_size is not None and min_samples > window_size:
        raise ValueError(
            f"min_samples ({min_samples}) must be <= window_size ({window_size})"
        )


def _validate_winsorized_schema(
    df: pl.DataFrame,
    index_col: str,
    val_col: str,
    group_by: str | None,
) -> tuple[pl.DataType, pl.DataType]:
    """Verify columns exist + are distinct, val_col is numeric (non-bool),
    index_col is Integer/Date/Datetime with no nulls, group_by (if given)
    is groupable with no nulls, and neither collides with the reserved
    output column names. Returns (idx_dtype, val_dtype)."""
    required_cols = [index_col, val_col] + ([group_by] if group_by else [])
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Column(s) not found in df: {missing}. "
            f"Available: {df.columns}"
        )

    if len(set(required_cols)) != len(required_cols):
        raise ValueError(
            f"index_col, val_col, and group_by must all be distinct; got "
            f"index_col={index_col!r}, val_col={val_col!r}, "
            f"group_by={group_by!r}"
        )

    val_dtype = df.schema[val_col]
    if not val_dtype.is_numeric() or isinstance(val_dtype, pl.Boolean):
        raise TypeError(
            f"val_col {val_col!r} must be a numeric (non-Boolean) dtype, "
            f"got {val_dtype}"
        )

    # rolling() supports Integer/Date/Datetime only — Time/Duration error opaquely.
    idx_dtype = df.schema[index_col]
    idx_ok = idx_dtype.is_integer() or isinstance(idx_dtype, (pl.Date, pl.Datetime))
    if not idx_ok:
        raise TypeError(
            f"index_col {index_col!r} must be an integer, Date, or Datetime "
            f"dtype; got {idx_dtype} (Time and Duration are not supported)"
        )
    if df[index_col].null_count() > 0:
        raise ValueError(
            f"index_col {index_col!r} contains null values; rolling windows "
            f"require a non-null monotonic index"
        )

    if group_by is not None:
        gb_dtype = df.schema[group_by]
        if isinstance(gb_dtype, (pl.List, pl.Struct, pl.Array, pl.Object)):
            raise TypeError(
                f"group_by {group_by!r} has dtype {gb_dtype}, which is not "
                f"groupable; use a scalar dtype (Int/Utf8/Categorical/Date/…)"
            )
        # Polars silently lumps null group keys together; fail loudly.
        if df[group_by].null_count() > 0:
            raise ValueError(
                f"group_by {group_by!r} contains null values; drop or fill "
                f"them before calling"
            )

    # Reserved-name collision: index_col / group_by can't shadow output names.
    _output_cols = {"winsorized_mean", "winsorized_std"}
    _reserved_conflict = _output_cols.intersection(
        {index_col, group_by} - {None}
    )
    if _reserved_conflict:
        raise ValueError(
            f"index_col/group_by cannot be named {sorted(_reserved_conflict)!r}; "
            f"those names are reserved for this function's output columns"
        )

    return idx_dtype, val_dtype


# Polars duration-string patterns: full-match, unit-only, (magnitude, unit).
_WINSOR_FULL_RE: re.Pattern[str] = re.compile(r"(\d+(ns|us|ms|mo|[smhdwqyi]))+")
_WINSOR_UNITS_RE: re.Pattern[str] = re.compile(r"(ns|us|ms|mo|[smhdwqyi])")
_WINSOR_COMPONENT_RE: re.Pattern[str] = re.compile(r"(\d+)(ns|us|ms|mo|[smhdwqyi])")


def _validate_winsorized_window_string(
    window: str,
    index_col: str,
    idx_dtype: pl.DataType,
) -> None:
    """Parse the window string; cross-check unit vs idx_dtype."""
    if not _WINSOR_FULL_RE.fullmatch(window):
        raise ValueError(
            f"window {window!r} is not a valid polars duration string; "
            f"expected e.g. '30s', '5m', '1d', '1mo', '10i' (or compound "
            f"forms like '1d12h')"
        )
    if all(int(mag) == 0 for mag, _ in _WINSOR_COMPONENT_RE.findall(window)):
        raise ValueError(
            f"window {window!r} has zero total magnitude; rolling windows "
            f"must be strictly positive"
        )
    window_units = _WINSOR_UNITS_RE.findall(window)
    if idx_dtype.is_integer():
        if any(u != "i" for u in window_units):
            raise ValueError(
                f"window {window!r} uses temporal units but index_col "
                f"{index_col!r} is integer ({idx_dtype}); use an "
                f"'i'-suffixed window like '10i'"
            )
    else:
        if "i" in window_units:
            raise ValueError(
                f"window {window!r} uses the integer unit 'i' but index_col "
                f"{index_col!r} is temporal ({idx_dtype}); use duration "
                f"units like 's', 'm', 'h', 'd', 'mo'"
            )


def _build_winsorized_plan(val_col: str, min_samples: int) -> _WinsorPlan:
    """Build per-window winsorized-mean / -std expressions. min_samples
    raises the per-branch gate floor."""
    # Collision-proof internal column names (caller may have "_sorted" etc).
    tag = f"__wrs_{uuid.uuid4().hex[:8]}"
    sorted_c, sum_c, ss_c, len_c = (
        f"{tag}_sorted", f"{tag}_sum", f"{tag}_ss", f"{tag}_len",
    )

    # Cast to Float64 (avoid x*x int overflow); neutralize ±Inf/NaN to null
    # so a single infinity can't contaminate w_sum/w_ss via Inf - Inf = NaN.
    _raw = pl.col(val_col).cast(pl.Float64, strict=True)
    val_expr = pl.when(_raw.is_finite()).then(_raw).otherwise(None)

    s = pl.col(sorted_c)
    n = pl.col(len_c)
    raw_sum = pl.col(sum_c)
    raw_ss = pl.col(ss_c)

    lo, lo2 = s.list.get(0, null_on_oob=True), s.list.get(1, null_on_oob=True)
    hi, hi2 = s.list.get(-1, null_on_oob=True), s.list.get(-2, null_on_oob=True)

    w_sum = raw_sum - lo - hi + lo2 + hi2
    w_ss = raw_ss - lo * lo - hi * hi + lo2 * lo2 + hi2 * hi2

    mean_big = w_sum / n
    # Clamp at 0 before sqrt: cancellation on near-constant windows can
    # make the numerator a tiny negative (sqrt → NaN).
    var_big = ((w_ss - n * mean_big * mean_big) / (n - 1)).clip(lower_bound=0)

    mean_small = raw_sum / n
    var_small = ((raw_ss - n * mean_small * mean_small) / (n - 1)).clip(lower_bound=0)

    # Per-branch minima: 3 for winsorized, 2 for std, 1 for mean.
    gate_big = max(3, min_samples)
    gate_mean_small = max(1, min_samples)
    gate_std_small = max(2, min_samples)
    mean_expr = (
        pl.when(n >= gate_big).then(mean_big)
        .when(n >= gate_mean_small).then(mean_small)
        .otherwise(None)
    )
    std_expr = (
        pl.when(n >= gate_big).then(var_big.sqrt())
        .when(n >= gate_std_small).then(var_small.sqrt())
        .otherwise(None)
    )

    # Final guard: residual NaN/±Inf (overflow on extreme inputs) → None.
    mean_final = pl.when(mean_expr.is_finite()).then(mean_expr).otherwise(None)
    std_final = pl.when(std_expr.is_finite()).then(std_expr).otherwise(None)

    return _WinsorPlan(
        val_expr=val_expr,
        mean_final=mean_final,
        std_final=std_final,
        tag=tag,
        sorted_c=sorted_c,
        sum_c=sum_c,
        ss_c=ss_c,
        len_c=len_c,
    )


def _run_winsorized_rolling(
    df: pl.DataFrame,
    index_col: str,
    val_col: str,
    group_by: str | None,
    window: str | None,
    window_size: int | None,
    plan: _WinsorPlan,
    idx_dtype: pl.DataType,
    val_dtype: pl.DataType,
) -> pl.DataFrame:
    """Run rolling().agg().with_columns(); both window modes; wrap polars
    exceptions with input-parameter context."""
    sort_keys = [group_by, index_col] if group_by else [index_col]

    # u32 row-index overflow on the count-based path.
    _U32_MAX = 2**32 - 1
    if window_size is not None and df.height > _U32_MAX:
        raise ValueError(
            f"winsorized_rolling_stats: df has {df.height:,} rows, which "
            f"exceeds the u32 row-index maximum ({_U32_MAX:,}) used by the "
            f"count-based window_size path. Use the time-based 'window' "
            f"parameter instead, or pre-chunk the input."
        )

    try:
        # Narrow before sort: caller's df may have pathological unrelated
        # cols (Object, unparameterized Categorical, deeply-nested Struct).
        df = df.select([*sort_keys, val_col]).sort(sort_keys)
        row_idx_c = f"{plan.tag}_rowidx"
        if window_size is not None:
            df = df.with_row_index(name=row_idx_c)
            rolling_period: str = f"{window_size}i"
            rolling_index = row_idx_c
        else:
            if window is None:
                raise RuntimeError(
                    "winsorized_rolling_stats: invariant violated — "
                    "window is None in the time-based branch"
                )
            rolling_period = window
            rolling_index = index_col
        result = (
            df.rolling(index_column=rolling_index, period=rolling_period, group_by=group_by)
            .agg(**{
                plan.sorted_c: plan.val_expr.drop_nulls().sort(),
                plan.sum_c: plan.val_expr.sum(),
                plan.ss_c: (plan.val_expr * plan.val_expr).sum(),
            })
            .with_columns(**{plan.len_c: pl.col(plan.sorted_c).list.len()})
            .with_columns(
                winsorized_mean=plan.mean_final,
                winsorized_std=plan.std_final,
            )
            .drop([plan.sorted_c, plan.len_c, plan.sum_c, plan.ss_c])
        )
        if window_size is not None:
            # Rejoin index_col via row-index key. ``.sort(row_idx_c)``
            # defends against a future polars left-join row-order change.
            result = (
                result
                .join(df.select(row_idx_c, index_col), on=row_idx_c, how="left")
                .sort(row_idx_c)
                .drop(row_idx_c)
            )

        # Canonical column order: [group_by, index_col, winsorized_mean, winsorized_std].
        canonical_cols = (
            ([group_by] if group_by is not None else [])
            + [index_col, "winsorized_mean", "winsorized_std"]
        )
        result = result.select(canonical_cols)
    except Exception as e:  # noqa: BLE001 — wrap polars errors with context
        raise RuntimeError(
            f"winsorized_rolling_stats: polars rolling pipeline failed "
            f"(index_col={index_col!r}, val_col={val_col!r}, "
            f"window={window!r}, group_by={group_by!r}, "
            f"n_rows={df.height}, index_dtype={idx_dtype}, "
            f"val_dtype={val_dtype}). Underlying error: "
            f"{type(e).__name__}: {e}"
        ) from e

    return result


def _verify_winsorized_output(
    result: pl.DataFrame,
    input_height: int,
    index_col: str,
    val_col: str,
    group_by: str | None,
    window: str | None,
) -> None:
    """Post-condition: rolling().agg() is 1:1 row-wise; mismatch breaks
    downstream joins-on-index. Other invariants (dtype, columns,
    finiteness) are pipeline-determined by ``_build_winsorized_plan``
    and ``_run_winsorized_rolling`` — no need to re-check here."""
    if result.height != input_height:
        raise RuntimeError(
            f"winsorized_rolling_stats: output row count {result.height} "
            f"does not match input {input_height} "
            f"(index_col={index_col!r}, val_col={val_col!r}, "
            f"window={window!r}, group_by={group_by!r})"
        )


def winsorized_rolling_stats(
    df: pl.DataFrame,
    index_col: str,
    val_col: str,
    group_by: str | None = None,
    *,
    window: str | None = None,
    window_size: int | None = None,
    min_samples: int = 1,
) -> pl.DataFrame:
    """
    Rolling 1-obs winsorized mean and sample stdev (ddof=1).

    Exactly one of ``window`` (time/integer duration string, e.g. ``"73d"``,
    ``"10i"``) or ``window_size`` (count-based, trailing N rows per group)
    must be provided. ``min_samples`` (default 1) nulls out outputs where
    the window contains fewer than that many non-null finite observations —
    use this to align the winsorized pipeline with a count-based
    ``rolling_*(window_size=N, min_samples=M)`` elsewhere.

    For each window: sort, replace min with 2nd-min and max with 2nd-max, then
    compute mean/std. Derives both from sum/sum-of-squares of the sorted list
    with a rank-1 tail substitution, avoiding per-row list materialization.

    Small-window behavior (no winsorization applied — not enough data to
    meaningfully trim extremes):
      * n == 0 : winsorized_mean = None, winsorized_std = None
      * n == 1 : winsorized_mean = the single value, winsorized_std = None
      * n == 2 : winsorized_mean = arithmetic mean of the two values,
                 winsorized_std = sample std of the two values
      * n >= 3 : full winsorization applied

    "n" here counts non-null finite values per window (nulls, NaN, and ±Inf
    are neutralized to null before aggregation).

    Outputs are always numeric-or-None: NaN and ±Inf are mapped to None.
    Defenses: explicit None for empty windows (avoid 0/0), variance clamped
    to >= 0 before sqrt (avoid NaN from floating-point cancellation on
    near-constant windows), and a final is_finite guard for overflow or
    Inf/NaN leaking in from the input column.

    Raises
    ------
    TypeError
        * df is not a polars.DataFrame
        * index_col / val_col / window is not a non-empty string
        * group_by is neither None nor a non-empty string
        * val_col dtype is non-numeric or Boolean
        * index_col dtype is not Integer / Date / Datetime
        * group_by dtype is List / Struct / Array / Object
    ValueError
        * a named column is missing from df
        * index_col / val_col / group_by are not all distinct
        * index_col contains null values
        * window is not a valid polars duration string
        * window magnitude is zero
        * window unit mismatches the index_col dtype
        * group_by contains null values
        * index_col / group_by collide with the output column names
        * window_size path is used with df.height > 2**32 - 1 (u32
          row-index overflow)
    RuntimeError
        * the polars rolling pipeline raises any other error; the original
          exception is chained via `__cause__` and the message includes
          the input parameters for diagnosis
        * the pipeline returns a non-DataFrame object, output row count
          does not match input row count, one of the expected output
          columns is missing, an output column has an unexpected dtype,
          or a non-finite (NaN/Inf) value leaks past the is_finite gate
          (silent-corruption guards)
    """
    # ── Input validation at the API boundary ────────────────────────────
    _validate_winsorized_args(
        index_col, val_col, group_by, window, window_size, min_samples,
    )
    idx_dtype, val_dtype = _validate_winsorized_schema(
        df, index_col, val_col, group_by,
    )
    if window is not None:
        _validate_winsorized_window_string(window, index_col, idx_dtype)

    # ── Build expressions, then execute the rolling pipeline ────────────
    plan = _build_winsorized_plan(val_col, min_samples)

    # Capture the caller's row count BEFORE sort/rolling so the post-condition
    # compares against the true input height. Reading df.height after the
    # in-pipeline narrowing would compare two equally-wrong heights if sort
    # itself had silently dropped or duplicated rows.
    input_height = df.height
    result = _run_winsorized_rolling(
        df, index_col, val_col, group_by, window, window_size,
        plan, idx_dtype, val_dtype,
    )
    _verify_winsorized_output(
        result, input_height, index_col, val_col, group_by, window,
    )
    return result

#################################################################################################
# Chart helper — eliminates repeated boilerplate for plotting time series with consistent formatting and robust Windows NAS path handling.
def plot_time_series(
    data_series: list[tuple[pl.Series, pl.Series, str]],
    title: str,
    y_format: str = "{x:.3f}",
) -> Figure:
    """Plot one or more (dates, values, label) series on a single axis.

    y_format examples: ``"{x:.3f}"``, ``"{x:.1f}K"`` (caller pre-scales),
    ``"{x:.1f}B"`` (caller pre-scales).

    The returned figure is OPEN — the caller owns the lifecycle. Pass it
    to ``save_matplotlib_charts_as_html`` (which closes after savefig)
    or ``plt.close(fig)`` explicitly.
    """
    # ── Input validation at the API boundary ────────────────────────────
    if not isinstance(data_series, list):
        raise TypeError(
            f"data_series must be a list, got {type(data_series).__name__}"
        )
    if not data_series:
        raise ValueError("data_series must not be empty")
    if not isinstance(title, str):
        raise TypeError(f"title must be a string, got {type(title).__name__}")
    if not isinstance(y_format, str):
        raise TypeError(f"y_format must be a string, got {type(y_format).__name__}")

    for i, item in enumerate(data_series):
        if not isinstance(item, tuple) or len(item) != 3:
            got = (
                f"tuple of length {len(item)}"
                if isinstance(item, tuple)
                else type(item).__name__
            )
            raise ValueError(
                f"data_series[{i}] must be a 3-tuple (dates, values, label), got {got}"
            )
        dates, values, label = item
        if not isinstance(dates, pl.Series):
            raise TypeError(
                f"data_series[{i}][0] (dates) must be a polars.Series, "
                f"got {type(dates).__name__}"
            )
        if not isinstance(values, pl.Series):
            raise TypeError(
                f"data_series[{i}][1] (values) must be a polars.Series, "
                f"got {type(values).__name__}"
            )
        if not isinstance(label, str):
            raise TypeError(
                f"data_series[{i}][2] (label) must be a string, "
                f"got {type(label).__name__}"
            )
        if len(dates) != len(values):
            raise ValueError(
                f"data_series[{i}] length mismatch: dates has {len(dates)} "
                f"elements, values has {len(values)}"
            )
        if len(dates) == 0:
            raise ValueError(f"data_series[{i}] series must not be empty")
        val_dtype = values.dtype
        if not val_dtype.is_numeric() or isinstance(val_dtype, pl.Boolean):
            raise TypeError(
                f"data_series[{i}][1] (values) must have a numeric (non-Boolean) "
                f"dtype, got {val_dtype}"
            )

    # Pre-flight y_format so format failures surface here, not as
    # per-tick FuncFormatter errors at render time.
    try:
        y_format.format(x=0.0, x_k=0.0, x_m=0.0, x_b=0.0)
    except (KeyError, IndexError, ValueError, AttributeError, TypeError) as e:
        raise ValueError(
            f"y_format {y_format!r} is not a valid str.format template: "
            f"{type(e).__name__}: {e}"
        ) from e

    # try/except so a mid-plot failure doesn't leak the global Figure ref.
    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    try:
        for dates, values, label in data_series:
            ax.plot(dates, values, label=label)

        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.legend(loc="center left", bbox_to_anchor=(1.1, 0.5), borderaxespad=0.5)
        ax.tick_params(axis="x", rotation=90)

        formatter = mtick.FuncFormatter(
            lambda x, _pos: y_format.format(x=x, x_k=x / 1e3, x_m=x / 1e6, x_b=x / 1e9)
        )
        ax.yaxis.set_major_formatter(formatter)

        # Mirror labels on right y-axis for readability
        ax_right = ax.twinx()
        ax_right.yaxis.set_major_formatter(formatter)
        ax_right.set_ylim(ax.get_ylim())

        plt.subplots_adjust(right=0.6)
        # Only show() in interactive mode — show() under a GUI backend on
        # headless Linux either raises or hangs cron jobs.
        if plt.isinteractive():
            plt.show()
        # Return OPEN — caller owns lifecycle (savefig on a closed fig
        # is undocumented, backend-dependent).
        return fig
    except Exception:
        plt.close(fig)
        raise



#################################################################################################
# function to create and save multiple matplotlib charts as embedded images in a single html file
# takes write_directory, subfolder and file_name_prefix as parameters to determine where to save the html file


def save_matplotlib_charts_as_html(
    write_directory: str,
    subfolder: str,
    file_name_prefix: str,
    figures: list[Figure],
) -> str:
    """Save pre-built matplotlib figures as embedded images in a single HTML file.

    Side effect: each figure is consumed (``plt.close(fig)`` after
    successful ``savefig``) to release canvas memory in long batch
    jobs. Failure paths leave figures OPEN for the caller to handle.

    Returns absolute path of the written HTML file.
    """
    # ── Step 1: Validate inputs ─────────────────────────────────────────
    if not isinstance(figures, list):
        raise TypeError(
            f"Expected list of matplotlib Figures, got {type(figures).__name__}"
        )
    if not figures:
        raise ValueError("figures list must not be empty")
    for i, fig in enumerate(figures):
        if not isinstance(fig, Figure):
            raise TypeError(
                f"figures[{i}] is {type(fig).__name__}, expected matplotlib Figure"
            )

    _validate_path_components(write_directory, subfolder, file_name_prefix)

    # ── Step 2: Resolve folder path with traversal check ────────────────
    folder_path = _resolve_folder_path(write_directory, subfolder)

    # ── Step 3: Build paths and check lengths ───────────────────────────
    # Timestamp (YYYY-MMDD-HHMM, _REPORT_TZ) baked into the filename so repeat
    # runs do not overwrite prior reports; same TZ as the "Generated:" header
    # below for cross-reference.
    save_ts = datetime.datetime.now(tz=_REPORT_TZ).strftime("%Y-%m%d-%H%M")
    file_name = f"{file_name_prefix}-{save_ts}.html"
    file_path = str(Path(folder_path) / file_name)
    tmp_path = f"{file_path}.{uuid.uuid4().hex[:8]}.tmp"
    _validate_path_lengths(tmp_path)

    # ── Step 4: Create folder with NAS-resilient retry ──────────────────
    _makedirs_with_retry(folder_path)

    # Sweep orphaned .tmp files from prior crashed runs (same 24h threshold
    # and server-time probe as save_results).  Without this, a crashed
    # invocation leaves a <prefix>_charts.html.<uuid>.tmp on the NAS
    # forever — over months on a shared folder these accumulate without
    # bound.  Cleanup is best-effort; any failure is silently suppressed
    # inside the helper.
    # 24h threshold for orphaned .tmp; 92d threshold for finished
    # <prefix>-YYYY-MMDD-HHMM.html outputs (each save now produces a
    # timestamped file rather than overwriting, so prune by age here).
    _cleanup_stale_files(
        folder_path,
        max_age_seconds=86400,
        prefix=file_name_prefix,
        html_output_max_age_seconds=92 * 86400,
    )

    # ── Step 5: Render figures to base64 PNGs ───────────────────────────
    safe_prefix = html.escape(file_name_prefix)
    html_parts: list[str] = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>{safe_prefix} — Charts</title>",
        "<style>body{font-family:sans-serif;margin:20px;} "
        "img{max-width:100%;margin-bottom:30px;}</style>",
        "</head><body>",
        f"<h1>{safe_prefix} — Charts</h1>",
        f"<p style='color:#666;font-size:0.9em;'>Generated: "
        f"{datetime.datetime.now(tz=_REPORT_TZ).strftime('%Y-%m-%d %H:%M:%S')} {_REPORT_TZ_LABEL}</p>",
    ]

    # Track cumulative encoded-image bytes so the size-cap failure names
    # the offending figure (incremental check) rather than silently letting
    # the loop balloon memory and failing only at end-of-loop total.
    cumulative_b64_bytes: int = 0

    for i, fig in enumerate(figures):
        axes_list = fig.get_axes()
        raw_title = axes_list[0].get_title() if axes_list else ""
        safe_title = html.escape(raw_title)

        buf = io.BytesIO()
        try:
            try:
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            except Exception as e:  # noqa: BLE001 — wrap with figure index/title
                raise RuntimeError(
                    f"save_matplotlib_charts_as_html: fig.savefig failed "
                    f"for figures[{i}] (title={raw_title!r}). "
                    f"Underlying error: {type(e).__name__}: {e}"
                ) from e
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode("ascii")
        finally:
            buf.close()
        # Close after savefig succeeds: this function consumes each
        # figure (renders to PNG bytes, embeds in HTML), so eagerly
        # releasing canvas memory keeps long notebook sessions / batch
        # jobs that build dozens of figures from accumulating pyplot
        # state.  Failure paths above re-raise without closing — the
        # caller's exception handler can decide whether to discard.
        plt.close(fig)

        # Enforce the aggregate HTML-size cap incrementally.  Fail-fast
        # with the offending figure's index + title so the caller can
        # act (lower dpi, drop a figure, split into multiple reports)
        # instead of discovering an unrenderable gigabyte HTML on disk.
        cumulative_b64_bytes += len(b64)
        if cumulative_b64_bytes > _MAX_CHARTS_HTML_BYTES:
            raise ValueError(
                f"save_matplotlib_charts_as_html: aggregate encoded image "
                f"size reached {cumulative_b64_bytes / (1024 * 1024):.1f} "
                f"MiB after figures[{i}] (title={raw_title!r}), exceeding "
                f"the "
                f"{_MAX_CHARTS_HTML_BYTES // (1024 * 1024)} MiB cap — "
                f"most browsers refuse to render HTML of this size. "
                f"Remedies: lower matplotlib dpi (this function uses 100), "
                f"split the figures across multiple calls, or raise "
                f"_MAX_CHARTS_HTML_BYTES in shared_util.py if the larger "
                f"single artifact is genuinely needed."
            )

        if safe_title:
            html_parts.append(f"<h2>{safe_title}</h2>")
        html_parts.append(f'<img src="data:image/png;base64,{b64}" alt="{safe_title}">')

    html_parts.append("</body></html>")

    # ── Step 6: Atomic write — tmp then os.replace ──────────────────────
    # Write+fsync+size-verify delegated to _write_bytes_and_fsync (shared
    # with the parquet pipeline's _write_and_fsync via _fsync_and_verify_size).
    # That gives us the 12× fsync retry envelope against AV/DLP locks and
    # tmp cleanup on any failure.
    html_bytes = "\n".join(html_parts).encode("utf-8")
    rename_done = False

    # Distinguish "saved" (new file) vs "updated" (overwrite of existing) for
    # the user-facing message below.  Probed before the rename because once
    # _atomic_replace runs, file_path always exists.  Path.exists() is best-
    # effort: a transient NAS hiccup that returns False on a present file
    # would mislabel the message as "saved" — purely cosmetic, no data risk.
    pre_existed = Path(file_path).exists()

    try:
        tmp_size = _write_bytes_and_fsync(html_bytes, tmp_path)
        _atomic_replace(tmp_path, file_path)
        rename_done = True
        # Post-rename size re-check catches AV/DLP truncation after rename.
        _verify_written_size(file_path, tmp_size)

    except BaseException:
        if not rename_done:
            _unlink_best_effort(tmp_path)
        raise

    action = "updated" if pre_existed else "saved"
    message = f"save_matplotlib_charts_as_html: {action} charts HTML to {file_path}"
    print(message, file=sys.stderr, flush=True)
    logging.getLogger(__name__).info(message)
    return file_path


###################################################################################################
# Sub-functions needed to write parquet files with atomic write and duplicate cleanup
# It is intended to write into shared Isilon network folder
# It also includes security checks to prevent path traversal and ensures the date column is valid for sorting.

## Maximum seconds to wait for acquiring the cross-process file lock before raising Timeout
_LOCK_TIMEOUT_SECONDS: int = 600

## Regex matching characters that are illegal in Windows file/folder names:
## < > : " / \ | ? * and ASCII control characters 0x00-0x1F
_WINDOWS_ILLEGAL_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

## Windows reserved device names: CON, PRN, AUX, NUL, COM0-9, LPT0-9.
## Forbidden as basenames regardless of extension (CON.parquet hangs).
## Per Microsoft's current "Naming Files, Paths, and Namespaces" doc the
## reserved numeric ranges are 0-9 (not 1-9) — COM0 and LPT0 were added
## to the modern list and also reserve the device namespace on Windows 10+.
_WINDOWS_RESERVED_NAMES = re.compile(
    r"^(con|prn|aux|nul|com[0-9]|lpt[0-9])$", re.IGNORECASE
)


def _reject_dot_only_basename(name: str, *, field: str) -> None:
    """Reject names like ``.`` / ``..`` / ``...`` that produce ambiguous
    filenames (``..parquet``, shell-glob hazards, NTFS oddities)."""
    if name and name.strip(".") == "":
        raise ValueError(
            f"{field}={name!r} consists entirely of '.' characters; "
            "use a name with at least one non-dot character."
        )

# zoneinfo on Windows needs the 'tzdata' PyPI pkg; degrade to UTC if missing.
try:
    _REPORT_TZ: datetime.tzinfo = zoneinfo.ZoneInfo("America/New_York")
    _REPORT_TZ_LABEL: str = "US Eastern"
except Exception as _tz_exc:  # noqa: BLE001 — degraded-mode fallback
    _REPORT_TZ = datetime.timezone.utc
    _REPORT_TZ_LABEL = "UTC (install 'tzdata' for US Eastern on Windows)"
    logging.getLogger(__name__).warning(
        "shared_util: could not resolve America/New_York via zoneinfo "
        "(%s: %s); HTML reports will timestamp in UTC.",
        type(_tz_exc).__name__, _tz_exc,
    )
    del _tz_exc


# 100 MiB cap on aggregate HTML — most browsers refuse larger.
_MAX_CHARTS_HTML_BYTES: int = 100 * 1024 * 1024


def _cleanup_stale_files(
    folder_path: str,
    max_age_seconds: int,
    prefix: str,
    *,
    html_output_max_age_seconds: int | None = None,
) -> None:
    """Remove orphaned .tmp / .time_probe_ files older than max_age_seconds,
    and optionally finished ``<prefix>-YYYY-MMDD-HHMM.html`` outputs older
    than html_output_max_age_seconds (opt-in; ``None`` skips that sweep).

    Patterns recognized:
      * save_results           → ``<prefix>.parquet.<uuid8>.tmp``
                              or ``<prefix>-YYYYMM-YYYYMM.parquet.<uuid8>.tmp``
      * save_matplotlib_charts → ``<prefix>-YYYY-MMDD-HHMM.html.<uuid8>.tmp``  (orphan)
                              or ``<prefix>-YYYY-MMDD-HHMM.html``              (finished, opt-in)

    Uses a probe file for the NAS server's current time (avoids client-NAS
    clock drift). If the probe fails, cleanup is aborted (better than
    falling back to local time and risking deletion of active .tmp files).
    All errors are silently suppressed — best-effort, never blocks writes.
    """
    folder = Path(folder_path)
    try:
        probe_path = str(folder / f".time_probe_{uuid.uuid4().hex}")
        probe_created = False
        try:
            fd = os.open(probe_path, os.O_CREAT | os.O_WRONLY, 0o666)
            probe_created = True
            try:
                server_now = os.fstat(fd).st_mtime
            finally:
                os.close(fd)
        except OSError:
            return  # Can't determine NAS time; abort rather than risk false-delete
        finally:
            if probe_created:
                _unlink_best_effort(probe_path)

        # casefold for cross-host NTFS/SMB stability (matches _remove_duplicates).
        prefix_cf = prefix.casefold()
        for entry in folder.iterdir():
            f = entry.name
            f_cf = f.casefold()
            # Strict prefix boundaries prevent cross-pipeline data loss
            # (prefix="sales" must not claim "sales-1-..." or "sales.historical").
            is_stale_tmp = False
            if f_cf.endswith(".tmp") and ".parquet." in f_cf:
                base_name_cf = f_cf.rsplit(".parquet.", 1)[0]
                if base_name_cf == prefix_cf:
                    is_stale_tmp = True
                elif base_name_cf.startswith(f"{prefix_cf}-"):
                    date_part = base_name_cf[len(prefix_cf) + 1 :]
                    # Two ≥6-digit blocks (matches strftime("%Y%m") format).
                    if re.match(r"^[+-]?\d{6,}-[+-]?\d{6,}$", date_part):
                        is_stale_tmp = True
            if not is_stale_tmp and f_cf.endswith(".tmp") and ".html." in f_cf:
                base_name_cf = f_cf.rsplit(".html.", 1)[0]
                # Match save_matplotlib_charts_as_html naming:
                # ``<prefix>-YYYY-MMDD-HHMM.html.<uuid8>.tmp``.
                if base_name_cf.startswith(f"{prefix_cf}-"):
                    ts_part = base_name_cf[len(prefix_cf) + 1 :]
                    if re.match(r"^\d{4}-\d{4}-\d{4}$", ts_part):
                        is_stale_tmp = True
            is_orphaned_probe = f.startswith(".time_probe_")
            # Finished HTML outputs from save_matplotlib_charts_as_html
            # (``<prefix>-YYYY-MMDD-HHMM.html``).  Opt-in via
            # html_output_max_age_seconds — ``None`` skips this sweep so the
            # parquet caller never accidentally deletes chart artifacts.
            is_stale_html_output = False
            if html_output_max_age_seconds is not None and f_cf.endswith(".html"):
                base_name_cf = f_cf[: -len(".html")]
                if base_name_cf.startswith(f"{prefix_cf}-"):
                    ts_part = base_name_cf[len(prefix_cf) + 1 :]
                    if re.match(r"^\d{4}-\d{4}-\d{4}$", ts_part):
                        is_stale_html_output = True
            if not (is_stale_tmp or is_orphaned_probe or is_stale_html_output):
                continue

            if is_stale_html_output and html_output_max_age_seconds is not None:
                age_threshold = html_output_max_age_seconds
            else:
                age_threshold = max_age_seconds
            try:
                if server_now - entry.stat().st_mtime > age_threshold:
                    entry.unlink()
            except OSError:
                pass  # Concurrent removal; ignore
    except OSError:
        pass  # Best-effort; never blocks the caller


def _validate_path_components(
    write_directory: str, subfolder: str, file_name_prefix: str
) -> None:
    """Cross-platform path-shape validation for the two output pipelines."""
    if not write_directory or not write_directory.strip():
        raise ValueError("write_directory must not be empty or whitespace")
    if not Path(write_directory).is_dir():
        raise ValueError(f"write_directory does not exist: {write_directory}")

    if not subfolder or not subfolder.strip():
        raise ValueError("subfolder must not be empty or whitespace")
    _subfolder_parts = subfolder.replace("\\", "/").split("/")
    for _part in _subfolder_parts:
        if _part and _part != _part.strip():
            raise ValueError(
                f"subfolder components must not have leading/trailing whitespace: {subfolder!r}"
            )
        # Reject `.` (silent base-dir collapse) and `..` (traversal); also
        # caught by post-resolve check in _resolve_folder_path, but the
        # error here is clearer.
        if _part in (".", ".."):
            raise ValueError(
                f"subfolder must not contain '.' or '..' components: {subfolder!r}"
            )
        if _part and _WINDOWS_ILLEGAL_CHARS.search(_part):
            raise ValueError(
                f"subfolder contains characters illegal in Windows paths: {subfolder!r}"
            )
        if _part and _WINDOWS_RESERVED_NAMES.match(_part.split(".")[0]):
            raise ValueError(
                f"subfolder contains reserved Windows device name: {_part!r}"
            )
    if not any(_subfolder_parts):
        raise ValueError(
            f"subfolder must contain at least one non-empty path component: {subfolder!r}"
        )

    # Explicit "/" + "\\" rather than os.path.basename (which is OS-specific).
    if (
        not file_name_prefix
        or file_name_prefix != file_name_prefix.strip()
        or "/" in file_name_prefix
        or "\\" in file_name_prefix
    ):
        raise ValueError(
            "file_name_prefix must not be empty, cannot have leading/trailing "
            "whitespace, and cannot contain path separators"
        )
    if _WINDOWS_ILLEGAL_CHARS.search(file_name_prefix):
        raise ValueError(
            f"file_name_prefix contains characters illegal in Windows filenames: {file_name_prefix!r}"
        )
    if _WINDOWS_RESERVED_NAMES.match(file_name_prefix.split(".")[0]):
        raise ValueError(
            f"file_name_prefix uses a reserved Windows device name: {file_name_prefix!r}"
        )
    _reject_dot_only_basename(file_name_prefix, field="file_name_prefix")


def _validate_inputs(
    dataframe: pl.DataFrame, write_directory: str, subfolder: str, file_name_prefix: str
) -> None:
    """Reject LazyFrame, empty df, and bad path shapes."""
    # LazyFrame has no .is_empty(); reject explicitly for actionable error.
    if not isinstance(dataframe, pl.DataFrame):
        raise TypeError(
            f"Expected eager polars.DataFrame, got {type(dataframe).__name__}. "
            f"Call .collect() first."
        )
    if dataframe.is_empty():
        raise ValueError("Cannot save an empty DataFrame")
    _validate_path_components(write_directory, subfolder, file_name_prefix)


def _resolve_folder_path(write_directory: str, subfolder: str) -> str:
    """Resolve target folder + traversal check (subfolder='../../etc' must
    not escape write_directory)."""
    try:
        base_dir = str(Path(write_directory).resolve())
    except OSError as exc:
        raise ValueError(
            f"write_directory could not be resolved (broken symlink, "
            f"missing parent, or permission denied): {write_directory!r} "
            f"({type(exc).__name__}: {exc})"
        ) from exc

    # Normalize "\\" → "/" (Linux: \ is filename-legal) and strip leading "/"
    # so the join is relative, not absolute.
    subfolder_safe = subfolder.replace("\\", "/").lstrip("/")
    if not subfolder_safe:
        raise ValueError(
            f"subfolder resolves to empty after stripping leading slashes: {subfolder!r}"
        )
    try:
        folder_path = str((Path(base_dir) / subfolder_safe).resolve())
    except OSError as exc:
        raise ValueError(
            f"subfolder could not be resolved under {base_dir!r} (broken "
            f"symlink, missing parent, or permission denied): "
            f"{subfolder!r} ({type(exc).__name__}: {exc})"
        ) from exc

    # normcase: case-insensitive on Windows (C:\Foo == c:\foo).
    if not PurePath(os.path.normcase(folder_path)).is_relative_to(
        os.path.normcase(base_dir)
    ):
        raise ValueError(
            f"Security Error: Path traversal detected in subfolder: {subfolder}"
        )

    return folder_path


def _build_filename(
    dataframe: pl.DataFrame, file_name_prefix: str, sort_by_date_column: str | None
) -> tuple[pl.DataFrame, str, re.Pattern[str] | None, bool]:
    """Return (sorted_df, file_name, dup_pattern, exact_match).

    With sort_by_date_column: filename = prefix-YYYYMM-YYYYMM.parquet,
    dup_pattern matches any file with same prefix + start date.
    Without it: filename = prefix.parquet, exact_match=True.
    """
    if sort_by_date_column is not None:
        if sort_by_date_column not in dataframe.columns:
            raise ValueError(
                f"Column '{sort_by_date_column}' not found. Available: {dataframe.columns}"
            )

        col_dtype = dataframe.schema[sort_by_date_column]
        # base_type() strips parameterization: Datetime("us") → Datetime.
        base_dtype = col_dtype.base_type()
        if base_dtype not in (pl.Date, pl.Datetime):
            raise TypeError(
                f"Column '{sort_by_date_column}' expected Date or Datetime, got {col_dtype}"
            )

        if dataframe[sort_by_date_column].null_count() == dataframe.height:
            raise ValueError(f"All values in '{sort_by_date_column}' are null")

        dataframe = dataframe.sort(sort_by_date_column, descending=False)
        first = dataframe.select(
            pl.col(sort_by_date_column).min().dt.strftime("%Y%m")
        ).item()
        last = dataframe.select(
            pl.col(sort_by_date_column).max().dt.strftime("%Y%m")
        ).item()

        if first is None or last is None:
            raise ValueError(
                f"Column '{sort_by_date_column}' min/max resolved to null"
            )

        file_name = f"{file_name_prefix}-{first}-{last}.parquet"

        # casefold (not lower) for locale-independent NTFS/SMB stability;
        # _remove_duplicates folds the listing side identically.
        escaped_prefix = re.escape(file_name_prefix.casefold())
        escaped_first = re.escape(first)
        dup_pattern = re.compile(
            rf"^{escaped_prefix}-{escaped_first}-[+-]?\d{{6,}}\.parquet$"
        )
        return dataframe, file_name, dup_pattern, False
    return dataframe, f"{file_name_prefix}.parquet", None, True


def _remove_duplicates(
    folder_path: str,
    file_name: str,
    file_path: str,
    dup_pattern: re.Pattern[str] | None,
    exact_match: bool,
) -> None:
    """Remove superseded parquet files (called inside the FileLock).
    Date-range mode: same prefix + same start date supersedes older end dates.
    Static mode: case-insensitive exact name match (NTFS/SMB stale listings).
    samefile guards against deleting our own output via case-variant listings;
    read-only attr cleared before unlink (Isilon backup tools)."""
    folder = Path(folder_path)
    # casefold matches _build_filename's fold; locale-independent.
    file_name_cf = file_name.casefold()
    try:
        for entry in folder.iterdir():
            f = entry.name
            if f == file_name or not f.casefold().endswith(".parquet"):
                continue

            fl = f.casefold()
            if exact_match:
                is_dup = fl == file_name_cf
            else:
                is_dup = dup_pattern is not None and bool(dup_pattern.match(fl))

            if is_dup:
                try:
                    # samefile guards against case-variant listing of self.
                    is_same_file = False
                    if entry.exists():
                        try:
                            is_same_file = entry.samefile(file_path)
                        except OSError:
                            is_same_file = fl == file_name_cf
                    if is_same_file:
                        continue
                    try:
                        dattrs = entry.stat().st_mode
                        if not (dattrs & stat.S_IWRITE):
                            entry.chmod(dattrs | stat.S_IWRITE)
                    except OSError:
                        pass  # unlink below will surface the real error
                    entry.unlink()
                    logging.getLogger(__name__).info(
                        "save_results: erased superseded duplicate %s", f
                    )
                except OSError:
                    pass  # File in use / AV-locked; skip
    except OSError as e:
        logging.getLogger(__name__).warning(
            "save_results: could not complete duplicate cleanup in %s: %s",
            folder_path, e,
        )


def _validate_path_lengths(tmp_path: str) -> None:
    """Reject basenames > 255 bytes (POSIX) and full paths >= 260 chars
    on Windows (MAX_PATH, unless ``\\\\?\\`` extended prefix)."""
    _MAX_BASENAME_BYTES = 255  # UTF-8 bytes (multi-byte chars can blow len())
    tmp_basename_bytes = len(Path(tmp_path).name.encode("utf-8"))
    if tmp_basename_bytes > _MAX_BASENAME_BYTES:
        raise ValueError(
            f"Generated filename exceeds {_MAX_BASENAME_BYTES}-byte POSIX limit: "
            f"{tmp_basename_bytes} bytes. Shorten file_name_prefix."
        )
    _WIN_MAX_PATH = 260
    if (
        os.name == "nt"
        and len(tmp_path) >= _WIN_MAX_PATH
        and not tmp_path.startswith("\\\\?\\")
    ):
        raise ValueError(
            f"Generated path ({len(tmp_path)} chars) exceeds Windows MAX_PATH limit ({_WIN_MAX_PATH})."
        )


def _makedirs_with_retry(folder_path: str) -> None:
    """mkdir with 12× transient-lock retry (0.5s..15s backoff). On
    Access Denied, clear the read-only attribute on the closest
    existing ancestor (Isilon backup/compliance tools)."""
    target = Path(folder_path)
    for _mkdir_attempt in range(12):
        try:
            target.mkdir(parents=True, exist_ok=True)
            return
        except OSError as e:
            if not _is_transient_lock(e):
                raise
            if _mkdir_attempt == 11:
                raise
            if _is_access_denied(e):
                ancestor = target
                _walk_limit = 64
                while _walk_limit > 0 and not ancestor.exists():
                    parent = ancestor.parent
                    if parent == ancestor:
                        break
                    ancestor = parent
                    _walk_limit -= 1
                if ancestor.exists():
                    _clear_readonly_attr(ancestor)
            time.sleep(min(15.0, 0.5 * (2**_mkdir_attempt)))


def _fsync_and_verify_size(tmp_path: str) -> int:
    """Reopen tmp_path with O_RDWR, fsync, return size via os.fstat.

    os.fstat on the open handle bypasses the SMB directory metadata cache.
    12× retry on transient AV/DLP/SMB locks. On any failure the partial
    tmp file is removed before propagating.
    """
    tmp_size = -1
    for _fsync_attempt in range(12):
        try:
            fd = os.open(tmp_path, os.O_RDWR | getattr(os, "O_BINARY", 0))
            try:
                os.fsync(fd)
                tmp_size = os.fstat(fd).st_size
            finally:
                os.close(fd)
            break
        except OSError as e:
            if _is_access_denied(e):
                _clear_readonly_attr(Path(tmp_path))
            if not _is_transient_lock(e):
                _unlink_best_effort(tmp_path)
                raise
            if _fsync_attempt == 11:
                _unlink_best_effort(tmp_path)
                raise
            time.sleep(min(15.0, 0.5 * (2**_fsync_attempt)))

    if tmp_size <= 0:
        _unlink_best_effort(tmp_path)
        raise OSError(
            f"fsync produced a zero-byte or unmeasured file: {tmp_path}"
        )
    return tmp_size


def _write_and_fsync(dataframe: pl.DataFrame, file_path: str) -> tuple[str, int]:
    """Stage DataFrame as parquet next to ``file_path`` via
    ``tempfile.mkstemp`` (atomic O_EXCL creation, no UUID-collision
    window), retry on transient NAS OSError (4× with 1s/2s/4s backoff),
    then fsync via ``_fsync_and_verify_size``. Returns
    ``(tmp_path, verified_size)``; caller commits via ``_atomic_replace``.

    ``mkstemp`` creates the file with mode 0o600; on POSIX we relax to
    0o666 (umask-honored on the rename target) so a shared-NAS service
    account doesn't ship a more restrictive mode than the prior
    ``write_parquet``-from-scratch behavior. Windows ignores Unix bits.

    Polars/Arrow non-OSError exceptions wrap to RuntimeError with context.
    Partial tmp file cleaned up on any failure before propagating.
    """
    folder = str(Path(file_path).parent)
    base = Path(file_path).name
    fd, tmp_path = tempfile.mkstemp(
        prefix=f"{base}.", suffix=".tmp", dir=folder,
    )
    os.close(fd)
    if os.name == "posix":
        try:
            Path(tmp_path).chmod(0o666)
        except OSError:
            pass  # best-effort; non-fatal mode mismatch on rename target

    for _write_attempt in range(4):
        try:
            dataframe.write_parquet(tmp_path)
            break
        except OSError:
            _unlink_best_effort(tmp_path)
            if _write_attempt == 3:
                raise
            time.sleep(min(15.0, 1.0 * (2**_write_attempt)))
        except Exception as e:  # noqa: BLE001 — wrap polars/arrow errors with context
            _unlink_best_effort(tmp_path)
            raise RuntimeError(
                f"_write_and_fsync: polars write_parquet failed "
                f"(tmp_path={tmp_path!r}, n_rows={dataframe.height}, "
                f"n_cols={dataframe.width}). Underlying error: "
                f"{type(e).__name__}: {e}"
            ) from e

    return tmp_path, _fsync_and_verify_size(tmp_path)


def _write_bytes_and_fsync(data: bytes | bytearray | memoryview, tmp_path: str) -> int:
    """Write a byte buffer atomically to tmp_path, fsync, return verified size.

    Bytes-equivalent of _write_and_fsync for non-parquet outputs (HTML reports,
    JSON sidecars, etc.).  Uses the same two-phase retry envelope:
      * Write phase: up to 4× on transient OSErrors (1s/2s/4s backoff).
      * Fsync phase: up to 12× on transient AV/DLP locks (0.5s … 15s cap) via
        the shared _fsync_and_verify_size helper.

    Opens with O_BINARY on Windows (no CRLF translation) and an explicit os.write
    loop — os.write may return fewer bytes than requested (POSIX spec), and a
    zero-byte return after partial progress indicates disk-full / broken pipe.

    Returns the verified size in bytes.  Raises OSError with ``size mismatch``
    in the message if fsync reports a size different from ``len(data)`` (could
    happen on a NAS with write-behind caching bugs).  On any failure, the
    partial tmp file is cleaned up before propagating.

    Raises
    ------
    OSError
        * write phase fails 4× consecutively
        * fsync phase fails (delegated to _fsync_and_verify_size)
        * final file size != len(data)
    """
    expected_size = len(data)

    # Write phase: retry up to 4× for transient OSErrors.
    for _write_attempt in range(4):
        try:
            # O_BINARY prevents \n→\r\n translation on Windows.
            # On POSIX the constant is absent, fall back to 0 (no-op).
            fd = os.open(
                tmp_path,
                os.O_CREAT | os.O_WRONLY | os.O_TRUNC | getattr(os, "O_BINARY", 0),
                0o666,
            )
            try:
                # os.write may return fewer bytes than requested (POSIX spec).
                # Loop until all bytes flushed, or raise on zero-write.
                view = memoryview(data)
                written_total = 0
                while written_total < expected_size:
                    written = os.write(fd, view[written_total:])
                    if written == 0:
                        raise OSError(
                            f"os.write returned 0 bytes after "
                            f"{written_total}/{expected_size} — disk may be full"
                        )
                    written_total += written
            finally:
                os.close(fd)
            break  # Write succeeded
        except OSError:
            # Remove partial tmp file before retrying.
            _unlink_best_effort(tmp_path)
            if _write_attempt == 3:
                raise  # Exhausted write retries
            time.sleep(min(15.0, 1.0 * (2**_write_attempt)))  # Backoff: 1s, 2s, 4s

    # Fsync phase: delegated to shared helper.  Partial-tmp cleanup on failure
    # is handled by the helper.
    tmp_size = _fsync_and_verify_size(tmp_path)

    # Size verification against caller-provided expectation.
    if tmp_size != expected_size:
        _unlink_best_effort(tmp_path)
        raise OSError(
            f"Post-write size mismatch: expected {expected_size} bytes, "
            f"got {tmp_size} bytes for {tmp_path}"
        )
    return tmp_size


def _atomic_replace(tmp_path: str, file_path: str) -> None:
    """Atomically rename tmp_path to file_path. 12× transient-lock retry
    (NAS / SMB / NFS) with read-only-attribute clearing on Access Denied.
    POSIX-only parent-dir fsync after successful rename for durability."""
    file_p = Path(file_path)
    for _attempt in range(12):
        try:
            Path(tmp_path).replace(file_path)
            _fsync_parent_dir(file_p)
            return
        except OSError as e:
            if not _is_transient_lock(e):
                raise
            if _attempt == 11:
                raise
            if _is_access_denied(e) and file_p.is_file():
                _clear_readonly_attr(file_p)
            time.sleep(min(15.0, 0.5 * (2**_attempt)))


def _fsync_parent_dir(file_p: Path) -> None:
    """POSIX-only parent-dir fsync (Windows ReplaceFile flushes implicitly).
    Best-effort; OSError swallowed since the file's own fsync already
    persisted the data."""
    o_directory = getattr(os, "O_DIRECTORY", None)
    if o_directory is None:
        return
    parent = file_p.parent
    fd: int | None = None
    try:
        fd = os.open(str(parent), os.O_RDONLY | o_directory)
        os.fsync(fd)
    except OSError:
        pass  # Best-effort; file fsync already committed the data
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass  # fd already closed


def _verify_written_size(file_path: str, expected_size: int) -> None:
    """Verify post-rename file size via os.fstat on an open handle
    (bypasses SMB cache). 12× retry on transient locks. Outcome:
        ==expected: OK
        > expected: log + return (concurrent overwrite, validated by writer)
        < expected: raise (external truncation, partial artifact)
    """
    written_size = -1
    for _verify_attempt in range(12):
        try:
            fd = os.open(file_path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
            try:
                written_size = os.fstat(fd).st_size
            finally:
                os.close(fd)
            if written_size == expected_size:
                return
            if written_size > 0:
                break  # fstat is authoritative; mismatch is real, not stale
        except OSError as e:
            if not _is_transient_lock(e):
                raise
            if _verify_attempt == 11:
                raise
        time.sleep(min(15.0, 0.5 * (2**_verify_attempt)))

    if written_size > expected_size:
        # Concurrent writer overwrote with different (larger) compression.
        logging.getLogger(__name__).info(
            "_verify_written_size: written size (%d) exceeds expected (%d) "
            "for %s — consistent with concurrent overwrite.",
            written_size, expected_size, file_path,
        )
        return

    raise OSError(
        f"Post-write verification failed: expected {expected_size} bytes, "
        f"got {written_size} bytes for {file_path}"
    )


###################################################################################################
# This is the main function to save the results to parquet files with atomic write and duplicate cleanup.


def save_results(
    dataframe: pl.DataFrame,
    write_directory: str,
    subfolder: str,
    file_name_prefix: str,
    sort_by_date_column: str | None = None,
) -> str:
    """Write a Polars DataFrame to parquet on a shared Isilon network drive.

    Atomic-write: write tmp, fsync, FileLock-guarded os.replace, size verify.

    Args:
        dataframe:          Polars DataFrame to write (must not be empty).
        write_directory:    Root directory that must already exist on disk.
        subfolder:          Relative subfolder under write_directory (created if needed).
        file_name_prefix:   Base name for the parquet file (no extension, no separators).
        sort_by_date_column: Optional Date/Datetime column to sort by and embed in filename.

    Returns:
        The absolute path of the written parquet file.

    Idempotency: the final filename is a pure function of
    ``(file_name_prefix, sort_by_date_column range)``. Two workers with
    the same tuple race for the same target path; ``os.replace`` accepts
    whichever commits last. Distinguish writers at the call site
    (per-worker prefix / subfolder / orchestrator coordination) if that
    matters. The FileLock guards only rename + duplicate cleanup, not
    the parquet write itself.
    """
    _validate_inputs(dataframe, write_directory, subfolder, file_name_prefix)
    folder_path = _resolve_folder_path(write_directory, subfolder)

    dataframe, file_name, dup_pattern, exact_match = _build_filename(
        dataframe, file_name_prefix, sort_by_date_column
    )
    folder = Path(folder_path)
    file_path = str(folder / file_name)
    lock_path = str(folder / f".{file_name_prefix}.lock")

    # Validate the worst-case tmp path before any I/O: mkstemp generates
    # ``<base>.<8-char-random>.tmp`` so reserve 13 chars on top of file_path.
    _validate_path_lengths(f"{file_path}.{'X' * 8}.tmp")
    _makedirs_with_retry(folder_path)
    _cleanup_stale_files(folder_path, max_age_seconds=86400, prefix=file_name_prefix)

    # Detect predecessor for "saved" vs "updated" message label (cosmetic).
    pre_existed_exact = Path(file_path).exists()
    pre_existed_predecessor = False
    if not pre_existed_exact and not exact_match and dup_pattern is not None:
        try:
            for _entry in Path(folder_path).iterdir():
                if dup_pattern.match(_entry.name.casefold()):
                    pre_existed_predecessor = True
                    break
        except OSError:
            pass  # Best-effort; defaults to "saved" label
    will_overwrite = pre_existed_exact or pre_existed_predecessor

    start_time = time.monotonic()
    rename_done = False
    tmp_path: str | None = None

    try:
        # Write parquet OUTSIDE the lock so concurrent workers writing
        # different date ranges can stream in parallel.
        tmp_path, tmp_size = _write_and_fsync(dataframe, file_path)

        # On POSIX, pre-create the lock file with 0o666 so different
        # service accounts can acquire the byte lock; umask race is
        # handled by the PermissionError retry below.
        if os.name == "posix":
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY, 0o666)
                try:
                    os.close(fd)
                finally:
                    Path(lock_path).chmod(0o666)
            except OSError:
                pass  # Best-effort; FileLock will report the real error

        lock = FileLock(lock_path, timeout=_LOCK_TIMEOUT_SECONDS)
        _lock_start = time.monotonic()
        while True:
            _lock_acquired = False
            try:
                with lock:
                    _lock_acquired = True
                    _atomic_replace(tmp_path, file_path)
                    rename_done = True
                    _remove_duplicates(
                        folder_path, file_name, file_path, dup_pattern, exact_match
                    )
                break
            except PermissionError:
                if _lock_acquired:
                    # PermissionError came from inside the lock body (real).
                    raise
                # umask race on lock-file create; retry until timeout.
                if time.monotonic() - _lock_start > _LOCK_TIMEOUT_SECONDS:
                    raise TimeoutError(
                        f"Cannot acquire lock file after "
                        f"{_LOCK_TIMEOUT_SECONDS}s: {lock_path}"
                    ) from None
                time.sleep(0.1)

        # Verify size OUTSIDE the lock to avoid AV-scanner priority inversion.
        _verify_written_size(file_path, tmp_size)

    finally:
        if not rename_done and tmp_path is not None:
            _unlink_best_effort(tmp_path)

    elapsed = (time.monotonic() - start_time) / 60
    action = "updated" if will_overwrite else "saved"
    message = (
        f"save_results: {action} parquet to {file_path}; "
        f"time taken (minutes): {elapsed:.2f}"
    )
    print(message, file=sys.stderr, flush=True)
    logging.getLogger(__name__).info(message)
    return file_path

#################################################################################################
# Function for reading Amazon Redshift data into a Polars DataFrame, with robust input validation and error handling.


class RedshiftIngestError(RuntimeError):
    """Boundary-contract failure while ingesting from Amazon Redshift."""


class _ArrowBatchReader(Protocol):
    """Structural type for the ``arrow_odbc`` BatchReader surface we use.

    ``arrow_odbc`` is imported as untyped (``type: ignore[import-untyped]``
    above), so referencing its ``BatchReader`` class by name would cascade
    ``Any`` through every helper that touches it and fail ``ruff``'s
    ``ANN401`` under strict typing. Declaring just the two attributes we
    actually consume (``schema`` and iteration yielding
    ``pa.RecordBatch``) keeps the typed surface honest and minimizes
    coupling to upstream internals.
    """

    @property
    def schema(self) -> pa.Schema: ...

    def __iter__(self) -> Iterator[pa.RecordBatch]: ...


# =============================================================================
# Module-level constants
# =============================================================================

# Boundary validators bound to ``RedshiftIngestError``.  See
# ``_BoundaryChecker`` at module top — one implementation, parameterized
# on the exception class.  ``_CONN_STR_FORBIDDEN`` / ``_MAX_BATCH_SIZE`` /
# ``_MAX_TIMEOUT_SEC`` / ``_MAX_ROWS_CAP`` / ``_MAX_VALUE_BYTES`` are
# defined once at module top.
_rs_check = _BoundaryChecker(RedshiftIngestError)

# Characters forbidden inside an individual DbGroups entry. We join groups
# with "," to form the DbGroups value, so a comma inside a name would be
# read as two groups by the driver. Whitespace would silently fail the
# server-side group lookup with a confusing "group not found" error, so
# reject at the boundary.
_DB_GROUP_FORBIDDEN = (",", ";", "{", "}", "\x00", "\n", "\r", " ", "\t")

_MAX_DB_GROUPS = 256  # Redshift has no documented hard limit; this catches typos

# Whitelist of SSLMode values supported by the Amazon Redshift ODBC driver.
# A whitelist is required because the value is interpolated into the
# connection string; allowing arbitrary text would reopen the conn-string
# injection surface that _CONN_STR_FORBIDDEN locks down. Values are
# case-sensitive (the driver matches the canonical lowercase forms below).
_SSL_MODES = frozenset(
    {"verify-full", "verify-ca", "require", "prefer", "disable"}
)

# Fields whose values must NEVER appear in an error message. A wrong-type
# value here could be a bytes/bytearray-wrapped password that would
# otherwise be printed verbatim via {value!r}.
_SENSITIVE_CRED_FIELDS = frozenset(
    {
        "password",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
    }
)

# Unicode codepoints forbidden inside ``query`` because they either
# (a) let a multi-line statement slip past a single-statement tokenizer,
# or (b) make the reviewed form of the query diverge from the executed
# form ("Trojan Source" — CVE-2021-42574).
#
#   * U+2028 LINE SEPARATOR / U+2029 PARAGRAPH SEPARATOR: some
#     ODBC/JDBC driver stacks treat these as line terminators inside
#     their SQL comment / string-literal state machine, letting a
#     ``-- comment<U+2028>INJECTED`` payload execute after the review
#     sees only the comment.
#   * U+202A..U+202E (LRE, RLE, PDF, LRO, RLO) and U+2066..U+2069
#     (LRI, RLI, FSI, PDI): bidirectional override / isolate controls.
#     A reviewer sees the visually-reordered query; the driver executes
#     the logical-order one. Classic source-trojaning vector.
_QUERY_FORBIDDEN_CODEPOINTS = frozenset(
    [chr(0x2028), chr(0x2029)]
    + [chr(cp) for cp in range(0x202A, 0x202F)]
    + [chr(cp) for cp in range(0x2066, 0x206A)]
)


# =============================================================================
# Per-field validators (called from the main function)
# =============================================================================


def _validate_endpoint_identifiers(
    *, host: str, database: str, driver: str
) -> None:
    _rs_check.conn_token("host", host)
    _rs_check.conn_token("database", database)
    _rs_check.conn_token("driver", driver)


def _validate_query(query: str) -> None:
    if not isinstance(query, str) or not query.strip():
        raise RedshiftIngestError(
            "query must be a non-empty, non-whitespace str"
        )
    if "\x00" in query:
        raise RedshiftIngestError("query contains a null byte")
    # ``chr(0xFEFF)`` (not a raw literal) survives a UTF-16 re-save.
    if chr(0xFEFF) in query:
        raise RedshiftIngestError(
            "query contains a UTF-8 BOM (U+FEFF); re-save the source "
            "file as plain UTF-8, or strip via query.replace('\\ufeff', '')"
        )
    for bad in _QUERY_FORBIDDEN_CODEPOINTS:
        if bad in query:
            raise RedshiftIngestError(
                f"query contains Unicode codepoint U+{ord(bad):04X} "
                "(LINE/PARAGRAPH SEP or bidi override/isolate — "
                "CVE-2021-42574 'Trojan Source'). Strip at source."
            )


def _validate_numeric_params(
    *,
    port: int,
    batch_size: int,
    login_timeout_sec: int,
    query_timeout_sec: int | None,
    max_rows: int | None,
    max_text_size: int | None,
    max_binary_size: int | None,
) -> int | None:
    """Bounds-check int params; return validated ``max_rows`` (the only
    one the streaming loop needs back). Timeout / max_rows reject ``0``
    (ODBC's "no timeout" sentinel — pass ``None`` for explicit no-cap)."""
    _rs_check.int_("port", port, minv=1, maxv=65_535)
    _rs_check.int_("batch_size", batch_size, minv=1, maxv=_MAX_BATCH_SIZE)
    _rs_check.int_(
        "login_timeout_sec", login_timeout_sec, minv=1, maxv=_MAX_TIMEOUT_SEC,
    )
    _rs_check.int_(
        "query_timeout_sec", query_timeout_sec,
        minv=1, maxv=_MAX_TIMEOUT_SEC, allow_none=True,
    )
    max_rows_val = _rs_check.int_(
        "max_rows", max_rows, minv=1, maxv=_MAX_ROWS_CAP, allow_none=True
    )
    _rs_check.int_(
        "max_text_size", max_text_size,
        minv=1, maxv=_MAX_VALUE_BYTES, allow_none=True,
    )
    _rs_check.int_(
        "max_binary_size", max_binary_size,
        minv=1, maxv=_MAX_VALUE_BYTES, allow_none=True,
    )
    return max_rows_val


def _validate_bool_flags(
    *,
    require_non_empty: bool,
    allow_extra_columns: bool,
    require_tz_aware_timestamps: bool,
    require_finite_numerics: bool,
    iam: bool,
    auto_create: bool,
) -> None:
    _rs_check.bool_("require_non_empty", require_non_empty)
    _rs_check.bool_("allow_extra_columns", allow_extra_columns)
    _rs_check.bool_("require_tz_aware_timestamps", require_tz_aware_timestamps)
    _rs_check.bool_("require_finite_numerics", require_finite_numerics)
    _rs_check.bool_("iam", iam)
    _rs_check.bool_("auto_create", auto_create)


def _validate_optional_credential_strings(
    fields: dict[str, str | None],
) -> None:
    """Two-pass: types first, then content (preserves error ordering —
    a type fault surfaces before content faults on other fields)."""
    for name, value in fields.items():
        _rs_check.optional_str(
            name, value, sensitive=name in _SENSITIVE_CRED_FIELDS
        )
    for name, value in fields.items():
        if value is None:
            continue
        if not value:
            raise RedshiftIngestError(
                f"{name} must be non-empty when provided"
            )
        _rs_check.no_control_chars(name, value)
        for ch in _CONN_STR_FORBIDDEN:
            if ch in value:
                raise RedshiftIngestError(
                    f"{name} contains forbidden character {ch!r}"
                )
        # Floor matches _redact_secrets: shorter secrets bypass redaction.
        if (
            name in _SENSITIVE_CRED_FIELDS
            and len(value) < _MIN_SECRET_REDACTION_LEN
        ):
            raise RedshiftIngestError(
                f"{name} must be at least {_MIN_SECRET_REDACTION_LEN} chars; "
                "shorter values cannot be safely redacted from driver errors."
            )


def _validate_conn_string_identifier_fields(
    *,
    aws_profile: str | None,
    aws_region: str | None,
    cluster_id: str | None,
    db_user: str | None,
) -> None:
    """Reject ``=`` and ASCII whitespace in fields that interpolate as
    ODBC ``KEY=<val>`` segments."""
    forbidden = ("=", " ", "\t")
    for name, value in (
        ("aws_profile", aws_profile),
        ("aws_region", aws_region),
        ("cluster_id", cluster_id),
        ("db_user", db_user),
    ):
        if value is None:
            continue
        for ch in forbidden:
            if ch in value:
                raise RedshiftIngestError(
                    f"{name}={value!r} contains forbidden character "
                    f"{ch!r}; conn-string identifier fields must not "
                    "contain '=' or whitespace"
                )


def _validate_ssl_mode(ssl_mode: str | None) -> None:
    # Type check already happened in the credential-string sweep; this
    # only validates the whitelist membership.
    if ssl_mode is not None and ssl_mode not in _SSL_MODES:
        raise RedshiftIngestError(
            f"ssl_mode={ssl_mode!r} not in whitelist; "
            f"allowed: {sorted(_SSL_MODES)}"
        )


def _validate_parameters(
    parameters: Sequence[str | None] | None,
) -> list[str | None] | None:
    if parameters is None:
        return None
    _rs_check.list_or_tuple("parameters", parameters)
    params = list(parameters)
    for i, p in enumerate(params):
        if p is not None and not isinstance(p, str):
            raise RedshiftIngestError(
                f"parameters[{i}] must be str or None, got "
                f"{type(p).__name__}={p!r}. arrow_odbc binds ODBC "
                "parameters as strings; convert numbers via str() at the "
                "call site."
            )
    return params


def _validate_expected_columns(
    expected_columns: Sequence[str] | None,
) -> list[str] | None:
    if expected_columns is None:
        return None
    _rs_check.list_or_tuple("expected_columns", expected_columns)
    out = list(expected_columns)
    for i, c in enumerate(out):
        if not isinstance(c, str) or not c:
            raise RedshiftIngestError(
                f"expected_columns[{i}] must be non-empty str, got "
                f"{type(c).__name__}={c!r}"
            )
    dupes = _rs_check.find_duplicates(out)
    if dupes:
        raise RedshiftIngestError(
            f"expected_columns contains duplicates: {dupes}"
        )
    return out


def _validate_db_groups(
    db_groups: Sequence[str] | None,
    *,
    iam: bool,
) -> str | None:
    """Return the comma-joined DbGroups value, or None to omit the key.
    Per-entry char check + boundary dedup-reject (Redshift silently
    dedupes, hiding caller copy-paste bugs). Requires iam=True."""
    if db_groups is None:
        return None
    _rs_check.list_or_tuple("db_groups", db_groups)
    groups = list(db_groups)
    if not groups:
        # Empty list is almost always a config bug; pass None to omit deliberately.
        raise RedshiftIngestError(
            "db_groups was provided as an empty list/tuple; pass None to "
            "omit DbGroups, or supply at least one group name."
        )
    if len(groups) > _MAX_DB_GROUPS:
        raise RedshiftIngestError(
            f"db_groups has {len(groups)} entries, exceeds cap "
            f"{_MAX_DB_GROUPS}"
        )
    for i, g in enumerate(groups):
        if not isinstance(g, str) or not g:
            raise RedshiftIngestError(
                f"db_groups[{i}] must be non-empty str, got "
                f"{type(g).__name__}={g!r}"
            )
        for ch in _DB_GROUP_FORBIDDEN:
            if ch in g:
                raise RedshiftIngestError(
                    f"db_groups[{i}]={g!r} contains forbidden character "
                    f"{ch!r}"
                )
    dupes = _rs_check.find_duplicates(groups)
    if dupes:
        raise RedshiftIngestError(
            f"db_groups contains duplicates: {dupes}"
        )
    if not iam:
        raise RedshiftIngestError(
            "db_groups requires iam=True (groups are assigned by "
            "GetClusterCredentials)"
        )
    return ",".join(groups)


def _validate_iam_auth(
    *,
    iam: bool,
    user: str | None,
    password: str | None,
    aws_profile: str | None,
    aws_access_key_id: str | None,
    aws_secret_access_key: str | None,
    aws_session_token: str | None,
    aws_region: str | None,
    cluster_id: str | None,
    db_user: str | None,
    auto_create: bool,
) -> None:
    """Cross-field native-vs-IAM validation. Native: reject IAM-only
    fields (catch half-migrated call sites). IAM: pick one credential
    source (profile XOR keys XOR default chain); reject native UID/PWD
    (driver overrides with temp password from GetClusterCredentials)."""
    iam_only_fields = {
        "aws_profile": aws_profile,
        "aws_access_key_id": aws_access_key_id,
        "aws_secret_access_key": aws_secret_access_key,
        "aws_session_token": aws_session_token,
        "aws_region": aws_region,
        "cluster_id": cluster_id,
        "db_user": db_user,
    }
    if not iam:
        leaked = [k for k, v in iam_only_fields.items() if v is not None]
        if auto_create:
            leaked.append("auto_create")
        if leaked:
            raise RedshiftIngestError(
                f"iam=False but IAM-only field(s) supplied: {sorted(leaked)}. "
                "Set iam=True to use IAM federated auth, or remove these."
            )
        if user is None or password is None:
            raise RedshiftIngestError(
                "iam=False requires both user and password (Redshift "
                "native auth)"
            )
        return

    # iam=True
    if user is not None or password is not None:
        raise RedshiftIngestError(
            "iam=True must not be combined with user/password; the "
            "Amazon Redshift ODBC driver mints a temp password via "
            "GetClusterCredentials. Pass db_user instead."
        )
    if db_user is None:
        raise RedshiftIngestError(
            "iam=True requires db_user (the Redshift database user that "
            "GetClusterCredentials will issue a temporary password for)"
        )
    explicit_keys = (
        aws_access_key_id is not None
        or aws_secret_access_key is not None
        or aws_session_token is not None
    )
    if aws_profile is not None and explicit_keys:
        raise RedshiftIngestError(
            "iam=True: aws_profile and explicit AWS keys are mutually "
            "exclusive credential sources; pick one"
        )
    if aws_access_key_id is not None and aws_secret_access_key is None:
        raise RedshiftIngestError(
            "iam=True: aws_access_key_id requires aws_secret_access_key"
        )
    if aws_secret_access_key is not None and aws_access_key_id is None:
        raise RedshiftIngestError(
            "iam=True: aws_secret_access_key requires aws_access_key_id"
        )
    if aws_session_token is not None and (
        aws_access_key_id is None or aws_secret_access_key is None
    ):
        raise RedshiftIngestError(
            "iam=True: aws_session_token must accompany "
            "aws_access_key_id+aws_secret_access_key (temporary STS creds)"
        )


# =============================================================================
# Connection-string composition
# =============================================================================


def _endpoint_segment(
    *, driver: str, host: str, port: int, database: str
) -> list[str]:
    # Brace-quote DRIVER / SERVER / DATABASE so an unusual char inside
    # the value (e.g. an `=` in a CNAME alias) cannot be re-parsed as a
    # KEY=VALUE boundary by a permissive driver. `_CONN_STR_FORBIDDEN`
    # already rejects `}`, so the closing brace is closed-form safe.
    return [
        f"DRIVER={{{driver}}}",
        f"SERVER={{{host}}}",
        f"PORT={port}",
        f"DATABASE={{{database}}}",
    ]


def _native_auth_segment(*, user: str, password: str) -> list[str]:
    return [f"UID={user}", f"PWD={password}"]


def _iam_auth_segment(
    *,
    db_user: str,
    db_groups_value: str | None,
    auto_create: bool,
    cluster_id: str | None,
    aws_region: str | None,
    aws_profile: str | None,
    aws_access_key_id: str | None,
    aws_secret_access_key: str | None,
    aws_session_token: str | None,
) -> list[str]:
    """Build the IAM-mode connection-string segment.

    Every value here was character-checked by
    ``_validate_optional_credential_strings`` and the credential-source
    XOR was enforced by ``_validate_iam_auth``; interpolation is safe.
    """
    parts: list[str] = ["IAM=1", f"DbUser={db_user}"]
    if db_groups_value is not None:
        parts.append(f"DbGroups={db_groups_value}")
    if auto_create:
        parts.append("AutoCreate=1")
    if cluster_id is not None:
        parts.append(f"ClusterID={cluster_id}")
    if aws_region is not None:
        parts.append(f"Region={aws_region}")
    if aws_profile is not None:
        parts.append(f"Profile={aws_profile}")
    if aws_access_key_id is not None:
        # _validate_iam_auth pairs access_key_id with secret_access_key.
        parts.append(f"AccessKeyID={aws_access_key_id}")
        parts.append(f"SecretAccessKey={cast(str, aws_secret_access_key)}")
        if aws_session_token is not None:
            parts.append(f"SessionToken={aws_session_token}")
    return parts


def _build_connection_string(
    *,
    driver: str,
    host: str,
    port: int,
    database: str,
    iam: bool,
    user: str | None,
    password: str | None,
    db_user: str | None,
    db_groups_value: str | None,
    auto_create: bool,
    cluster_id: str | None,
    aws_region: str | None,
    aws_profile: str | None,
    aws_access_key_id: str | None,
    aws_secret_access_key: str | None,
    aws_session_token: str | None,
    ssl_mode: str | None,
) -> str:
    parts = _endpoint_segment(
        driver=driver, host=host, port=port, database=database
    )
    if iam:
        # _validate_iam_auth narrowed db_user non-None for iam=True; cast
        # without re-checking (perimeter discipline — CLAUDE.md).
        parts.extend(
            _iam_auth_segment(
                db_user=cast(str, db_user),
                db_groups_value=db_groups_value,
                auto_create=auto_create,
                cluster_id=cluster_id,
                aws_region=aws_region,
                aws_profile=aws_profile,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
            )
        )
    else:
        # _validate_iam_auth narrowed user/password non-None for iam=False.
        parts.extend(
            _native_auth_segment(user=cast(str, user), password=cast(str, password))
        )
    if ssl_mode is not None:
        parts.append(f"SSLMode={ssl_mode}")
    return ";".join(parts) + ";"


# =============================================================================
# arrow_odbc reader: open, schema, stream
# =============================================================================


def _open_arrow_reader(
    *,
    conn_str: str,
    query: str,
    parameters: list[str | None] | None,
    batch_size: int,
    login_timeout_sec: int,
    query_timeout_sec: int | None,
    max_text_size: int | None,
    max_binary_size: int | None,
    user: str | None,
    password: str | None,
    iam: bool,
    host: str,
    database: str,
    secrets: tuple[str, ...],
) -> _ArrowBatchReader:
    """Open the streaming reader; never returns None. iam=True doesn't
    forward user/password (IAM mints them via GetClusterCredentials).
    ``secrets`` are redacted from any echoed driver error."""
    odbc_user = None if iam else user
    odbc_password = None if iam else password
    try:
        reader = read_arrow_batches_from_odbc(
            connection_string=conn_str,
            query=query,
            batch_size=batch_size,
            user=odbc_user,
            password=odbc_password,
            parameters=parameters,
            login_timeout_sec=login_timeout_sec,
            query_timeout_sec=query_timeout_sec,
            max_text_size=max_text_size,
            max_binary_size=max_binary_size,
        )
    except TypeError as exc:
        # arrow_odbc kw-arg evolution; TypeError doesn't echo conn_str.
        raise RedshiftIngestError(
            f"arrow_odbc rejected a keyword argument "
            f"(likely a version mismatch): "
            f"{_redact_secrets(str(exc), secrets)}"
        ) from exc
    except Exception as exc:  # arrow_odbc raises various native errors
        raise RedshiftIngestError(
            f"failed to open ODBC reader against {host}/{database}: "
            f"{_redact_secrets(str(exc), secrets)}"
        ) from None

    if reader is None:
        raise RedshiftIngestError(
            "query produced no result set; redshift_query expects a SELECT"
        )
    return cast("_ArrowBatchReader", reader)


def _read_arrow_schema(
    reader: _ArrowBatchReader, *, secrets: tuple[str, ...]
) -> pa.Schema:
    try:
        schema = reader.schema
    except Exception as exc:
        raise RedshiftIngestError(
            f"failed to read arrow schema from ODBC result: "
            f"{_redact_secrets(str(exc), secrets)}"
        ) from None
    if schema is None:
        raise RedshiftIngestError("ODBC result reported a null arrow schema")
    return schema


def _check_schema_no_duplicates(schema: pa.Schema) -> list[str]:
    """Names + dupe check (pre-conversion → clearer than pl.from_arrow's
    ComputeError on dupes)."""
    names: list[str] = list(schema.names)
    dupes = _rs_check.find_duplicates(names)
    if dupes:
        raise RedshiftIngestError(
            f"duplicate column names in server result set: {dupes}"
        )
    return names


def _stream_batches(
    reader: _ArrowBatchReader,
    *,
    max_rows: int | None,
    host: str,
    database: str,
    secrets: tuple[str, ...],
) -> tuple[list[pa.RecordBatch], int]:
    """Drain ``reader`` into batches, aborting at ``max_rows`` (manual
    iteration so we don't balloon memory on a runaway result; at most
    one extra ``batch_size`` chunk is buffered beyond the cap)."""
    batches: list[pa.RecordBatch] = []
    total_rows = 0
    over_cap = False
    try:
        for batch in reader:
            batches.append(batch)
            total_rows += batch.num_rows
            if max_rows is not None and total_rows > max_rows:
                over_cap = True
                break
    except Exception as exc:
        batches.clear()
        raise RedshiftIngestError(
            f"failed while streaming batches from {host}/{database}: "
            f"{_redact_secrets(str(exc), secrets)}"
        ) from None

    if over_cap:
        # max_rows was non-None for over_cap to be set; explicit raise
        # (not assert) so the invariant survives python -O.
        if max_rows is None:
            raise RedshiftIngestError(
                "invariant violated: over_cap=True but max_rows is None"
            )
        batches.clear()
        raise RedshiftIngestError(
            f"result exceeded max_rows={max_rows}: streamed "
            f"{total_rows} rows before abort against {host}/{database}"
        )
    return batches, total_rows


# =============================================================================
# Arrow -> Polars conversion + invariants
# =============================================================================


def _assemble_arrow_table(
    batches: list[pa.RecordBatch],
    schema: pa.Schema,
    expected_rows: int,
) -> pa.Table:
    try:
        table = pa.Table.from_batches(batches=batches, schema=schema)
    except Exception as exc:
        raise RedshiftIngestError(
            f"failed to assemble arrow table: {exc}"
        ) from exc
    if table.num_rows != expected_rows:
        raise RedshiftIngestError(
            "row-count invariant violated assembling arrow Table: "
            f"sum(batch.num_rows)={expected_rows} vs table.num_rows="
            f"{table.num_rows}"
        )
    # Table holds the data zero-copy; release the per-batch Python refs.
    batches.clear()
    return table


def _arrow_table_to_polars(
    table: pa.Table,
    *,
    schema_names: list[str],
    expected_rows: int,
) -> pl.DataFrame:
    """Convert; assert column-order and row-count preserved across the
    arrow→polars boundary."""
    try:
        df_or_series = pl.from_arrow(table)
    except Exception as exc:
        raise RedshiftIngestError(
            f"failed to convert arrow Table to Polars DataFrame: {exc}"
        ) from exc
    # ``pl.from_arrow`` returns DataFrame for Table input; the Series
    # branch exists for Array/ChunkedArray. Guard at runtime — a static
    # ``cast`` would let an API drift slip through as an opaque
    # AttributeError on the next ``df.columns`` access.
    if not isinstance(df_or_series, pl.DataFrame):
        raise RedshiftIngestError(
            f"expected pl.DataFrame from arrow Table, got "
            f"{type(df_or_series).__name__}"
        )
    df = df_or_series

    cols = list(df.columns)
    if cols != schema_names:
        raise RedshiftIngestError(
            "schema mismatch between arrow Table and Polars DataFrame "
            f"(arrow={schema_names} polars={cols}); "
            "pl.from_arrow must not reorder or rename columns"
        )
    if df.height != expected_rows:
        raise RedshiftIngestError(
            "row-count invariant violated in arrow->polars conversion: "
            f"expected {expected_rows}, got {df.height}"
        )
    return df


# =============================================================================
# Caller-facing post-load contract checks
# =============================================================================


def _check_expected_columns(
    df: pl.DataFrame,
    expected: list[str] | None,
    *,
    allow_extra_columns: bool,
) -> None:
    if expected is None:
        return
    cols = list(df.columns)
    missing = [c for c in expected if c not in cols]
    extra = [c for c in cols if c not in expected]
    if missing or (extra and not allow_extra_columns):
        err_msg = f"schema mismatch: missing={missing}"
        if not allow_extra_columns:
            err_msg += f" extra={extra}"
        err_msg += f" got={cols}"
        raise RedshiftIngestError(err_msg)


def _has_naive_datetime(dtype: pl.DataType) -> bool:
    """Recursively check for a naive Datetime leaf — Redshift SUPER can
    project as Struct/List/Array, hiding a naive inner Datetime."""
    if isinstance(dtype, pl.Datetime):
        return _datetime_tz(dtype) is None
    inner = getattr(dtype, "inner", None)
    if isinstance(inner, pl.DataType) and _has_naive_datetime(inner):
        return True
    fields = getattr(dtype, "fields", None)
    if fields is not None:
        for f in fields:
            f_dtype = getattr(f, "dtype", None)
            if isinstance(f_dtype, pl.DataType) and _has_naive_datetime(f_dtype):
                return True
    return False


def _check_tz_aware_timestamps(
    df: pl.DataFrame, *, require: bool
) -> None:
    if not require:
        return
    # Recurse into Struct/List/Array (Redshift SUPER projections).
    naive = [
        name
        for name, dtype in df.schema.items()
        if _has_naive_datetime(dtype)
    ]
    if naive:
        raise RedshiftIngestError(
            f"require_tz_aware_timestamps=True but columns contain "
            f"naive Datetime (top-level or nested in Struct/List/"
            f"Array): {naive}"
        )


def _check_non_empty(
    df: pl.DataFrame, *, require: bool, host: str, database: str
) -> None:
    if require and df.height == 0:
        raise RedshiftIngestError(
            f"query returned 0 rows against {host}/{database} "
            "(require_non_empty=True)"
        )


# =============================================================================
# Public entry point
# =============================================================================


def redshift_query(
    host: str,
    database: str,
    query: str,
    *,
    parameters: Sequence[str | None] | None = None,
    driver: str = "Amazon Redshift (x64)",
    port: int = 5439,
    user: str | None = None,
    password: str | None = None,
    batch_size: int = 100_000,
    login_timeout_sec: int = 30,
    query_timeout_sec: int | None = None,
    max_rows: int | None = None,
    max_text_size: int | None = None,
    max_binary_size: int | None = None,
    ssl_mode: str | None = None,
    iam: bool = False,
    aws_profile: str | None = None,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    aws_session_token: str | None = None,
    aws_region: str | None = None,
    cluster_id: str | None = None,
    db_user: str | None = None,
    db_groups: Sequence[str] | None = None,
    auto_create: bool = False,
    expected_columns: Sequence[str] | None = None,
    allow_extra_columns: bool = False,
    require_non_empty: bool = False,
    require_tz_aware_timestamps: bool = False,
    require_finite_numerics: bool = False,
) -> pl.DataFrame:
    """Run a SELECT against Amazon Redshift and return a Polars DataFrame.

    Parameters are bound via ODBC (``parameters``), never via string
    formatting, so callers should always prefer ``?`` placeholders over
    f-strings in ``query``. All boundary-contract failures surface as
    ``RedshiftIngestError``; ``MemoryError`` and signal exceptions
    (``KeyboardInterrupt`` / ``SystemExit``) propagate unwrapped.

    Memory: the full result set is materialized in memory. For multi-GB
    queries, shrink the server-side projection or filter before calling,
    or pass ``max_rows`` to abort early.

    ``max_rows`` (opt-in): caps how many rows the stream is allowed to
    accumulate before we abort with ``RedshiftIngestError``. The abort
    fires during streaming, so at most one extra ``batch_size`` chunk of
    rows is buffered beyond the cap before memory is released.

    ``max_text_size`` / ``max_binary_size`` (opt-in, bytes): forwarded to
    ``arrow_odbc`` to size the buffers it allocates for unbounded SQL
    types. Redshift's max ``VARCHAR`` length is 65535 bytes (no MAX
    type), but stored procedures and SUPER->VARCHAR projections can
    return wider rows than the driver's default buffer; if your SELECT
    returns large text columns, set these explicitly.

    ``ssl_mode`` (opt-in): emitted as ``SSLMode=<value>`` when not
    ``None``. Whitelist: ``verify-full``, ``verify-ca`` (driver default),
    ``require``, ``prefer``, ``disable``. Redshift clusters always speak
    TLS, so ``disable`` is rejected by most cluster configurations and
    is included only for parity with the driver's documented values.
    Use ``verify-full`` (CA + hostname) for production.

    Authentication modes:

      * **Native** (``iam=False``, the default): supply ``user`` +
        ``password`` for a Redshift database user. All IAM-only fields
        must be ``None``.

      * **IAM federated** (``iam=True``): the driver calls
        ``redshift:GetClusterCredentials`` to mint a temporary password
        for ``db_user`` (required). Pick exactly one credential source:

          - ``aws_profile``: read AWS creds from the named profile in
            ``~/.aws/credentials`` / ``~/.aws/config``.
          - ``aws_access_key_id`` + ``aws_secret_access_key``
            (+ optional ``aws_session_token`` for temporary STS creds):
            explicit creds.
          - Neither set: fall back to the AWS default credential chain
            (env vars ``AWS_ACCESS_KEY_ID`` etc., EC2 instance role,
            ECS task role, IMDSv2). Intended for code running inside
            AWS with a role attached.

        ``db_groups`` (optional): the IAM-issued temp user joins these
        Redshift groups for the session. Names are validated and
        comma-joined into the connection string.

        ``auto_create=True``: if ``db_user`` doesn't exist in Redshift,
        create it on first login. Off by default — enabling silently
        creates database principals, which most production roles
        prohibit.

        ``cluster_id`` + ``aws_region``: required when the driver cannot
        infer them from ``host`` (i.e. when ``host`` is a custom CNAME
        rather than the standard cluster endpoint). Safe to omit when
        ``host`` is the canonical
        ``<cluster>.<id>.<region>.redshift.amazonaws.com`` endpoint —
        the driver parses cluster id and region from the hostname.

    IdP plugins (Okta, Azure AD, AD FS, JWT, BrowserAzureAD, etc.) are
    NOT supported by this wrapper. They each require their own subset
    of ``plugin_name``-specific connection keys; adding them here would
    bloat the surface and the validation matrix. If you need an IdP
    plugin, build the connection string at the call site and call
    ``arrow_odbc.read_arrow_batches_from_odbc`` directly.

    Cross-platform:
      * Driver name differs by platform / installer:
          - Windows: ``"Amazon Redshift (x64)"`` (the default here),
            installed via the ``AmazonRedshiftODBC64`` MSI.
          - Linux: typically ``"Amazon Redshift ODBC Driver"`` as
            registered in ``/etc/odbcinst.ini`` by the RPM/DEB
            package; pass it explicitly via ``driver=``.
          - macOS: install + register via unixODBC (NOT iODBC, which
            macOS ships by default and which is not supported).
      * Driver manager: Windows uses the built-in ODBC Driver Manager
        (case-INSENSITIVE driver lookup); Linux/macOS use ``unixODBC``
        (case-SENSITIVE — match the ``[Amazon Redshift ODBC Driver]``
        section header in ``odbcinst.ini`` exactly).

    Multiple result sets: Redshift's wire protocol returns one result
    per statement, but a multi-statement ``query`` (e.g. ``SET ...; SELECT
    ...``) only surfaces the FIRST result set through ``arrow_odbc`` —
    subsequent ones are silently dropped. Run session ``SET`` separately
    or fold them into a single SELECT.

    Set ``require_tz_aware_timestamps=True`` to enforce the CLAUDE.md
    inbound contract that every Datetime column carries a timezone.
    Redshift's ``TIMESTAMP`` is naive (no zone) and ``TIMESTAMPTZ`` is
    UTC-aware; cast naive columns to ``TIMESTAMPTZ`` server-side (e.g.
    ``ts AT TIME ZONE 'UTC'``) before querying with this flag set.
    The check recurses through ``pl.Struct`` / ``pl.List`` / ``pl.Array``
    (Redshift SUPER projections can land as nested arrow types on some
    driver builds), so a naive Datetime nested inside a struct is also
    flagged.

    Set ``require_finite_numerics=True`` to enforce the CLAUDE.md
    inbound contract item 4 ("Finiteness"). Every numeric column —
    ``Int*`` / ``UInt*`` / ``Decimal`` / ``Float32`` / ``Float64`` —
    must be entirely free of NULL; float columns additionally must be
    free of ``NaN`` / ``+Inf`` / ``-Inf``. Defaults off because
    Redshift NULL is a documented expectation in many tables (per
    CLAUDE.md's "except where NaN is documented as expected"); enable
    only for tables where NULL/NaN indicates upstream corruption.

    Output contract:
      * Column order and names are preserved exactly as Redshift returns
        them. Case follows Redshift's default lowercasing of unquoted
        identifiers; double-quote identifiers in the SELECT list to
        preserve mixed case.
      * The returned DataFrame is NOT sorted — caller must sort
        explicitly before any asof join.
      * Row count equals the number of rows the server streamed; this
        function never silently drops, dedupes, or truncates.
      * ``expected_columns`` is a NAMES-ONLY contract; dtypes are not
        compared. Downstream callers that need dtype guarantees should
        follow up with ``df.cast(...)`` or validate ``df.schema``
        explicitly at the call site.

    Thread-safety: each call opens its own ODBC cursor via arrow_odbc and
    holds no module-level state, so concurrent calls from independent
    threads are safe. Callers must not share a returned DataFrame across
    threads without their own synchronization (standard Polars rules).
    """
    # --- 1. Validate every input field at the inbound boundary ---------------
    _validate_endpoint_identifiers(host=host, database=database, driver=driver)
    _validate_query(query)
    max_rows_val = _validate_numeric_params(
        port=port,
        batch_size=batch_size,
        login_timeout_sec=login_timeout_sec,
        query_timeout_sec=query_timeout_sec,
        max_rows=max_rows,
        max_text_size=max_text_size,
        max_binary_size=max_binary_size,
    )
    _validate_bool_flags(
        require_non_empty=require_non_empty,
        allow_extra_columns=allow_extra_columns,
        require_tz_aware_timestamps=require_tz_aware_timestamps,
        require_finite_numerics=require_finite_numerics,
        iam=iam,
        auto_create=auto_create,
    )
    # Structural validation for caller-supplied sequences runs BEFORE
    # the credential content sweep so a type error like
    # ``parameters="?"`` or ``expected_columns={"a","b"}`` surfaces
    # first. Otherwise a caller with multiple faults would chase the
    # credential error (e.g. a trailing ``\n`` in a password copy-paste)
    # while the more fundamental "wrong container type" went unmentioned.
    params = _validate_parameters(parameters)
    expected = _validate_expected_columns(expected_columns)
    _validate_optional_credential_strings(
        {
            "user": user,
            "password": password,
            "aws_profile": aws_profile,
            "aws_access_key_id": aws_access_key_id,
            "aws_secret_access_key": aws_secret_access_key,
            "aws_session_token": aws_session_token,
            "aws_region": aws_region,
            "cluster_id": cluster_id,
            "db_user": db_user,
            "ssl_mode": ssl_mode,
        }
    )
    _validate_ssl_mode(ssl_mode)
    _validate_conn_string_identifier_fields(
        aws_profile=aws_profile,
        aws_region=aws_region,
        cluster_id=cluster_id,
        db_user=db_user,
    )
    _validate_iam_auth(
        iam=iam,
        user=user,
        password=password,
        aws_profile=aws_profile,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        aws_region=aws_region,
        cluster_id=cluster_id,
        db_user=db_user,
        auto_create=auto_create,
    )
    db_groups_value = _validate_db_groups(db_groups, iam=iam)

    # --- 2. Compose the ODBC connection string -------------------------------
    conn_str = _build_connection_string(
        driver=driver,
        host=host,
        port=port,
        database=database,
        iam=iam,
        user=user,
        password=password,
        db_user=db_user,
        db_groups_value=db_groups_value,
        auto_create=auto_create,
        cluster_id=cluster_id,
        aws_region=aws_region,
        aws_profile=aws_profile,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        ssl_mode=ssl_mode,
    )

    # --- 3. Open reader, stream rows, release cursor synchronously -----------
    # ``del reader`` triggers arrow_odbc's Rust Drop in CPython, closing
    # the server-side cursor before any exception propagates.
    secrets = tuple(
        s
        for s in (
            password,
            aws_access_key_id,
            aws_secret_access_key,
            aws_session_token,
        )
        if s
    )
    reader = _open_arrow_reader(
        conn_str=conn_str,
        query=query,
        parameters=params,
        batch_size=batch_size,
        login_timeout_sec=login_timeout_sec,
        query_timeout_sec=query_timeout_sec,
        max_text_size=max_text_size,
        max_binary_size=max_binary_size,
        user=user,
        password=password,
        iam=iam,
        host=host,
        database=database,
        secrets=secrets,
    )
    try:
        schema = _read_arrow_schema(reader, secrets=secrets)
        schema_names = _check_schema_no_duplicates(schema)
        batches, total_rows = _stream_batches(
            reader,
            max_rows=max_rows_val,
            host=host,
            database=database,
            secrets=secrets,
        )
    finally:
        del reader

    # --- 4. Convert to Polars while preserving invariants --------------------
    table = _assemble_arrow_table(batches, schema, total_rows)
    df = _arrow_table_to_polars(
        table, schema_names=schema_names, expected_rows=total_rows
    )

    # --- 5. Caller-facing post-load contract checks --------------------------
    # Post-load contract checks ordered to match the CLAUDE.md inbound
    # boundary contract enumeration: schema (1), timezone (2), row
    # count (3), finiteness (4). Universe-sanity (5) and timestamp
    # monotonicity (6) are caller-knowledge and stay caller-side.
    _check_expected_columns(
        df, expected, allow_extra_columns=allow_extra_columns
    )
    _check_tz_aware_timestamps(df, require=require_tz_aware_timestamps)
    _check_non_empty(
        df, require=require_non_empty, host=host, database=database
    )
    _rs_check.finite_numerics(df, require=require_finite_numerics)

    return df

