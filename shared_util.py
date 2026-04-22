#################################################################################################
# Import Libraries
from __future__ import annotations
from collections.abc import Sequence
import polars as pl
import pyarrow as pa  # type: ignore[import-untyped]
from arrow_odbc import read_arrow_batches_from_odbc  # type: ignore[import-untyped]

import argparse
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
import time
import uuid
import zoneinfo
from collections.abc import Callable
from pathlib import Path, PurePath
from typing import Any, Literal, NamedTuple, overload

import exchange_calendars as xcals  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import polars as pl
from filelock import FileLock
from matplotlib.figure import Figure

#################################################################################################
# Parse command line arguments (or use defaults in interactive mode), validate all inputs,
# and resolve the write directory path. Returns (startdate, enddate, write_directory).
#
# Supports two execution modes:
#   1. Interactive (Jupyter / IPython): skips argparse, uses defaults directly.
#   2. CLI (python script.py ...): parses -s, -e, -w flags via argparse.
#
# CLI usage examples:
#   python price-volume-01-EDA.py                                          # all defaults
#   python price-volume-01-EDA.py -s 20230101 -e 20231231                  # custom date range
#   python price-volume-01-EDA.py --start-date 20230101 --end-date 20231231 --write-to prod


def parse_arguments(
    default_startdate: datetime.date,
    default_enddate: datetime.date,
    default_write_mode: Literal["dev", "prod"],
    dev_folder: str,
    prod_folder: str,
) -> tuple[datetime.date, datetime.date, str]:

    # ── Step 1: Detect execution mode ────────────────────────────────────
    # Detect Jupyter/IPython ('ipykernel') or standard REPL (sys.ps1).
    # Avoid checking sys.argv[0].endswith(".py") as it incorrectly categorizes
    # execution via `python -m`, PyInstaller (.exe), or script wrappers (.exe/.sh)
    # as interactive, which would silently swallow CLI arguments.
    is_interactive = hasattr(sys, "ps1") or "ipykernel" in sys.modules

    # ── Step 2: Obtain raw string inputs ─────────────────────────────────
    if is_interactive:
        # Interactive mode: convert defaults to YYYYMMDD strings (same format argparse would produce)
        print("Running in interactive mode. Using default parameters.")
        start_date_str = default_startdate.strftime("%Y%m%d")
        end_date_str = default_enddate.strftime("%Y%m%d")
        write_to = default_write_mode
    else:
        # CLI mode: define and parse command line arguments
        parser = argparse.ArgumentParser()

        # -s / --start-date: start of the date range (inclusive), format YYYYMMDD
        parser.add_argument(
            "-s",
            "--start-date",
            type=str,
            default=default_startdate.strftime("%Y%m%d"),
            help=f"\nStart date default is {default_startdate.strftime('%Y%m%d')} in YYYYMMDD format",
        )

        # -e / --end-date: end of the date range (inclusive), format YYYYMMDD
        parser.add_argument(
            "-e",
            "--end-date",
            type=str,
            default=default_enddate.strftime("%Y%m%d"),
            help=f"\nEnd date default is {default_enddate.strftime('%Y%m%d')} in YYYYMMDD format",
        )

        # -w / --write-to: target environment for output files ('dev' or 'prod')
        parser.add_argument(
            "-w",
            "--write-to",
            type=str,
            default=default_write_mode,
            help=f"\nDestination to write the results. Default is {default_write_mode}. Options are dev or prod",
        )

        # argparse converts --start-date -> start_date, --end-date -> end_date, --write-to -> write_to
        args = parser.parse_args()
        start_date_str = args.start_date
        end_date_str = args.end_date
        write_to = args.write_to

    # ── Step 3: Validate and convert date strings to datetime.date ───────
    # strptime raises a cryptic error on bad formats, so wrap with a clear message
    try:
        startdate = datetime.datetime.strptime(start_date_str, "%Y%m%d").date()
        enddate = datetime.datetime.strptime(end_date_str, "%Y%m%d").date()
    except ValueError as e:
        raise ValueError(
            f"Invalid date format. Expected YYYYMMDD (e.g. 20241231), "
            f"got start_date='{start_date_str}', end_date='{end_date_str}'. Original error: {e}"
        ) from e

    # Ensure the date range is logically valid (start on or before end)
    if startdate > enddate:
        raise ValueError(
            f"start_date ({startdate}) must be on or before end_date ({enddate})"
        )

    # ── Step 4: Resolve write directory from the mode string ─────────────
    # Guard against empty folder strings before resolving.  Path("").resolve()
    # returns the current working directory, so an empty dev_folder/prod_folder
    # would silently pass the isdir check and write files to the wrong location.
    if not dev_folder or not dev_folder.strip():
        raise ValueError("dev_folder must not be empty or whitespace")
    if not prod_folder or not prod_folder.strip():
        raise ValueError("prod_folder must not be empty or whitespace")

    # Normalize to lowercase so CLI input like 'DEV', 'Prod' is accepted
    write_to = write_to.strip().lower()
    if write_to == "dev":
        write_directory = dev_folder
    elif write_to == "prod":
        write_directory = prod_folder
    else:
        raise ValueError(
            f"Invalid value for write_to: '{write_to}'. Options are 'dev' or 'prod'"
        )

    # Resolve to absolute path so the returned directory remains valid even if
    # the caller changes the working directory before using it (same guard as
    # lazy_parquet applies to folder_path).
    write_directory = str(Path(write_directory).resolve())

    # Fail early if the resolved directory doesn't exist on disk
    if not Path(write_directory).is_dir():
        raise FileNotFoundError(f"Write directory does not exist: {write_directory}")

    return startdate, enddate, write_directory

#################################################################################################
# Compute default month-to-date date range from the composite exchange calendar of the given venues.

def _mtd_safe_log_warning(fmt: str, *args: object) -> None:
    """Log a warning without letting handler failures propagate.

    The logging calls in this family surface library-inconsistency
    signals to operators — they must not themselves become a new
    failure mode. A misconfigured handler (broken formatter, dead
    network-logging socket) raising inside ``.warning(...)`` would
    otherwise escape the surrounding ``except`` and mask the caller's
    conservative-return path with a logging-traceback. Swallow any
    logger-side failure silently (the whole point of this helper is
    best-effort logging).
    """
    try:
        logging.getLogger(__name__).warning(fmt, *args)
    except Exception:
        # Intentionally silent: logger-side failures are best-effort
        # only; the caller's conservative-return path is load-bearing.
        pass


def _mtd_validate_inputs(
    venues: list[str],
    combine_type: Literal["union", "intersect"],
) -> list[str]:
    """Validate inputs and return a deduplicated venue list.

    Rejects bare str/bytes (which would silently iterate character-by-
    character downstream) and non-list/tuple inputs. Deduplicates while
    preserving caller order — duplicates are functionally harmless but
    waste calendar loads and make error messages noisy.

    ``combine_type`` is required with no default: the intersect↔union
    distinction silently changes which dates flow downstream, so an
    explicit choice prevents callers from picking up a silent behavior
    change on future default edits.
    """
    if isinstance(venues, (str, bytes)):
        raise TypeError(
            f"compute_mtd_date_range: venues must be a list of strings, "
            f"got {type(venues).__name__}"
        )
    if not isinstance(venues, (list, tuple)):
        raise TypeError(
            f"compute_mtd_date_range: venues must be a list or tuple, "
            f"got {type(venues).__name__}"
        )
    if not venues:
        raise ValueError("compute_mtd_date_range: venues must be non-empty")
    if combine_type not in ("union", "intersect"):
        raise ValueError(
            f"compute_mtd_date_range: combine_type must be 'union' or "
            f"'intersect', got {combine_type!r}"
        )

    # Materialize once before iterating. A hostile list subclass with an
    # overridden ``__iter__`` could yield different elements on separate
    # passes, letting an invalid venue pass the validation loop but
    # reach the dedup loop (silent-wrong-answer). ``list(...)`` takes a
    # single snapshot and validation + dedup then walk the same copy.
    venues_snapshot: list[str] = list(venues)
    seen: set[str] = set()
    unique_venues: list[str] = []
    for v in venues_snapshot:
        if not isinstance(v, str) or not v.strip():
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
    """Load each venue's xcals calendar and resolve a single shared tz.

    Loads per-venue (not via comprehension) so a failing venue names
    itself. Requires all venues to share one timezone; the composite-
    session logic is tz-ambiguous otherwise (US equity venues share
    America/New_York — mixing US with non-US calendars is out of scope).
    Mocks that slip a plain string through the str-label check are
    caught by a final ``isinstance(tz, datetime.tzinfo)`` guard.
    """
    cals: list[Any] = []
    for v in unique_venues:
        try:
            cal = xcals.get_calendar(v)
        except Exception as e:
            # xcals' exception class drifts across versions — catch
            # broadly and re-raise with caller context.
            raise ValueError(
                f"compute_mtd_date_range: failed to load calendar for "
                f"venue {v!r}: {e!s}"
            ) from e
        if cal is None:
            # Documented to raise on unknown venue; guard against a
            # mock/stub returning None so downstream ``.tz`` doesn't
            # explode with an opaque AttributeError.
            raise RuntimeError(
                f"compute_mtd_date_range: xcals returned None for "
                f"venue {v!r}"
            )
        cals.append(cal)
    if not cals:
        # unique_venues is non-empty by construction; belt-and-suspenders
        # against a stray IndexError further down.
        raise RuntimeError(
            f"compute_mtd_date_range: no calendars loaded for "
            f"venues={unique_venues}"
        )

    tzs: list[Any] = []
    for v, c in zip(unique_venues, cals, strict=True):
        try:
            tzs.append(c.tz)
        except Exception as e:
            raise ValueError(
                f"compute_mtd_date_range: failed to read tz from calendar "
                f"for venue {v!r}: {e!s}"
            ) from e

    # Per-venue missing-tz check BEFORE the cross-venue comparison so
    # the error names the offending venue(s) instead of surfacing a
    # confusing ``{"None", "America/New_York"}`` multi-tz set.
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

    # Defense-in-depth against the pytz↔zoneinfo↔dateutil.tz mix: two
    # different tzinfo implementations can both stringify to
    # "America/New_York" but be non-identical objects. CPython's tz-
    # aware comparison handles mixed implementations correctly via UTC
    # normalization, so this is not a silent-wrong-answer hazard today,
    # but a narrower contract (single implementation) fails faster and
    # survives future stdlib changes. Check via type name to avoid
    # hashing tzinfo instances (which a pathological mock could break).
    tz_types = {type(t).__name__ for t in tzs}
    if len(tz_types) != 1:
        raise ValueError(
            f"compute_mtd_date_range: venues use multiple tzinfo "
            f"implementations {tz_types} — mixing pytz / zoneinfo / "
            f"dateutil.tz is rejected to prevent implementation-drift "
            f"artifacts in session_close comparisons"
        )

    tz = tzs[0]
    if not isinstance(tz, datetime.tzinfo):
        # A mock could slip a plain string through the str-label check
        # (str of a str is itself). ``datetime.now`` strictly requires
        # a tzinfo subclass.
        raise TypeError(
            f"compute_mtd_date_range: calendar tz must be a datetime.tzinfo "
            f"subclass, got {type(tz).__name__} ({tz!r})"
        )
    return cals, tz


def _mtd_now_local(tz: datetime.tzinfo) -> datetime.datetime:
    """Return ``datetime.now(tz=tz)``. ``tz`` is isinstance-validated upstream."""
    return datetime.datetime.now(tz=tz)


def _mtd_make_session_probe(
    cals: list[Any],
    combine_type: Literal["union", "intersect"],
) -> tuple[Callable[[datetime.date], bool], list[BaseException]]:
    """Build the session probe and its shared error-capture list.

    The returned probe treats OOB or erroring dates as "not a session"
    so the lookback loop can step past rather than crash. Captured
    exceptions let a lookback-exhaustion error distinguish "library
    fault" from "genuine long closure" without spamming logs on every
    OOB probe. ``all`` for intersect, ``any`` for union.
    """
    probe_errors: list[BaseException] = []
    combine_fn = all if combine_type == "intersect" else any

    def _is_target_session(d: datetime.date) -> bool:
        try:
            return combine_fn(c.is_session(d) for c in cals)
        except Exception as e:
            probe_errors.append(e)
            return False

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
    """Aggressive "today" rule: include today iff ``now_local`` is at or
    past the latest session close among venues whose calendar has today
    as a session.

    intersect: ``is_target_session(today)`` guarantees every venue has
        today as a session, so ``session_close`` yields a concrete close
        for each; any None is a library regression and we conservatively
        exclude today.
    union: only the subset of venues that trade today contribute — the
        rest may return None or raise (OOB). Both are filtered; the
        "has the day ended" judgment is against the latest-closing venue
        in the contributing subset (same conservative posture as
        intersect, scoped to the subset).

    Two-layer sentinel handling: (1) fast-reject on explicit ``None``;
    (2) any other pathological value (``pd.NaT``, tz-naive datetime,
    un-comparable types, ``bool``-raisers) is caught by the outer
    ``except`` and logged. Do NOT add a per-element ``bool(cl)`` filter —
    ``bool(pd.NaT)`` raises, defeating the purpose.
    """
    if not is_target_session(today):
        return False
    try:
        if combine_type == "intersect":
            closes = [c.session_close(today) for c in cals]
            if any(cl is None for cl in closes):
                return False
            latest_close = max(closes)
            # bool(numpy.bool_) → Python bool (fine).
            # bool(pd.NaT) → TypeError → caught below.
            # Plain datetime comparison → Python bool.
            return bool(now_local >= latest_close)
        # union
        union_closes: list[Any] = []
        union_close_errors: list[BaseException] = []
        for c in cals:
            try:
                cl = c.session_close(today)
            except Exception as e:
                # Venue doesn't have today as a session (OOB or non-
                # trading day) — it doesn't contribute. Capture for
                # the post-loop library-fault check below.
                union_close_errors.append(e)
                continue
            if cl is not None:
                union_closes.append(cl)
        if not union_closes:
            # is_target_session(today) was True (at least one venue
            # reports today as a session), but no venue yielded a
            # concrete close — library inconsistency. Surface the last
            # per-cal error so operators don't silently stale-date
            # every run when e.g. session_close regresses library-wide.
            _mtd_safe_log_warning(
                "compute_mtd_date_range: union today-inclusion found "
                "is_target_session(today)=True but zero concrete closes "
                "(venues=%s, today=%s, close_errors=%d, last=%s)",
                unique_venues, today, len(union_close_errors),
                (f"{type(union_close_errors[-1]).__name__}: "
                 f"{union_close_errors[-1]!s}"
                 if union_close_errors else "no-exception"),
            )
            return False
        latest_close = max(union_closes)
        return bool(now_local >= latest_close)
    except Exception as e:
        # session_close / max / comparison / bool-coerce failure → be
        # conservative. Log (not silent-swallow) so upstream regressions
        # — e.g., session_close regressing to tz-naive — surface in
        # operator logs instead of silently stale-dating every run.
        _mtd_safe_log_warning(
            "compute_mtd_date_range: today-inclusion check failed, "
            "excluding today (venues=%s, combine_type=%s, today=%s): "
            "%s: %s",
            unique_venues, combine_type, today, type(e).__name__, e,
        )
        return False


def _mtd_resolve_lookback_days() -> int:
    """Resolve the lookback window, honoring an env-var override.

    Default 30d covers every major-venue closure on record. Long
    closures (e.g. Russian equities 2022, Greek banks 2015) or rare
    holiday+outage overlaps may exceed this — operators override via
    ``COMPUTE_MTD_MAX_LOOKBACK_DAYS`` instead of patching this file.

    A safety cap (10 years) rejects runaway values. Without it, a typo
    like ``COMPUTE_MTD_MAX_LOOKBACK_DAYS=100000`` (extra zero) would
    silently trigger ~100k probe iterations, each loading and querying
    calendars — a pipeline-hang / soft-DOS with no surfacing error.
    10y is far beyond any plausible continuous market closure while
    still finite and bounded.
    """
    default_days = 30
    max_allowed_days = 3650  # 10 years — cap against env-var typos.
    raw = os.environ.get("COMPUTE_MTD_MAX_LOOKBACK_DAYS")
    if raw is None or raw == "":
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
            f"{max_allowed_days} days (likely a misconfiguration; raise "
            f"the cap in this helper if a legitimate need exists)"
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
    is found, up to ``max_lookback`` days.

    On exhaustion, surface the last probe exception in the error so
    operators can tell a library fault apart from a genuine long closure.
    """
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
            f"last was {type(last).__name__}: {last} (possible calendar-"
            f"library fault rather than long closure)"
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
    """Return the first combined session in ``enddate``'s calendar month.

    ``enddate`` itself is already a combined session inside the probed
    window, so this loop is guaranteed to succeed under normal
    operation; the final raise exists only to survive a future refactor
    that might change the loop boundaries.
    """
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


def _mtd_check_postconditions(
    *,
    startdate: datetime.date,
    enddate: datetime.date,
    today: datetime.date,
    is_target_session: Callable[[datetime.date], bool],
    unique_venues: list[str],
    combine_type: Literal["union", "intersect"],
) -> None:
    """Validate the result tuple against every advertised invariant.

    Explicit raises (not ``assert``) so these guards survive ``python
    -O`` / PYTHONOPTIMIZE, which strips assertions. A corrupt date
    range silently flowing downstream is the worst failure mode this
    function can produce, so the checks run unconditionally.
    """
    if startdate > enddate:
        raise RuntimeError(
            f"compute_mtd_date_range: post-condition violated — "
            f"startdate={startdate} > enddate={enddate}"
        )
    if (startdate.year, startdate.month) != (enddate.year, enddate.month):
        raise RuntimeError(
            f"compute_mtd_date_range: post-condition violated — startdate "
            f"{startdate} and enddate {enddate} span different months"
        )
    if enddate > today:
        # Aggressive-today rule must never produce a future enddate.
        raise RuntimeError(
            f"compute_mtd_date_range: post-condition violated — "
            f"enddate={enddate} is after today={today}"
        )
    if not (is_target_session(startdate) and is_target_session(enddate)):
        # Re-verify session membership via the guarded probe (OOB-safe)
        # so a pathological calendar cannot explode here on a raw
        # is_session call.
        raise RuntimeError(
            f"compute_mtd_date_range: post-condition violated — "
            f"startdate={startdate} or enddate={enddate} is not a "
            f"{combine_type} session across {unique_venues}"
        )


def compute_mtd_date_range(
    venues: list[str],
    combine_type: Literal["union", "intersect"],
) -> tuple[datetime.date, datetime.date]:
    """
    Return (default_enddate, default_startdate) from the combined trading
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
    # Resolve lookback BEFORE loading calendars. A malformed env var
    # (non-int, negative, or exceeding the 10y safety cap) should fail
    # fast — before xcals.get_calendar incurs its cache / network cost
    # for every venue. The result is consumed by _mtd_find_enddate
    # below; binding it here keeps env-var parse failures at the
    # earliest sensible boundary.
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
    _mtd_check_postconditions(
        startdate=startdate,
        enddate=enddate,
        today=today,
        is_target_session=is_target_session,
        unique_venues=unique_venues,
        combine_type=combine_type,
    )
    return enddate, startdate

###################################################################################################
# Define function to lazy scan parquet files, concatenate them into a single polars lazyframe
# and filter by date range
def lazy_parquet(
    folder_path: str,
    date_column: str,
    start_date: datetime.date,
    end_date: datetime.date,
) -> pl.LazyFrame:
    """Lazy-scan every parquet file in ``folder_path``, validate schemas,
    and return a single date-filtered ``LazyFrame``.

    Notes
    -----
    The returned object is a ``LazyFrame``.  The NAS transient-lock retry
    envelope inside this function covers ONLY the metadata reads
    (``read_parquet_schema`` and ``scan_parquet``).  The actual payload
    read happens later, when the caller invokes ``.collect()`` /
    ``.sink_parquet()`` / ``.fetch()``, entirely outside this function's
    control.

    If the pipeline may run while ``save_results`` concurrently replaces
    files on the shared NAS, the caller MUST wrap its materialization
    call in its own transient-lock retry (mirroring the envelope in
    ``save_results``: retry on WinError 5/32/33 and
    EACCES/EBUSY/EAGAIN/ESTALE/ETXTBSY with 0.5s → 15s backoff).  Without
    that, a collect-time lock or vanish will surface as an unhandled
    ``OSError`` / ``ComputeError`` deep in downstream aggregation logic.

    For callers that prefer eager materialization with the same retry
    guarantees end-to-end, add a sibling ``eager_parquet`` — do NOT change
    this function, which is contracted to stay lazy.
    """
    if not folder_path or not folder_path.strip():
        raise ValueError("folder_path must not be empty or whitespace")
    if not date_column or not date_column.strip():
        raise ValueError("date_column must not be empty or whitespace")

    # Resolve to absolute path so that scan_parquet paths remain valid
    # even if the caller changes the working directory before collect().
    folder = Path(folder_path).resolve()
    folder_path = str(folder)

    if not folder.is_dir():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")

    # Normalize datetime.datetime to datetime.date.  datetime.datetime is a
    # subclass of datetime.date so it passes type checks, but pl.lit() would
    # create a Datetime literal whose time component silently shifts the filter
    # boundary (e.g. noon on Jan 1 excludes the morning of Jan 1).
    if isinstance(start_date, datetime.datetime):
        start_date = start_date.date()
    if isinstance(end_date, datetime.datetime):
        end_date = end_date.date()

    if start_date > end_date:
        raise ValueError(
            f"start_date ({start_date}) must be on or before end_date ({end_date})"
        )

    # Retry directory listing on transient NAS errors (SMB reconnect, NFS stale
    # handle).  Matches the retry pattern of _makedirs_with_retry: 3 attempts
    # with 0.5s / 1.0s backoff; the final attempt propagates the OSError so
    # the caller sees a real failure rather than an opaque empty folder.
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
        # Loop exits via break or raise; this guard survives python -O
        # where asserts are stripped.
        raise RuntimeError("lazy_parquet: listdir retry loop exhausted unexpectedly")

    # Filter on extension only.  Do NOT gate with Path.is_file(): it
    # swallows OSError (including ENOENT) and returns False, which would
    # silently drop a file that vanished between iterdir and this check —
    # bypassing the fail-loud logic in _read_schema_with_retry.  If a
    # directory happens to carry a .parquet suffix, the schema read will
    # raise loudly, which is the correct behavior.
    parquet_file_names = [
        file for file in file_names if file.lower().endswith(".parquet")
    ]

    if not parquet_file_names:
        raise ValueError(f"No parquet files found in folder: {folder_path}")

    # Read schemas and check compatibility before concatenating.
    # On a shared NAS, files can vanish between listdir and schema read
    # (e.g. concurrent save_results deleting old date-range files) — those
    # skip silently.  But transient AV/DLP/SMB locks (WinError 5/32/33,
    # EACCES/EBUSY/EAGAIN/ESTALE/ETXTBSY) must NOT be treated as "vanished":
    # swallowing them would drop a valid partition and silently under-count
    # rows downstream.  Mirror the save_results retry envelope: 12× with
    # 0.5s → 15s backoff; after that, raise loudly.
    def _is_transient_lock(e: OSError) -> bool:
        win_err = getattr(e, "winerror", 0)
        posix_err = getattr(e, "errno", 0)
        return win_err in (5, 32, 33) or posix_err in (
            errno.EACCES,
            errno.EBUSY,
            errno.EAGAIN,
            _ESTALE,
            _ETXTBSY,
        )

    def _read_schema_with_retry(path: str) -> dict[str, pl.DataType]:
        """Return schema.  Raises on any failure after exhausting retries.

        ENOENT is NOT silently skipped: in the date-range-filename pattern
        used by save_results, a concurrently-deleted file implies a
        replacement file now exists that our stale iterdir snapshot did
        not see.  Skipping the vanished file would leave a silent hole in
        the dataset.  Raise and let the caller retry the whole load.
        """
        for _attempt in range(12):
            try:
                return pl.read_parquet_schema(path)
            except OSError as e:
                if not _is_transient_lock(e):
                    raise
                if _attempt == 11:
                    raise
                time.sleep(min(15.0, 0.5 * (2**_attempt)))
        raise AssertionError("unreachable")  # loop exits via return or raise

    file_paths = [str(folder / file) for file in parquet_file_names]
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

    mismatches: list[str] = []
    for path, schema in schemas.items():
        if schema == reference_schema:
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

    if mismatches:
        raise ValueError(
            f"Schema mismatch across parquet files in {folder_path}:\n"
            + "\n".join(mismatches)
        )

    # Validate date_column exists and is Date or Datetime before building lazy filters.
    # Without this, a missing or wrong-typed column defers the error to collect-time
    # with a confusing message far from the root cause.
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

    # Build lazy scans with per-file error handling.
    # Files can vanish between schema-read and scan on a shared NAS
    # (e.g. concurrent save_results replacing old date-range files).
    # scan_parquet is lazy and won't fail on a missing file, but it can
    # fail eagerly in some Polars versions or on permission errors.
    #
    # Select columns in reference order on each frame. pl.concat with
    # how="vertical" (the default for LazyFrames) matches columns by
    # *position*, not by name. If two parquet files store the same columns
    # in a different order, positional concat silently produces wrong data
    # or raises a dtype-mismatch error at collect-time.
    def _scan_with_retry(path: str) -> pl.LazyFrame:
        """Build the lazy scan.  Raises on any failure after retries.

        Same rationale as _read_schema_with_retry: ENOENT is not skipped
        because the date-range-filename pattern means a vanished file
        implies an unseen replacement.  Fail loud and let the caller retry.
        """
        for _attempt in range(12):
            try:
                return pl.scan_parquet(path)
            except OSError as e:
                if not _is_transient_lock(e):
                    raise
                if _attempt == 11:
                    raise
                time.sleep(min(15.0, 0.5 * (2**_attempt)))
        raise AssertionError("unreachable")

    reference_columns = list(reference_schema.keys())
    dataframes = []
    for path in file_paths:
        # Cast to Date for day-level comparison.  For Date columns this
        # is a no-op.  For Datetime columns it truncates to date, which
        # prevents the <= filter from silently excluding same-day rows
        # with non-midnight timestamps (e.g. 2024-01-15 23:59:59 would
        # be excluded by <= Date(2024-01-15) without the cast, because
        # Polars supercasts Date to Datetime midnight).
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

    # Explicit how="vertical" — pl.concat's default on LazyFrames requires
    # matching schema AND column order.  The .select(reference_columns) above
    # guarantees both, but being explicit makes the safety property obvious
    # and guards against a future default change.
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
    df: pl.DataFrame,
    index_col: str,
    val_col: str,
    group_by: str | None,
    window: str | None,
    window_size: int | None,
    min_samples: int,
) -> None:
    """Parameter-shape validation — no df inspection beyond the type check.

    Verifies df is a DataFrame, column-name arguments are non-empty strings,
    exactly one of ``window`` / ``window_size`` is provided, and
    ``min_samples`` is a positive int not exceeding ``window_size``.
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError(
            f"df must be polars.DataFrame, got {type(df).__name__}"
        )
    for arg_name, arg_val in (
        ("index_col", index_col),
        ("val_col", val_col),
    ):
        if not isinstance(arg_val, str) or not arg_val:
            raise TypeError(
                f"{arg_name} must be a non-empty string, got {arg_val!r}"
            )
    if group_by is not None and (not isinstance(group_by, str) or not group_by):
        raise TypeError(
            f"group_by must be None or a non-empty string, got {group_by!r}"
        )

    # Exactly one of window / window_size must be provided.
    if (window is None) == (window_size is None):
        raise ValueError(
            "exactly one of window (duration string) or window_size "
            "(positive int, count-based) must be provided; got "
            f"window={window!r}, window_size={window_size!r}"
        )
    if window is not None and (not isinstance(window, str) or not window):
        raise TypeError(
            f"window must be a non-empty string, got {window!r}"
        )
    if window_size is not None:
        if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size <= 0:
            raise ValueError(
                f"window_size must be a positive int, got {window_size!r}"
            )
    if not isinstance(min_samples, int) or isinstance(min_samples, bool) or min_samples < 1:
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
    """Schema-dependent validation; returns ``(idx_dtype, val_dtype)``.

    Verifies column existence + distinctness, ``val_col`` is a non-Boolean
    numeric dtype, ``index_col`` is Integer / Date / Datetime with no
    nulls, ``group_by`` (if given) is a groupable dtype with no nulls,
    and neither ``index_col`` nor ``group_by`` collides with the reserved
    output column names.
    """
    required_cols = [index_col, val_col] + ([group_by] if group_by else [])
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Column(s) not found in df: {missing}. "
            f"Available: {df.columns}"
        )

    # Aliased column names (e.g. group_by == index_col) would make rolling()
    # produce a duplicate-column error or silently mis-aggregate. Reject here.
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

    # polars.DataFrame.rolling() only supports Integer, Date, or Datetime
    # index columns (a monotonic timeline). Time and Duration pass the
    # general is_temporal() gate but are rejected by rolling() with an
    # opaque error; restrict to the officially supported set up front.
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

    # group_by column dtype must be groupable; nested/object dtypes cause
    # polars to error inside rolling() with an opaque message.
    if group_by is not None:
        gb_dtype = df.schema[group_by]
        if isinstance(gb_dtype, (pl.List, pl.Struct, pl.Array, pl.Object)):
            raise TypeError(
                f"group_by {group_by!r} has dtype {gb_dtype}, which is not "
                f"groupable; use a scalar dtype (Int/Utf8/Categorical/Date/…)"
            )
        # Polars silently lumps all null group keys together, which is
        # almost never what the caller wants; fail loudly instead.
        if df[group_by].null_count() > 0:
            raise ValueError(
                f"group_by {group_by!r} contains null values; drop or fill "
                f"them before calling (polars would otherwise group all "
                f"nulls into a single bucket)"
            )

    # Output-column collision: rolling().agg() preserves index_col and
    # group_by in the output. If either is named "winsorized_mean" or
    # "winsorized_std", the final with_columns() would silently overwrite
    # it — a genuine silent-corruption risk. Reject up front.
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


def _validate_winsorized_window_string(
    window: str,
    index_col: str,
    idx_dtype: pl.DataType,
) -> None:
    """Parse the window string and cross-check its unit against ``idx_dtype``.

    Polars accepts compound durations like ``"1d12h"``, plus the integer-
    index unit ``"i"``. Catching malformed / zero-magnitude / unit-
    mismatched cases here turns opaque mid-plan parse errors into
    actionable messages.
    """
    _units_re = re.compile(r"(ns|us|ms|mo|[smhdwqyi])")
    _window_components_re = re.compile(r"(\d+)(ns|us|ms|mo|[smhdwqyi])")
    if not re.fullmatch(r"(\d+(ns|us|ms|mo|[smhdwqyi]))+", window):
        raise ValueError(
            f"window {window!r} is not a valid polars duration string; "
            f"expected e.g. '30s', '5m', '1d', '1mo', '10i' (or compound "
            f"forms like '1d12h')"
        )
    # Zero-magnitude windows ("0s", "0i", "0d0h") trigger a deep polars
    # "window must be positive" error; catch it here.
    if all(int(mag) == 0 for mag, _ in _window_components_re.findall(window)):
        raise ValueError(
            f"window {window!r} has zero total magnitude; rolling windows "
            f"must be strictly positive"
        )
    window_units = _units_re.findall(window)
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
    """Build the per-window winsorized-mean / -std expressions.

    ``min_samples`` raises the per-branch gate floor so the caller can
    align with a count-based ``rolling_*(window_size=N, min_samples=M)``.
    """
    # Collision-proof internal column names — a caller's df may already
    # have a column literally called "_sorted", "_sum", etc.
    tag = f"__wrs_{uuid.uuid4().hex[:8]}"
    sorted_c, sum_c, ss_c, len_c = (
        f"{tag}_sorted", f"{tag}_sum", f"{tag}_ss", f"{tag}_len",
    )

    # Cast val_col to Float64 up-front, then neutralize non-finite inputs:
    #   * Float64 avoids silent integer overflow in x*x / sum-of-squares
    #   * mapping ±Inf/NaN to null keeps them out of min/max, which would
    #     otherwise make a single infinity contaminate w_sum / w_ss via
    #     Inf - Inf = NaN and mask the entire window's output as None.
    #   * a null row stays excluded by drop_nulls() and by sum()'s null-skip
    #     semantics, so the rest of the window still produces valid stats.
    _raw = pl.col(val_col).cast(pl.Float64, strict=True)
    val_expr = pl.when(_raw.is_finite()).then(_raw).otherwise(None)

    # Shorthand Expressions
    s = pl.col(sorted_c)
    n = pl.col(len_c)
    raw_sum = pl.col(sum_c)
    raw_ss = pl.col(ss_c)

    # Get the 1st and 2nd lowest and highest values, with nulls for out-of-bounds
    lo, lo2 = s.list.get(0, null_on_oob=True), s.list.get(1, null_on_oob=True)
    hi, hi2 = s.list.get(-1, null_on_oob=True), s.list.get(-2, null_on_oob=True)

    w_sum = raw_sum - lo - hi + lo2 + hi2
    w_ss = raw_ss - lo * lo - hi * hi + lo2 * lo2 + hi2 * hi2

    mean_big = w_sum / n
    # Clamp at 0 before sqrt: catastrophic cancellation on near-constant
    # windows can make the numerator a tiny negative, which sqrt turns to NaN.
    var_big = ((w_ss - n * mean_big * mean_big) / (n - 1)).clip(lower_bound=0)

    mean_small = raw_sum / n
    var_small = ((raw_ss - n * mean_small * mean_small) / (n - 1)).clip(lower_bound=0)

    # Gate every branch on the minimum n required so disallowed sizes hit
    # `otherwise(None)` instead of computing 0/0 (NaN) or a degenerate value.
    # min_samples raises the floor on top of the intrinsic per-branch minima
    # (3 for winsorized, 1/2 for the small-window fallbacks).
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

    # Final guard: any residual NaN or ±Inf (e.g. overflow on extreme inputs,
    # or non-finite values leaking in from val_col) becomes None.
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
    """Run the ``rolling().agg().with_columns()`` pipeline.

    Handles both time-based (``window``) and count-based (``window_size``)
    modes, narrowing + sorting the frame first, and wrapping the polars
    call in a try/except that chains the original exception with input-
    parameter context.
    """
    # rolling() requires the frame to be monotonic by (group_by, index_col);
    # the defensive sort happens inside the try/except below so any failure
    # (sort comparison panic, etc.) is surfaced with input-parameter context.
    sort_keys = [group_by, index_col] if group_by else [index_col]

    # u32 row-index overflow guard (count-based path only).
    # polars.DataFrame.with_row_index produces a UInt32 column by default
    # (confirmed against the installed polars version). A frame with more
    # than 2**32 - 1 rows would overflow or panic deep inside the pipeline;
    # reject up front with an actionable message so the caller can switch
    # to the time-based window or chunk the input.
    _U32_MAX = 2**32 - 1
    if window_size is not None and df.height > _U32_MAX:
        raise ValueError(
            f"winsorized_rolling_stats: df has {df.height:,} rows, which "
            f"exceeds the u32 row-index maximum ({_U32_MAX:,}) used by the "
            f"count-based window_size path. Use the time-based 'window' "
            f"parameter instead, or pre-chunk the input."
        )

    # Wrap the polars pipeline: despite upstream validation, polars can still
    # raise ComputeError / SchemaError / PanicException / version-specific
    # exceptions from inside rolling/agg/with_columns. Re-raise with context
    # so the caller sees which inputs triggered the failure rather than a
    # deep-stack polars internal error.
    try:
        # Narrow df to only the columns this function actually reads before
        # sorting. Reasons:
        #   * The caller's df may have unrelated columns with pathological
        #     dtypes (Object holding unhashable Python objects, Categorical
        #     without a string cache, deeply-nested Structs) that would make
        #     sort/rolling copy-of-all fail or panic, even though we never
        #     use those columns.
        #   * Halves memory overhead on wide DataFrames — sort materializes
        #     a full copy, so narrowing first is cheaper.
        # rolling().agg() only preserves index_col + group_by + agg columns
        # anyway, so output semantics are identical.
        df = df.select([*sort_keys, val_col]).sort(sort_keys)
        row_idx_c = f"{plan.tag}_rowidx"
        if window_size is not None:
            # Count-based: build a per-group contiguous integer row index
            # (global `with_row_index` after sort gives each group a
            # contiguous monotonic range, which is what polars rolling
            # requires as its index column). We rejoin the agg output on
            # this index so the original index_col is preserved unchanged.
            df = df.with_row_index(name=row_idx_c)
            rolling_period: str = f"{window_size}i"
            rolling_index = row_idx_c
        else:
            # Runtime re-check rather than `assert`: survives `python -O`
            # (which strips asserts), so a future refactor that weakens
            # API-level validation cannot silently pass `None` as the
            # rolling period. Unreachable under normal flow.
            if window is None:
                raise RuntimeError(
                    "winsorized_rolling_stats: internal invariant violated "
                    "— window is None in the time-based branch"
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
            # Rejoin the preserved index_col from the sorted input frame
            # via the row-index key, then drop the synthetic index.
            result = (
                result
                .join(df.select(row_idx_c, index_col), on=row_idx_c, how="left")
                .drop(row_idx_c)
            )

        # Canonical column ordering:
        #   [group_by (if present), index_col, winsorized_mean, winsorized_std].
        # The count-based path rejoins index_col at the end, so its
        # natural order differs from the time-based path. A positional
        # consumer would silently receive two different schemas depending
        # on which mode was selected; normalize here so both paths are
        # contract-identical. The select also trims any column that
        # somehow survived the earlier `.drop(...)` (e.g. a polars
        # behavior change), turning a silent leak into an explicit
        # ColumnNotFoundError caught by the except wrap below.
        canonical_cols = (
            ([group_by] if group_by is not None else [])
            + [index_col, "winsorized_mean", "winsorized_std"]
        )
        result = result.select(canonical_cols)
    except Exception as e:
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
    """Post-condition checks — rolling().agg() is a 1:1 row-wise op.

    Catching a mismatch turns a silent-corruption failure (joins-on-index
    quietly go wrong) into a clear error, and guards against future polars
    behavior changes. Every error includes the same input-parameter
    context so the caller always gets the full picture regardless of
    which specific guard tripped.
    """
    _ctx = (
        f"index_col={index_col!r}, val_col={val_col!r}, "
        f"window={window!r}, group_by={group_by!r}, "
        f"input_height={input_height}"
    )

    # Type post-condition: a future polars version might change rolling().agg()
    # to return a LazyFrame (or some wrapper) — without this check, the
    # subsequent .height access would raise a cryptic AttributeError from the
    # post-condition code itself, masking the actual semantic change.
    if not isinstance(result, pl.DataFrame):
        raise RuntimeError(
            f"winsorized_rolling_stats: expected pl.DataFrame output from "
            f"rolling pipeline, got {type(result).__name__}; polars API may "
            f"have changed ({_ctx})"
        )
    if result.height != input_height:
        raise RuntimeError(
            f"winsorized_rolling_stats: output row count {result.height} "
            f"does not match input {input_height}; polars may have dropped "
            f"or duplicated rows ({_ctx})"
        )
    # Full-schema post-condition: result columns must be EXACTLY the
    # expected set (group_by if any, index_col, the two stat columns).
    # A narrower "missing stats only" check would miss two silent-
    # corruption modes: (a) index_col or group_by dropped by a future
    # polars change in rolling().agg() semantics, and (b) an internal
    # __wrs_* column leaking past the earlier `.drop(...)` and the
    # canonical `.select(...)` in _run_winsorized_rolling.
    expected_cols = {"winsorized_mean", "winsorized_std", index_col}
    if group_by is not None:
        expected_cols.add(group_by)
    actual_cols = set(result.columns)
    if actual_cols != expected_cols:
        missing_cols = expected_cols - actual_cols
        extra_cols = actual_cols - expected_cols
        raise RuntimeError(
            f"winsorized_rolling_stats: unexpected output schema — "
            f"missing={sorted(missing_cols)!r}, extra={sorted(extra_cols)!r}; "
            f"expected {sorted(expected_cols)!r}, got {result.columns} "
            f"({_ctx})"
        )
    # Dtype post-condition: both stat columns must be Float64 (the cast
    # target) or Null (only possible when result is empty and polars
    # couldn't infer from zero rows). Anything else would silently break
    # downstream numeric ops like .cast(Float64) / arithmetic.
    for _out in ("winsorized_mean", "winsorized_std"):
        _out_dtype = result.schema[_out]
        if not isinstance(_out_dtype, (pl.Float64, pl.Null)):
            raise RuntimeError(
                f"winsorized_rolling_stats: output column {_out!r} has "
                f"unexpected dtype {_out_dtype}; expected Float64 "
                f"(or Null for an empty result) ({_ctx})"
            )

    # Finiteness post-condition: mean_final / std_final gate every branch
    # on `.is_finite()`, so any non-null output value MUST be finite. A
    # leaked NaN or ±Inf would indicate a polars is_finite regression
    # (or a bug in the gating plan) and would silently corrupt downstream
    # aggregations like portfolio variance / z-score. Catch it here
    # instead of propagating. Skip the scan on empty (Null-dtype) columns
    # — there is nothing to check.
    for _out in ("winsorized_mean", "winsorized_std"):
        if isinstance(result.schema[_out], pl.Float64):
            n_non_finite = result.select(
                (pl.col(_out).is_not_null() & ~pl.col(_out).is_finite()).sum()
            ).item()
            if n_non_finite:
                raise RuntimeError(
                    f"winsorized_rolling_stats: {n_non_finite} non-finite "
                    f"value(s) leaked into output column {_out!r} despite "
                    f"the is_finite gate in the expression plan; possible "
                    f"polars regression ({_ctx})"
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
        df, index_col, val_col, group_by, window, window_size, min_samples,
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

    Parameters
    ----------
    data_series : list of (dates, values, label) tuples
    title : chart title
    y_format : format string for y-axis ticks (use ``{x}`` as placeholder).
        Examples: ``"{x:.3f}"``, ``"{x:.1f}K"`` (caller must pre-scale),
        ``"{x:.1f}B"`` (caller must pre-scale).

    Raises
    ------
    TypeError
        * data_series is not a list
        * a series element's dates / values is not a polars.Series
        * a series element's label is not a string
        * val_col dtype is non-numeric or Boolean
        * title or y_format is not a string
    ValueError
        * data_series is empty
        * a tuple is not length 3
        * dates and values have different lengths
        * a series pair is empty
        * y_format is not a valid str.format template (e.g. unknown
          substitution key), which would otherwise fail per-tick at render.
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

    # Validate y_format before plotting. FuncFormatter would otherwise raise
    # a KeyError per tick at render time, producing a stack of confusing
    # matplotlib warnings instead of a single clear error.
    try:
        y_format.format(x=0.0, x_k=0.0, x_m=0.0, x_b=0.0)
    except (KeyError, IndexError, ValueError) as e:
        raise ValueError(
            f"y_format {y_format!r} is not a valid str.format template: "
            f"{type(e).__name__}: {e}"
        ) from e

    # ── Plot ─────────────────────────────────────────────────────────────
    # Wrap in try/except so a mid-plot failure (e.g. from user-supplied data
    # sneaking past validation) doesn't leak the Figure — plt tracks open
    # figures globally, and a leaked figure can exhaust memory in long-
    # running notebooks.
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
        # Only render interactively when matplotlib is in interactive mode
        # (Jupyter %matplotlib inline, IPython --pylab, etc.).  In batch
        # scripts the default is non-interactive, and calling plt.show()
        # against a GUI backend (TkAgg/Qt5Agg) on headless Linux either
        # raises (no DISPLAY) or blocks waiting for the window to close —
        # silently hanging cron jobs.  Under Agg it would already be a
        # no-op, so gating here is strictly a headless-Linux safety net.
        if plt.isinteractive():
            plt.show()
        # Close the specific figure we just created.  plt.close() with no
        # argument closes pyplot's *current* figure, which may not be
        # ``fig`` if any other figure was created between the subplots
        # call and here.  Explicit fig avoids that footgun.
        plt.close(fig)
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

    Applies the same robustness patterns as save_results:
      1. Validate inputs (types, empty/whitespace, illegal chars, reserved names)
      2. Resolve folder path with traversal security check
      3. Verify Windows MAX_PATH / POSIX 255-byte basename limits
      4. Create folder with NAS-resilient retry
      5. Render each figure to a base64-encoded PNG
      6. Atomic write: write to .tmp, then os.replace to final path

    Parameters
    ----------
    write_directory : str
        Parent output directory — must already exist on disk.
    subfolder : str
        Relative subfolder within *write_directory* (created if needed).
    file_name_prefix : str
        Used for the HTML filename and page title.
    figures : list[Figure]
        Already-created matplotlib Figure objects (e.g. [fig1, fig3, ...]).

    Returns
    -------
    str
        Absolute path of the written HTML file.
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

    # Reuse the same string-level checks that save_results applies
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

    # Explicit check for both separators rather than os.path.basename(), which
    # only treats "\" as a separator on Windows.  This ensures identical
    # validation behaviour on both Windows and Linux.
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
            f"file_name_prefix contains characters illegal in Windows filenames: "
            f"{file_name_prefix!r}"
        )
    if _WINDOWS_RESERVED_NAMES.match(file_name_prefix.split(".")[0]):
        raise ValueError(
            f"file_name_prefix uses a reserved Windows device name: {file_name_prefix!r}"
        )

    # ── Step 2: Resolve folder path with traversal check ────────────────
    folder_path = _resolve_folder_path(write_directory, subfolder)

    # ── Step 3: Build paths and check lengths ───────────────────────────
    file_name = f"{file_name_prefix}_charts.html"
    file_path = str(Path(folder_path) / file_name)
    tmp_path = f"{file_path}.{uuid.uuid4().hex[:8]}.tmp"
    _validate_path_lengths(tmp_path)

    # ── Step 4: Create folder with NAS-resilient retry ──────────────────
    _makedirs_with_retry(folder_path)

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

    for i, fig in enumerate(figures):
        axes_list = fig.get_axes()
        raw_title = axes_list[0].get_title() if axes_list else ""
        safe_title = html.escape(raw_title)

        buf = io.BytesIO()
        try:
            try:
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            except Exception as e:
                # fig.savefig can fail if the figure was already closed
                # upstream, its canvas is in a bad state, or dpi × size
                # overflows memory.  Wrap with index + title context so the
                # caller can identify the offending figure without digging
                # through a raw matplotlib stack trace.
                raise RuntimeError(
                    f"save_matplotlib_charts_as_html: fig.savefig failed "
                    f"for figures[{i}] (title={raw_title!r}). "
                    f"Underlying error: {type(e).__name__}: {e}"
                ) from e
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode("ascii")
        finally:
            buf.close()

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

    try:
        _write_bytes_and_fsync(html_bytes, tmp_path)

        # Rename phase — delegate to _atomic_replace for 12× retry with
        # read-only-attribute clearing.  Reuses the same tested path as the
        # parquet pipeline rather than maintaining a parallel retry loop.
        _atomic_replace(tmp_path, file_path)
        rename_done = True

    except BaseException:
        if not rename_done:
            # Best-effort cleanup of partial temp file
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass  # Temp file may not exist if os.open failed
        raise

    print(f"Saved charts HTML to {file_path}")
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

## Safe errno access: these POSIX constants may be absent on some Windows Python builds.
## Using -1 as sentinel ensures they never accidentally match a real errno value.
_ESTALE: int = getattr(errno, "ESTALE", -1)  # NFS stale file handle
_ETXTBSY: int = getattr(errno, "ETXTBSY", -1)  # Text file busy


## IANA tz database lookup for report timestamps.  Linux distros ship the
## tzdb system-wide, but Windows does not — stdlib ``zoneinfo`` on Windows
## requires the ``tzdata`` PyPI package (``pip install tzdata``) to resolve
## names like "America/New_York".  Resolve once at module load and fall back
## to UTC with an unambiguous label so HTML reports still generate in
## degraded environments rather than crashing inside save_matplotlib_charts_as_html.
try:
    _REPORT_TZ: datetime.tzinfo = zoneinfo.ZoneInfo("America/New_York")
    _REPORT_TZ_LABEL: str = "US Eastern"
except zoneinfo.ZoneInfoNotFoundError:
    _REPORT_TZ = datetime.timezone.utc
    _REPORT_TZ_LABEL = "UTC (install 'tzdata' for US Eastern on Windows)"


def _cleanup_stale_files(folder_path: str, max_age_seconds: int, prefix: str) -> None:
    """Remove orphaned .tmp and .time_probe_ files older than max_age_seconds.

    Uses a probe file to determine the NAS server's current time, avoiding
    false deletions caused by clock drift between the client and the Isilon node.
    If the probe cannot be created or read, cleanup is aborted entirely rather
    than falling back to the local clock — a client clock running 24+ hours ahead
    could cause active .tmp files from concurrent pipelines to be falsely deleted.

    This function is intentionally best-effort: all errors are silently suppressed
    so that a cleanup failure never blocks the main write operation.
    """
    folder = Path(folder_path)
    try:
        # --- Determine the server's current time via a disposable probe file ---
        # Why: Client and NAS clocks can differ by minutes. Using the server's
        # own mtime avoids deleting files that only *appear* old due to drift.
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
            # Cannot determine NAS server time — abort cleanup entirely.
            # Falling back to local time risks deleting active .tmp files
            # from concurrent pipelines if the client clock is ahead by 24+ hours.
            return
        finally:
            # Always attempt to remove the probe; if it fails, orphaned probes
            # are cleaned up below as part of the regular stale-file sweep.
            if probe_created:
                try:
                    Path(probe_path).unlink()
                except OSError:
                    pass  # Orphaned probe cleaned up by the stale-file sweep below

        # --- Sweep the folder for stale temp files and orphaned time probes ---
        for entry in folder.iterdir():
            f = entry.name
            # Only touch files belonging to our pipeline.  .lock files are left
            # alone because OS-level byte locks are automatically released when
            # the owning process exits or the SMB session drops.
            #
            # Strict prefix boundaries prevent cross-pipeline data loss.
            # We split on the ".parquet." boundary to isolate the base name
            # (prefix + optional date range) from the UUID suffix.  Then:
            #   Date-range branch: validate the portion after "prefix-" matches
            #     exactly two numeric blocks (with optional sign for BCE/Y10K+).
            #     This prevents prefix="sales" from matching "sales-1-..." which
            #     belongs to a pipeline with prefix="sales-1".
            #   Static branch: base_name must equal prefix exactly.
            #     This prevents prefix="sales" from matching "sales.historical".
            is_stale_tmp = False
            if f.endswith(".tmp") and ".parquet." in f:
                # rsplit from the right: the suffix (.{uuid8}.tmp) is strictly
                # controlled and never contains ".parquet.", so rsplit correctly
                # isolates the base name even if the prefix itself contains
                # ".parquet." (e.g. "data.parquet.v1").
                base_name = f.rsplit(".parquet.", 1)[0]
                if base_name == prefix:
                    is_stale_tmp = True  # Static filename match
                elif base_name.startswith(f"{prefix}-"):
                    date_part = base_name[len(prefix) + 1 :]
                    # Two numeric blocks (≥6 digits each) with optional sign.
                    # The 6-digit minimum matches strftime("%Y%m") output which
                    # always produces ≥6 chars (4-digit year + 2-digit month,
                    # zero-padded).  This prevents prefix="sales" from claiming
                    # files belonging to prefix="sales-123-456" (3-digit blocks).
                    if re.match(r"^[+-]?\d{6,}-[+-]?\d{6,}$", date_part):
                        is_stale_tmp = True
            is_orphaned_probe = f.startswith(".time_probe_")
            if not (is_stale_tmp or is_orphaned_probe):
                continue

            try:
                if server_now - entry.stat().st_mtime > max_age_seconds:
                    entry.unlink()
            except OSError:
                pass  # File may have been removed by another process; ignore
    except OSError:
        pass  # Entire cleanup is best-effort; never block the caller


def _validate_inputs(
    dataframe: pl.DataFrame, write_directory: str, subfolder: str, file_name_prefix: str
) -> None:
    """Validate all inputs before attempting to save.

    Raises ValueError / TypeError with a descriptive message for any input
    that would cause a confusing downstream failure (empty DataFrame, bad
    path characters, whitespace in components, etc.).
    """
    # Fail fast on LazyFrames: pl.LazyFrame has no .is_empty(), so passing one
    # would throw a cryptic AttributeError instead of a clear type error.
    if not isinstance(dataframe, pl.DataFrame):
        raise TypeError(
            f"Expected eager polars.DataFrame, got {type(dataframe).__name__}. "
            f"Call .collect() first."
        )

    if dataframe.is_empty():
        raise ValueError("Cannot save an empty DataFrame")

    # --- write_directory: must exist on disk right now ---
    if not write_directory or not write_directory.strip():
        raise ValueError("write_directory must not be empty or whitespace")
    if not Path(write_directory).is_dir():
        raise ValueError(f"write_directory does not exist: {write_directory}")

    # --- subfolder: no whitespace padding, no illegal chars per component ---
    if not subfolder or not subfolder.strip():
        raise ValueError("subfolder must not be empty or whitespace")
    _subfolder_parts = subfolder.replace("\\", "/").split("/")
    for _part in _subfolder_parts:
        if _part and _part != _part.strip():
            raise ValueError(
                f"subfolder components must not have leading/trailing whitespace: {subfolder!r}"
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

    # --- file_name_prefix: no whitespace, no separators, no illegal chars ---
    # Explicit check for both separators rather than os.path.basename(), which
    # only treats "\" as a separator on Windows.  This ensures identical
    # validation behaviour on both Windows and Linux.
    if (
        not file_name_prefix
        or file_name_prefix != file_name_prefix.strip()
        or "/" in file_name_prefix
        or "\\" in file_name_prefix
    ):
        raise ValueError(
            "file_name_prefix must not be empty, cannot have leading/trailing whitespace, and cannot contain path separators"
        )
    if _WINDOWS_ILLEGAL_CHARS.search(file_name_prefix):
        raise ValueError(
            f"file_name_prefix contains characters illegal in Windows filenames: {file_name_prefix!r}"
        )
    if _WINDOWS_RESERVED_NAMES.match(file_name_prefix.split(".")[0]):
        raise ValueError(
            f"file_name_prefix uses a reserved Windows device name: {file_name_prefix!r}"
        )

    # NOTE: Basename byte-length and Windows MAX_PATH checks are performed in
    # save_results() AFTER _build_filename() generates the exact filenames.
    # This avoids hardcoding suffix length estimates that break for extended
    # dates (year 10000+, BCE) where strftime produces 7+ character tokens.


def _resolve_folder_path(write_directory: str, subfolder: str) -> str:
    """Build the target folder path with traversal security check.

    Resolves symlinks via Path.resolve() and then verifies that the
    resulting path is still inside write_directory.  This prevents a
    crafted subfolder like '../../etc' from escaping the intended root.
    """
    base_dir = str(Path(write_directory).resolve())

    # Normalize backslashes to forward slashes so that subfolder paths like
    # "data\output" are treated identically on both Windows (where \ is a
    # path separator) and Linux (where \ is a literal filename character).
    # Then strip leading slashes so the join treats it as relative, not as
    # an absolute path that would silently override base_dir.
    subfolder_safe = subfolder.replace("\\", "/").lstrip("/")
    if not subfolder_safe:
        raise ValueError(
            f"subfolder resolves to empty after stripping leading slashes: {subfolder!r}"
        )
    folder_path = str((Path(base_dir) / subfolder_safe).resolve())

    # Traversal check: the resolved folder must be inside (or equal to) base_dir.
    # normcase ensures case-insensitive comparison on Windows (C:\Foo == c:\foo).
    # is_relative_to handles cross-drive paths (C: vs D:) by returning False.
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
    """Build the output filename and duplicate-detection pattern.

    When sort_by_date_column is provided:
      - Validates the column exists and is Date or Datetime (not all-null).
      - Sorts the DataFrame ascending by that column for parquet row-group ordering.
      - Embeds the min/max YYYYMM range in the filename: prefix-YYYYMM-YYYYMM.parquet
      - Returns a regex pattern to detect older date-range files for cleanup.

    When sort_by_date_column is None:
      - Uses a static filename: prefix.parquet
      - Returns exact_match=True so duplicate cleanup matches by name only.

    Returns (sorted_dataframe, file_name, dup_pattern, exact_match).
    """
    if sort_by_date_column is not None:
        if sort_by_date_column not in dataframe.columns:
            raise ValueError(
                f"Column '{sort_by_date_column}' not found. Available: {dataframe.columns}"
            )

        col_dtype = dataframe.schema[sort_by_date_column]

        # base_type() strips parameterization: Datetime("us") → Datetime, so
        # the comparison works for all time-unit variants.
        base_dtype = col_dtype.base_type()
        if base_dtype not in (pl.Date, pl.Datetime):
            raise TypeError(
                f"Column '{sort_by_date_column}' expected Date or Datetime, got {col_dtype}"
            )

        # Guard against an all-null column: min()/max() would return None, and
        # strftime would produce "None" in the filename.
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
                f"Column '{sort_by_date_column}' min/max resolved to null "
                f"despite non-null values existing"
            )

        # Example filename: price-volume-202001-202412.parquet
        file_name = f"{file_name_prefix}-{first}-{last}.parquet"

        # Supersession pattern: match any file with the same prefix and start
        # date, regardless of end date.  The pipeline always writes the full
        # range from a fixed start date to the current month, so each new file
        # fully contains the data of any older file that shares the same start.
        #
        # Example: writing prefix-201401-202504.parquet should delete
        #          prefix-201401-202503.parquet (same start, older end)
        #          but NOT prefix-201301-202503.parquet (different start).
        escaped_prefix = re.escape(file_name_prefix.lower())
        escaped_first = re.escape(first)
        dup_pattern = re.compile(
            rf"^{escaped_prefix}-{escaped_first}-[+-]?\d{{6,}}\.parquet$"
        )
        return dataframe, file_name, dup_pattern, False
    else:
        # No date column: static filename, exact-match duplicate detection
        return dataframe, f"{file_name_prefix}.parquet", None, True


def _remove_duplicates(
    folder_path: str,
    file_name: str,
    file_path: str,
    dup_pattern: re.Pattern[str] | None,
    exact_match: bool,
) -> None:
    """Remove superseded parquet files after a successful write.

    Called while holding the FileLock, so no other writer for this prefix
    can create new files during the scan.

    Supersession logic:
      - Date-range filenames: the pipeline always writes the full range from
        a fixed start date to the current month.  When we write
        prefix-201401-202504.parquet, any older file with the SAME prefix and
        SAME start date (e.g. prefix-201401-202503.parquet) is fully contained
        in the new file and must be deleted.  Files with a DIFFERENT start date
        are left alone — they belong to a different partition window.
      - Static filenames: exact case-insensitive name match (the new file
        atomically replaced the old one, but NTFS/SMB may show stale entries).

    Safety measures:
      - Uses os.path.samefile to avoid deleting the freshly written file when
        the directory listing returns it with different casing (SMB quirk).
      - Clears the read-only attribute before removal, because
        backup/compliance tools on Isilon may mark parquet files read-only.
      - All errors are suppressed per-file so one stuck file doesn't prevent
        cleanup of the others.
    """
    folder = Path(folder_path)
    try:
        for entry in folder.iterdir():
            f = entry.name
            # Skip the file we just wrote (case-sensitive), and non-parquet files
            if f == file_name or not f.lower().endswith(".parquet"):
                continue

            fl = f.lower()
            if exact_match:
                # Static filename: only match the exact name (case-insensitive)
                is_dup = fl == file_name.lower()
            else:
                # Date-range supersession: matches any file with the same prefix
                # and start date but ANY end date (the new file supersedes it).
                # e.g. writing prefix-201401-202504.parquet deletes
                #      prefix-201401-202503.parquet (older end date)
                is_dup = dup_pattern is not None and bool(dup_pattern.match(fl))

            if is_dup:
                try:
                    # On case-insensitive filesystems (NTFS, SMB), the same physical
                    # file can appear with different casing.  We MUST detect this and
                    # skip deletion to avoid destroying our own output.  The check
                    # uses Path.samefile (inode identity) when it doesn't fail on
                    # network paths lacking GetFileInformationByHandle support,
                    # falling back to case-insensitive name comparison otherwise.
                    is_same_file = False
                    if entry.exists():
                        try:
                            is_same_file = entry.samefile(file_path)
                        except OSError:
                            is_same_file = fl == file_name.lower()
                    if is_same_file:
                        continue  # Same file — do not delete our own output
                    # Clear read-only attribute before removal.
                    # Isilon backup/compliance tools may set this flag on
                    # both Windows (NTFS) and Linux (NFS/SMB).
                    try:
                        dattrs = entry.stat().st_mode
                        if not (dattrs & stat.S_IWRITE):
                            entry.chmod(dattrs | stat.S_IWRITE)
                    except OSError:
                        pass  # If we can't clear it, unlink below will fail (caught)
                    entry.unlink()
                    print(f"Erased duplicate: {f}")
                except OSError:
                    pass  # File may be in use by a reader or locked by AV; skip it
    except OSError as e:
        # Directory iteration itself can fail on a network path; warn but don't crash
        print(f"Warning: Could not complete duplicate cleanup in {folder_path}: {e}")


def _validate_path_lengths(tmp_path: str) -> None:
    """Verify basename and full path lengths before touching the filesystem.

    Checks:
      - POSIX 255-byte basename limit (UTF-8 encoded, covers Ext4/XFS/OneFS).
        Measured in bytes because multi-byte chars (Kanji, emojis) can pass a
        len() check but blow the byte limit (e.g. 80 emojis = 320 bytes).
      - Windows 260-character MAX_PATH limit (unless \\\\?\\ extended path prefix).
    """
    _MAX_BASENAME_BYTES = 255
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
            f"Generated path ({len(tmp_path)} chars) exceeds Windows MAX_PATH limit ({_WIN_MAX_PATH}). "
            f"Shorten write_directory, subfolder, or file_name_prefix."
        )


def _makedirs_with_retry(folder_path: str) -> None:
    """Create the target folder, retrying up to 12× for transient NAS errors.

    Uses exponential backoff (0.5s … 15s cap) between retries — matches the
    envelope of _atomic_replace and _write_and_fsync's fsync phase.  Filters
    on the same transient errno set as _atomic_replace: permission and
    sharing violations retry; path-shape errors (ENOTDIR, ENOENT, etc.) fail
    fast so a misconfigured path doesn't stall the pipeline for ~100s.  The
    final attempt lets the OSError propagate so the caller sees the real
    failure.
    """
    for _mkdir_attempt in range(12):
        try:
            Path(folder_path).mkdir(parents=True, exist_ok=True)
            return
        except OSError as e:
            win_err = getattr(e, "winerror", 0)
            posix_err = getattr(e, "errno", 0)
            if win_err not in (5, 32, 33) and posix_err not in (
                errno.EACCES,
                errno.EBUSY,
                errno.EAGAIN,
                _ESTALE,
                _ETXTBSY,
            ):
                raise  # Not a transient NAS/permission error — fail fast
            if _mkdir_attempt == 11:
                raise  # Exhausted all retries
            time.sleep(min(15.0, 0.5 * (2**_mkdir_attempt)))  # Backoff: 0.5s … 15s cap


def _fsync_and_verify_size(tmp_path: str) -> int:
    """Re-open tmp_path with O_RDWR, fsync, return file size via os.fstat.

    Shared fsync path for _write_and_fsync (parquet) and _write_bytes_and_fsync
    (HTML, other byte buffers).  Keeps the 12× retry envelope and read-only
    attribute clearing in one place.

    Uses os.fstat on the open handle to bypass the SMB directory metadata cache
    (the cache can return stale sizes for FileInfoCacheLifetime seconds — 10s
    default, up to 60s on Isilon).

    Retries up to 12× with 0.5s → 15s backoff for transient AV/DLP/SMB locks
    (WinError 5/32/33, EACCES/EBUSY/EAGAIN/ESTALE/ETXTBSY).  On any failure
    (non-transient errno, retry exhaustion, or zero-byte result), the partial
    tmp file is removed before propagating so the caller never sees a stale
    artifact.

    Returns the verified size in bytes.

    Raises
    ------
    OSError
        * fsync fails with a non-transient errno
        * fsync fails 12× consecutively with a transient errno
        * fstat reports zero bytes (disk full / truncation / fsync lie)
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
            break  # fsync succeeded
        except OSError as e:
            win_err = getattr(e, "winerror", 0)
            posix_err = getattr(e, "errno", 0)
            # Clear read-only attribute if a NAS compliance/DLP tool set it.
            if win_err == 5 or posix_err == errno.EACCES:
                try:
                    tmp_p = Path(tmp_path)
                    attrs = tmp_p.stat().st_mode
                    if not (attrs & stat.S_IWRITE):
                        tmp_p.chmod(attrs | stat.S_IWRITE)
                except OSError:
                    pass  # Best-effort chmod; next retry will surface the real error
            if win_err not in (5, 32, 33) and posix_err not in (
                errno.EACCES,
                errno.EBUSY,
                errno.EAGAIN,
                _ESTALE,
                _ETXTBSY,
            ):
                # Not a transient lock — clean up partial tmp and propagate.
                try:
                    Path(tmp_path).unlink()
                except OSError:
                    pass  # Best-effort cleanup; tmp may already be gone
                raise
            if _fsync_attempt == 11:
                # Exhausted all retries — clean up partial tmp and propagate.
                try:
                    Path(tmp_path).unlink()
                except OSError:
                    pass  # Best-effort cleanup; tmp may already be gone
                raise
            time.sleep(min(15.0, 0.5 * (2**_fsync_attempt)))  # Backoff: 0.5s … 15s cap

    if tmp_size <= 0:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass  # Best-effort cleanup; tmp may already be gone
        raise OSError(
            f"fsync produced a zero-byte or unmeasured file: {tmp_path}"
        )
    return tmp_size


def _write_and_fsync(dataframe: pl.DataFrame, tmp_path: str) -> int:
    """Write DataFrame to a temp parquet file and fsync to NAS backend.

    The write_parquet call is retried up to 4× for transient NAS errors
    (network drops, SMB sharing violations, NFS stale handles).  Each retry
    removes the corrupted partial file first — parquet is not appendable, so
    a fresh write from scratch is required.

    After write_parquet, re-opens the file with O_RDWR to fsync and capture
    the file size via os.fstat (bypassing the SMB directory metadata cache).

    O_RDWR is required because Windows FlushFileBuffers needs write access;
    O_RDONLY causes EBADF on Windows.

    The fsync phase retries up to 12× for transient AV/DLP locking errors:
    after write_parquet closes the file, scanners (Defender, CrowdStrike)
    or Isilon SmartPool indexers may briefly lock it for inspection.  Their
    read lock denies O_RDWR access, causing WinError 32 (sharing violation)
    on Windows or EACCES/EBUSY on Linux.  Dashboard queries and enterprise
    AV scans can hold locks for tens of seconds, so the retry window is
    generous.

    Returns the verified file size in bytes.

    Raises
    ------
    RuntimeError
        * write_parquet raises any non-OSError exception (polars
          ComputeError / SchemaError / PanicException, OOM, etc.).  The
          partial tmp file is cleaned up first; the original exception is
          chained via ``__cause__``.
    OSError
        * write_parquet fails 4× consecutively with a transient I/O error
        * fsync fails either with a non-retriable error code or 12× in a row
        * the final file is zero-byte or unmeasurable
        In all cases the partial tmp file is removed before propagating.
    """
    # Retry write_parquet for transient NAS I/O errors (network drops, SMB
    # sharing violations, NFS stale handles).  Before each retry, remove the
    # corrupted partial .tmp file — parquet format is not appendable, so
    # write_parquet must start from a clean slate.  Non-I/O errors (polars
    # ComputeError / SchemaError / PanicException / OOM) are not retried:
    # they are almost always deterministic, so retrying wastes the backoff
    # window.  Instead they are wrapped in RuntimeError with input context
    # and the partial tmp file is cleaned up first — matches the pattern in
    # winsorized_rolling_stats and save_results.
    for _write_attempt in range(4):
        try:
            dataframe.write_parquet(tmp_path)
            break
        except OSError:
            # Remove corrupted partial file before retrying.
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass  # File may not have been created yet; ignore
            if _write_attempt == 3:
                raise  # Exhausted all retries
            time.sleep(min(15.0, 1.0 * (2**_write_attempt)))  # Backoff: 1s, 2s, 4s
        except Exception as e:
            # Broad catch for non-OSError polars/arrow errors — never
            # swallowed, always re-raised with context per CLAUDE.md.
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass  # Best-effort cleanup; tmp may not exist yet
            raise RuntimeError(
                f"_write_and_fsync: polars write_parquet failed "
                f"(tmp_path={tmp_path!r}, n_rows={dataframe.height}, "
                f"n_cols={dataframe.width}). Underlying error: "
                f"{type(e).__name__}: {e}"
            ) from e

    # Fsync + size verification are delegated to the shared helper so the
    # parquet and byte-buffer paths share one tested retry envelope.  The
    # helper removes the partial tmp file on any failure before propagating.
    return _fsync_and_verify_size(tmp_path)


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
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass  # Best-effort cleanup; tmp may not exist yet
            if _write_attempt == 3:
                raise  # Exhausted write retries
            time.sleep(min(15.0, 1.0 * (2**_write_attempt)))  # Backoff: 1s, 2s, 4s

    # Fsync phase: delegated to shared helper.  Partial-tmp cleanup on failure
    # is handled by the helper.
    tmp_size = _fsync_and_verify_size(tmp_path)

    # Size verification against caller-provided expectation.
    if tmp_size != expected_size:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass  # Best-effort cleanup; tmp may already be gone
        raise OSError(
            f"Post-write size mismatch: expected {expected_size} bytes, "
            f"got {tmp_size} bytes for {tmp_path}"
        )
    return tmp_size


def _atomic_replace(tmp_path: str, file_path: str) -> None:
    """Atomically rename tmp_path to file_path with retry for transient NAS errors.

    Retries up to 12× for:
      - WinError 32/33: SMB sharing/lock violation (another reader has the file open)
      - EACCES/EBUSY/EAGAIN: POSIX transient permission or locking errors
      - ESTALE: NFS stale file handle after server failover
      - ETXTBSY: File is being executed (rare, but possible on NFS)

    Clears the read-only attribute on the target file if Access Denied
    (WinError 5 / EACCES) suggests it was set by backup/compliance tools.
    """
    file_p = Path(file_path)
    for _attempt in range(12):
        try:
            Path(tmp_path).replace(file_path)
            return
        except OSError as e:
            win_err = getattr(e, "winerror", 0)
            posix_err = getattr(e, "errno", 0)
            if win_err not in (5, 32, 33) and posix_err not in (
                errno.EACCES,
                errno.EBUSY,
                errno.EAGAIN,
                _ESTALE,
                _ETXTBSY,
            ):
                raise
            if _attempt == 11:
                raise  # Exhausted all retries
            # Access Denied can mean the target file is read-only.
            # Backup/compliance tools on Isilon may set this attribute
            # on both Windows (WinError 5) and Linux (EACCES on NFS/SMB).
            # Clear it before the next retry so Path.replace can overwrite.
            if (win_err == 5 or posix_err == errno.EACCES) and file_p.is_file():
                try:
                    attrs = file_p.stat().st_mode
                    if not (attrs & stat.S_IWRITE):
                        file_p.chmod(attrs | stat.S_IWRITE)
                except OSError:
                    pass  # Best-effort chmod; next retry will surface the real error
            time.sleep(min(15.0, 0.5 * (2**_attempt)))  # Backoff: 0.5s … 15s cap


def _verify_written_size(file_path: str, expected_size: int) -> None:
    """Verify that the written file matches the expected size.

    Uses os.fstat(fd) on an open handle to bypass the SMB directory metadata
    cache, which can return the OLD file's size for up to
    FileInfoCacheLifetime seconds (default 10s, often tuned to 30-60s on
    enterprise Isilon clusters).  os.fstat on an open handle queries the
    NAS directly (IRP on Windows, GETATTR RPC on NFS), guaranteeing fresh
    metadata regardless of local cache TTL.

    Retries up to 12× with 1-second intervals for transient AV/DLP locks on
    os.open.  If a positive size is read that differs from expected, stops
    retrying immediately — fstat on an open handle is authoritative, so a
    mismatch is a genuine different file (concurrent overwrite), not stale
    metadata.

    Raises OSError only if the file cannot be opened or read at all (size
    stays ≤ 0).  A positive size mismatch is downgraded to a printed notice
    because the atomic commit (os.replace) already succeeded — crashing here
    would cause false-negative failures in the orchestrator.
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
            # fstat on an open handle is authoritative (bypasses SMB cache).
            # A positive mismatch means a concurrent worker overwrote the file
            # (Polars parquet is not byte-deterministic across threads).  No
            # amount of retrying will change the size — break immediately.
            if written_size > 0:
                break
        except OSError:
            pass  # File may briefly be locked by AV scanners; retry
        time.sleep(1.0)

    # If the file is readable with a positive size but different from expected,
    # the atomic commit succeeded and a concurrent worker safely overwrote it.
    # Downgrade to a notice: crashing would cause the orchestrator to retry
    # indefinitely against an already-committed transaction.
    if written_size > 0 and written_size != expected_size:
        print(
            f"Notice: Written size ({written_size}) differs from expected "
            f"({expected_size}) for {file_path}. This is expected if a "
            f"concurrent task safely overwrote the same partition."
        )
        return

    raise OSError(
        f"Post-write verification failed: expected {expected_size} bytes, "
        f"got {written_size} bytes for {file_path}. "
        f"File was renamed successfully but size could not be confirmed."
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

    Implements the full atomic-write pattern for NAS reliability:
      1. Validate inputs
      2. Resolve and secure the target folder path
      3. Build the output filename (with optional date-range embedding)
      4. Verify path lengths before touching the filesystem
      5. Create the folder (with retry for transient SMB errors)
      6. Clean up stale temp files from prior crashed runs
      7. Write to a unique .tmp file, fsync to flush to NAS backend
      8. Acquire a cross-process FileLock, then atomically os.replace
      9. Remove older duplicate parquet files  (inside lock)
     10. Verify the file size matches            (outside lock)
     11. Clean up the .tmp on failure; print elapsed time on success

    Args:
        dataframe:          Polars DataFrame to write (must not be empty).
        write_directory:    Root directory that must already exist on disk.
        subfolder:          Relative subfolder under write_directory (created if needed).
        file_name_prefix:   Base name for the parquet file (no extension, no separators).
        sort_by_date_column: Optional Date/Datetime column to sort by and embed in filename.

    Returns:
        The absolute path of the written parquet file.
    """
    # ── Steps 1-2: Validate inputs and resolve target folder ─────────────
    _validate_inputs(dataframe, write_directory, subfolder, file_name_prefix)
    folder_path = _resolve_folder_path(write_directory, subfolder)

    # ── Step 3: Build the output filename and assemble exact paths ────────
    # This runs BEFORE makedirs and length checks so we validate the real
    # generated filenames — not hardcoded suffix estimates that break for
    # extended dates (year 10000+, BCE) where strftime produces 7+ char tokens.
    dataframe, file_name, dup_pattern, exact_match = _build_filename(
        dataframe, file_name_prefix, sort_by_date_column
    )
    folder = Path(folder_path)
    file_path = str(folder / file_name)
    tmp_path = (
        f"{file_path}.{uuid.uuid4().hex[:8]}.tmp"  # Random suffix prevents collisions
    )
    lock_path = str(folder / f".{file_name_prefix}.lock")

    # ── Steps 4-6: Pre-flight checks and folder setup ────────────────────
    _validate_path_lengths(tmp_path)
    _makedirs_with_retry(folder_path)
    # 24-hour threshold prevents deleting large files still being written
    _cleanup_stale_files(folder_path, max_age_seconds=86400, prefix=file_name_prefix)

    start_time = time.monotonic()
    rename_done = (
        False  # Tracks whether os.replace succeeded (controls finally cleanup)
    )

    try:
        # ── Step 7: Write parquet and fsync to NAS backend ───────────────
        # Runs OUTSIDE the lock so concurrent workers writing different date
        # ranges for the same prefix can stream their data simultaneously.
        # Each worker writes to a UUID-suffixed .tmp file, so no collisions.
        # The lock only wraps the fast metadata operations (Steps 8-10).
        tmp_size = _write_and_fsync(dataframe, tmp_path)

        # ── Steps 8-9: Acquire lock for fast metadata operations only ────
        # FileLock uses OS-level byte-range locks (msvcrt.locking on Windows,
        # fcntl.flock on POSIX), which are released automatically if the
        # process crashes or the SMB session drops.  The lock wraps only the
        # atomic rename and duplicate cleanup — millisecond-level operations —
        # so concurrent writers are never blocked during heavy I/O.
        #
        # On POSIX, pre-create the lock file with broad write permissions so
        # different service accounts can acquire the byte lock.  The umask may
        # restrict the initial creation to 644; os.chmod fixes it immediately.
        # A microsecond race exists between os.open and os.chmod where a
        # concurrent worker's FileLock open could fail with PermissionError.
        # This is handled by retrying lock acquisition below — NOT by
        # manipulating os.umask, which is process-global and thread-unsafe
        # (would cause unrelated threads to create world-writable files).
        if os.name == "posix":
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY, 0o666)
                # On NFS, os.close triggers an implicit CLOSE RPC that can
                # fail with EIO if pending metadata flush hits a quota or
                # network error.  Wrapping close in try/finally ensures chmod
                # fires regardless — it operates on the path, not the fd.
                try:
                    os.close(fd)
                finally:
                    # chmod sets exact permissions regardless of umask.
                    # Only succeeds if we own the file (we created it first).
                    Path(lock_path).chmod(0o666)
            except OSError:
                pass  # Best-effort; FileLock will report the real error

        # Acquire the lock with a PermissionError retry.  Uses `with lock:`
        # instead of manual acquire/release to guarantee CPython atomicity:
        # SETUP_WITH registers __exit__ before the body executes, so an async
        # cancellation (KeyboardInterrupt, CancelledError) between acquisition
        # and the body cannot leak the cross-process byte lock.  The
        # _lock_acquired flag discriminates PermissionError from lock.__enter__
        # (transient umask race → retry) vs from inner operations (real → raise).
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
                break  # Context manager safely exited and released the lock
            except PermissionError:
                if _lock_acquired:
                    # Lock was held; PermissionError came from _atomic_replace
                    # or _remove_duplicates.  __exit__ already released the lock.
                    raise
                # Transient: another worker created the lock file with a
                # restrictive umask and hasn't run chmod yet.
                if time.monotonic() - _lock_start > _LOCK_TIMEOUT_SECONDS:
                    raise TimeoutError(
                        f"Cannot acquire lock file after "
                        f"{_LOCK_TIMEOUT_SECONDS}s: {lock_path}"
                    ) from None
                time.sleep(0.1)

        # ── Step 10: Verify file size (OUTSIDE the lock) ────────────────
        # Size verification reads only the file we just renamed — no
        # cross-process exclusion is needed.  Running it inside the lock
        # would cause priority inversion: if AV scanners delay the open,
        # the 12-second retry window compounds across queued workers
        # (e.g. 50 workers × 12s = 600s), starving trailing workers past
        # _LOCK_TIMEOUT_SECONDS and crashing the pipeline.
        _verify_written_size(file_path, tmp_size)

    finally:
        # ── Step 11: Clean up the .tmp file if rename never happened ─────
        # After a successful os.replace, tmp_path no longer exists (renamed).
        # We only attempt removal when rename_done is False (write failed,
        # lock timed out, or verification failed before rename).
        if not rename_done:
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass  # tmp may already be gone or still locked; stale cleanup handles it

    elapsed = (time.monotonic() - start_time) / 60
    print(f"Results saved to: {file_path}, Time taken (minutes): {elapsed:.2f}")
    return file_path

#################################################################################################
# Function for reading SQL data

__all__ = ["sql_query", "SqlIngestError"]

class SqlIngestError(RuntimeError):
    """Boundary-contract failure while ingesting from SQL Server."""

# Characters that would let an attacker (or a malformed config value) escape
# a DRIVER={...} / SERVER=... / DATABASE=... segment and inject arbitrary
# ODBC connection-string keys.
_CONN_STR_FORBIDDEN = (";", "{", "}", "\x00", "\n", "\r")

# Generous but finite upper bounds to catch typos without over-constraining.
_MAX_BATCH_SIZE = 10_000_000
_MAX_TIMEOUT_SEC = 86_400  # 1 day; beyond this the caller almost certainly meant "no timeout"
_MAX_ROWS_CAP = 10**12  # trillion-row hard ceiling on the caller-supplied max_rows
_MAX_VALUE_BYTES = 2**31 - 1  # 2 GiB; SQL Server's per-value MAX-type upper bound

# Whitelist of `Authentication=...` values accepted by the Microsoft SQL
# ODBC driver. A whitelist (not free-form str) is required because this
# value is interpolated into the connection string; allowing arbitrary
# text would reopen the conn-string injection surface we lock down with
# _CONN_STR_FORBIDDEN.
_AUTH_REQUIRES_CREDS = frozenset(
    {
        "ActiveDirectoryPassword",
        "ActiveDirectoryServicePrincipal",
        "SqlPassword",
    }
)
_AUTH_NO_CREDS = frozenset(
    {
        "ActiveDirectoryIntegrated",
        "ActiveDirectoryMsi",
        "ActiveDirectoryDefault",
        "ActiveDirectoryInteractive",
    }
)
_AUTHENTICATION_VALUES = _AUTH_REQUIRES_CREDS | _AUTH_NO_CREDS


def _check_conn_token(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SqlIngestError(
            f"{name} must be a non-empty, non-whitespace str, got "
            f"{type(value).__name__}={value!r}"
        )
    # Leading/trailing whitespace is silently tolerated by some ODBC driver
    # managers and rejected by others; rather than silently strip (which
    # would hide caller-side typos like copy-pasting from docs), reject at
    # the boundary with an actionable message.
    if value != value.strip():
        raise SqlIngestError(
            f"{name} has leading/trailing whitespace: {value!r}. "
            "Strip at the call site for consistent cross-platform behavior."
        )
    for ch in _CONN_STR_FORBIDDEN:
        if ch in value:
            raise SqlIngestError(
                f"{name} contains forbidden character {ch!r}: {value!r}"
            )
    return value


@overload
def _check_int(
    name: str,
    value: int | None,
    *,
    minv: int,
    maxv: int,
    allow_none: Literal[False] = ...,
) -> int: ...


@overload
def _check_int(
    name: str,
    value: int | None,
    *,
    minv: int,
    maxv: int,
    allow_none: Literal[True],
) -> int | None: ...


def _check_int(
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
        raise SqlIngestError(f"{name} must not be None")
    # bool is an int subclass in Python; reject explicitly to prevent
    # `port=True` silently meaning 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise SqlIngestError(
            f"{name} must be int, got {type(value).__name__}={value!r}"
        )
    if not (minv <= value <= maxv):
        raise SqlIngestError(
            f"{name} must be in [{minv}, {maxv}], got {value}"
        )
    return value


def _check_bool(name: str, value: bool) -> None:
    # Guard against truthy-string / int-1 sneaking in through kwargs.
    if not isinstance(value, bool):
        raise SqlIngestError(
            f"{name} must be bool, got {type(value).__name__}={value!r}"
        )


def _check_optional_bool(name: str, value: bool | None) -> None:
    if value is None:
        return
    _check_bool(name, value)


def _check_optional_str(name: str, value: str | None) -> None:
    if value is not None and not isinstance(value, str):
        raise SqlIngestError(
            f"{name} must be str or None, got {type(value).__name__}={value!r}"
        )


def _check_no_control_chars(name: str, value: str) -> None:
    """Reject NUL / LF / CR inside a credential or similar token.

    These are almost always artifacts of reading the value from stdin or
    from a text file without stripping the trailing newline, and the
    downstream ODBC error message differs by platform (Windows returns
    SQLSTATE 28000 while unixODBC on Linux typically returns IM002), so
    catch them at our boundary with an actionable message.
    """
    for bad in ("\x00", "\n", "\r"):
        if bad in value:
            raise SqlIngestError(
                f"{name} contains a control character {bad!r}; commonly "
                "seen when reading credentials from stdin or a file "
                "without stripping a trailing newline"
            )


def _find_duplicates(names: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    dupes: list[str] = []
    for n in names:
        if n in seen and n not in dupes:
            dupes.append(n)
        seen.add(n)
    return sorted(dupes)


def _datetime_tz(dtype: pl.DataType) -> str | None:
    """Return the timezone of a Polars Datetime dtype, or None if naive.

    Polars renamed the attribute from ``tz`` to ``time_zone`` around 0.18;
    check both so older installs don't crash with ``AttributeError`` when
    the caller opts into ``require_tz_aware_timestamps``.
    """
    tz = getattr(dtype, "time_zone", None)
    if tz is None:
        tz = getattr(dtype, "tz", None)
    return tz


def _normalize_server(server: str) -> tuple[str, bool]:
    """Return ``(server_value, emit_verbatim)``.

    When ``emit_verbatim`` is True the caller must build
    ``SERVER={server_value}`` with no ``tcp:`` prefix and no trailing
    ``,port``. This covers three cases:

      * SQL Server named instances (``HOST\\INSTANCE``), which resolve via
        the SQL Browser service and must not carry an explicit port.
      * Windows-only protocol prefixes — ``np:`` (named pipes), ``lpc:``
        (shared memory / local procedure call), ``admin:`` (Dedicated
        Admin Connection) — which specify a transport inline and must be
        passed through unchanged. On non-Windows these paths fail at the
        driver layer with a clear ODBC error, which is the intended
        cross-platform behavior.

    For the common ``tcp:HOST`` form we strip the redundant prefix so
    the caller doesn't emit ``tcp:tcp:HOST``. Stripping happens BEFORE
    the non-TCP prefix / named-instance detection so callers who
    programmatically prepend ``tcp:`` to e.g. ``np:HOST`` or
    ``HOST\\INSTANCE`` still route correctly.
    """
    s = server
    low = s.lower()
    # Strip redundant leading tcp: prefixes first — otherwise tcp:np:HOST,
    # tcp:HOST\\INSTANCE, and the pathological tcp:tcp:HOST all bypass
    # the verbatim-emit branches below. Loop (rather than strip once) so
    # multi-layer programmatic prepends normalize cleanly.
    while low.startswith("tcp:"):
        s = s[4:]
        if not s.strip():
            raise SqlIngestError("server is empty after stripping 'tcp:' prefix")
        low = s.lower()
    # Windows-only protocol prefixes — keep the prefix, no tcp:/port.
    for pfx in ("np:", "lpc:", "admin:"):
        if low.startswith(pfx):
            return s, True
    return s, ("\\" in s)


def _validate_parameters(
    parameters: Sequence[str | None] | None,
) -> list[str | None] | None:
    if parameters is None:
        return None
    params = list(parameters)
    for i, p in enumerate(params):
        if p is not None and not isinstance(p, str):
            raise SqlIngestError(
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
    out = list(expected_columns)
    for i, c in enumerate(out):
        if not isinstance(c, str) or not c:
            raise SqlIngestError(
                f"expected_columns[{i}] must be non-empty str, got "
                f"{type(c).__name__}={c!r}"
            )
    dupes = _find_duplicates(out)
    if dupes:
        raise SqlIngestError(f"expected_columns contains duplicates: {dupes}")
    return out


def sql_query(
    server: str,
    database: str,
    query: str,
    *,
    parameters: Sequence[str | None] | None = None,
    driver: str = "ODBC Driver 17 for SQL Server",
    port: int = 1433,
    user: str | None = None,
    password: str | None = None,
    batch_size: int = 100_000,
    login_timeout_sec: int = 30,
    query_timeout_sec: int | None = None,
    max_rows: int | None = None,
    max_text_size: int | None = None,
    max_binary_size: int | None = None,
    encrypt: bool | None = None,
    trust_server_certificate: bool | None = None,
    authentication: str | None = None,
    expected_columns: Sequence[str] | None = None,
    allow_extra_columns: bool = False,
    require_non_empty: bool = False,
    require_tz_aware_timestamps: bool = False,
) -> pl.DataFrame:
    """Run a SELECT against SQL Server and return a Polars DataFrame.

    Parameters are bound via ODBC (``parameters``), never via string
    formatting, so callers should always prefer ``?`` placeholders over
    f-strings in ``query``. All failures surface as ``SqlIngestError``.

    Memory: the full result set is materialized in memory. For multi-GB
    queries, shrink the server-side projection or filter before calling,
    or pass ``max_rows`` to abort early.

    ``max_rows`` (opt-in): caps how many rows the stream is allowed to
    accumulate before we abort with ``SqlIngestError``. The abort fires
    during streaming, so at most one extra ``batch_size`` chunk of rows
    is buffered beyond the cap before memory is released.

    ``max_text_size`` / ``max_binary_size`` (opt-in, bytes): forwarded to
    ``arrow_odbc`` to size the buffers it allocates for unbounded SQL
    types (``VARCHAR(MAX)``, ``NVARCHAR(MAX)``, ``VARBINARY(MAX)``, etc.).
    arrow_odbc's default behavior for these types is version-dependent
    and has historically truncated to ~1 MB or ~4096 characters. If your
    SELECT returns MAX-type columns, set these explicitly to the maximum
    value size you expect; otherwise silent truncation is possible.

    ``encrypt`` / ``trust_server_certificate`` (opt-in): emitted as
    ``Encrypt=yes|no`` and ``TrustServerCertificate=yes|no`` in the ODBC
    connection string. ``Encrypt`` is ALWAYS emitted explicitly — when
    the caller passes ``encrypt=None`` we pin ``Encrypt=no`` so that
    ``driver="ODBC Driver 17 for SQL Server"`` and
    ``driver="ODBC Driver 18 for SQL Server"`` produce identical
    connection semantics for an otherwise-unchanged call. (Driver 18
    flipped the implicit default from ``no`` to ``yes``; without the
    pin a bare driver-string swap would silently enable TLS + server
    cert validation and typically fail against servers with
    self-signed or internal-CA certificates.) Pass ``encrypt=True`` to
    opt into TLS, and with either driver set
    ``trust_server_certificate=True`` as well if the server presents a
    self-signed or internal-CA certificate. ``trust_server_certificate
    =None`` leaves the driver default in place (strict validation).

    ``authentication`` (opt-in): emitted as ``Authentication=<value>``.
    Accepted values (whitelist): ``SqlPassword``,
    ``ActiveDirectoryPassword``, ``ActiveDirectoryServicePrincipal``
    (require ``user``+``password``); ``ActiveDirectoryIntegrated``,
    ``ActiveDirectoryMsi``, ``ActiveDirectoryDefault``,
    ``ActiveDirectoryInteractive`` (must not pass ``user``/``password``).
    Use this to authenticate against Azure SQL / SQL Server via Entra ID
    from any platform. Values are case-sensitive (must match the
    canonical PascalCase above).

    Headless-auth caveats:
      * ``ActiveDirectoryInteractive`` launches a browser for OAuth —
        UNSUITABLE for servers, Docker containers, systemd services,
        cron jobs, or CI on any platform. Use ``ActiveDirectoryPassword``
        or ``ActiveDirectoryServicePrincipal`` for non-interactive
        contexts.
      * ``ActiveDirectoryIntegrated`` resolves against the platform's
        native credential source — Windows SSPI (current logged-in user)
        on Windows, Kerberos ticket cache (``KRB5CCNAME``) on Linux /
        macOS. Same config string, different prerequisites.
      * ``ActiveDirectoryMsi`` requires the process to run on an Azure
        VM / App Service / container with a managed identity assigned;
        fails outside Azure.

    Cross-platform:
      * Driver names ``"ODBC Driver 17 for SQL Server"`` and
        ``"ODBC Driver 18 for SQL Server"`` are both supported and
        produce identical connection semantics from this wrapper (see
        the ``encrypt`` section above — ``Encrypt`` is always pinned
        explicitly, so the Driver 18 default flip does not change
        behavior here). The strings are identical on Windows and
        Linux, but each driver must be installed separately: MSI on
        Windows, ``msodbcsql17`` / ``msodbcsql18`` package on Linux
        via Microsoft's apt/yum repo, Homebrew on macOS. Driver 18
        also requires the target SQL Server to negotiate TLS 1.2+;
        only relevant when ``encrypt=True`` (our default ``Encrypt=no``
        short-circuits TLS altogether).
      * Driver manager: Windows uses the built-in ODBC Driver Manager
        (case-INSENSITIVE driver lookup); Linux/macOS must use
        ``unixODBC`` (case-SENSITIVE driver lookup against
        ``/etc/odbcinst.ini`` — the Microsoft package installs the
        driver under the exact string above, so our default matches,
        but a case-typo that works on Windows will fail on Linux with
        "Data source name not found". Note that macOS defaults to
        iODBC, which is NOT supported by ``msodbcsql``; install and
        configure ``unixODBC`` on macOS.
      * The DEFAULT auth mode is ``Trusted_Connection=yes`` (triggered
        when ``user``, ``password``, and ``authentication`` are all
        ``None``). This works natively via SSPI on Windows but requires
        a valid Kerberos ticket (``kinit``) on Linux / macOS. Without a
        ticket, Linux callers should supply explicit ``user``/
        ``password`` (SQL auth) or use
        ``authentication="ActiveDirectoryPassword"`` (Entra ID).
      * Server-value protocol prefixes are preserved verbatim without
        auto-adding ``tcp:`` or ``,port``:
          - ``tcp:HOST`` — explicit TCP, works on both; we strip the
            redundant prefix so you don't get ``tcp:tcp:HOST``.
          - ``np:HOST``, ``lpc:HOST``, ``admin:HOST`` — Windows-only
            (named pipes, shared memory, DAC). Passed through; on
            Linux/macOS these will fail at the driver layer with a
            clear ODBC error.
      * Named-instance syntax (``HOST\\INSTANCE``) works on both
        platforms (SQL Browser discovery over UDP 1434).
      * ``login_timeout_sec=0`` is accepted for API symmetry but is
        interpreted differently across driver versions and platforms —
        some treat it as "fail immediately" and some as "use driver
        default (~15 s)". Prefer an explicit positive value (e.g. 30)
        for predictable behavior everywhere.

    Multiple result sets: ``arrow_odbc`` materializes only the FIRST
    result set produced by the statement; subsequent result sets from
    stored procedures that ``SELECT`` more than once are silently
    dropped. Split such procedures into per-result-set calls, or wrap
    the logic in a single SELECT.

    Named instances: if ``server`` contains ``\\`` (e.g. ``HOST\\SQLEXPRESS``)
    the ``port`` argument is ignored and the instance is resolved via the
    SQL Browser service.

    Set ``require_tz_aware_timestamps=True`` to enforce the CLAUDE.md
    inbound contract that every Datetime column carries a timezone
    (rejects SQL Server ``datetime2`` / ``datetime`` columns unless your
    query casts them to ``datetimeoffset``).

    Output contract:
      * Column order and names are preserved exactly as SQL Server returns
        them. Case matches the server's projection; callers doing asof
        joins or positional selects must match this ordering.
      * The returned DataFrame is NOT sorted — caller must sort explicitly
        before any asof join.
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
    # --- input validation (inbound boundary) ---------------------------------
    _check_conn_token("server", server)
    _check_conn_token("database", database)
    _check_conn_token("driver", driver)

    if not isinstance(query, str) or not query.strip():
        raise SqlIngestError("query must be a non-empty, non-whitespace str")
    if "\x00" in query:
        raise SqlIngestError("query contains a null byte")
    if "\ufeff" in query:
        # UTF-8 BOM (U+FEFF) leaks in from files saved as "UTF-8 with
        # BOM" (common on Windows editors / older PowerShell Set-Content)
        # and from naive concatenation of two such files. SQL Server
        # sees the U+FEFF and fails with "incorrect syntax near ''".
        # Using the "\ufeff" escape (rather than a literal invisible
        # character in the source) keeps this check robust across
        # editor re-encodings and source-file BOM handling. Reject
        # anywhere in the query, not just at the start, since mid-string
        # BOMs from file concatenation produce the same error.
        raise SqlIngestError(
            "query contains a UTF-8 BOM (U+FEFF); re-save the source "
            "file as plain UTF-8, or strip via query.replace('\\ufeff', '')"
        )

    _check_int("port", port, minv=1, maxv=65_535)
    _check_int("batch_size", batch_size, minv=1, maxv=_MAX_BATCH_SIZE)
    _check_int(
        "login_timeout_sec", login_timeout_sec, minv=0, maxv=_MAX_TIMEOUT_SEC
    )
    _check_int(
        "query_timeout_sec",
        query_timeout_sec,
        minv=0,
        maxv=_MAX_TIMEOUT_SEC,
        allow_none=True,
    )
    max_rows_val = _check_int(
        "max_rows", max_rows, minv=0, maxv=_MAX_ROWS_CAP, allow_none=True
    )
    _check_int(
        "max_text_size",
        max_text_size,
        minv=1,
        maxv=_MAX_VALUE_BYTES,
        allow_none=True,
    )
    _check_int(
        "max_binary_size",
        max_binary_size,
        minv=1,
        maxv=_MAX_VALUE_BYTES,
        allow_none=True,
    )
    _check_bool("require_non_empty", require_non_empty)
    _check_bool("allow_extra_columns", allow_extra_columns)
    _check_bool("require_tz_aware_timestamps", require_tz_aware_timestamps)
    _check_optional_bool("encrypt", encrypt)
    _check_optional_bool("trust_server_certificate", trust_server_certificate)

    _check_optional_str("user", user)
    _check_optional_str("password", password)
    if (user is None) != (password is None):
        raise SqlIngestError(
            "user and password must be supplied together, or both omitted "
            "for trusted (SSPI on Windows / Kerberos on Linux & macOS) or "
            "credential-less (AAD) auth"
        )
    if user is not None:
        if not user:
            raise SqlIngestError("user must be non-empty when provided")
        _check_no_control_chars("user", user)
    if password is not None:
        if not password:
            # Empty password is almost always an unset-env-var typo; reject
            # so the caller finds out at our boundary, not via a cryptic
            # ODBC auth failure later.
            raise SqlIngestError("password must be non-empty when provided")
        _check_no_control_chars("password", password)

    _check_optional_str("authentication", authentication)
    if authentication is not None:
        if authentication not in _AUTHENTICATION_VALUES:
            raise SqlIngestError(
                f"authentication={authentication!r} not in whitelist; "
                f"allowed: {sorted(_AUTHENTICATION_VALUES)}"
            )
        if authentication in _AUTH_REQUIRES_CREDS and user is None:
            raise SqlIngestError(
                f"authentication={authentication!r} requires user and password"
            )
        if authentication in _AUTH_NO_CREDS and user is not None:
            raise SqlIngestError(
                f"authentication={authentication!r} must not be combined "
                "with user/password"
            )

    params = _validate_parameters(parameters)
    expected = _validate_expected_columns(expected_columns)
    server_canon, emit_server_verbatim = _normalize_server(server)

    # --- build connection string ---------------------------------------------
    # Trusted auth only kicks in when the caller supplied neither SQL creds
    # nor an explicit `authentication` mode. On Linux this path requires a
    # Kerberos ticket (see the Cross-platform note in the docstring).
    use_trusted = user is None and authentication is None
    if emit_server_verbatim:
        # Named instance (HOST\\INSTANCE) or Windows-only protocol prefix
        # (np:/lpc:/admin:). Both forms must not carry an auto-added
        # `tcp:` or `,PORT`.
        server_segment = f"SERVER={server_canon}"
    else:
        server_segment = f"SERVER=tcp:{server_canon},{port}"
    parts = [
        f"DRIVER={{{driver}}}",
        server_segment,
        f"DATABASE={database}",
    ]
    if use_trusted:
        parts.append("Trusted_Connection=yes")
    if authentication is not None:
        # Value is whitelist-constrained above, so safe to interpolate.
        parts.append(f"Authentication={authentication}")
    # Pin Encrypt explicitly, independent of driver version. ODBC Driver
    # 18 flipped the implicit default from `no` (Driver 17) to `yes`, so
    # an unspecified `encrypt` would silently change TLS semantics across
    # a bare driver-string swap. Defaulting to `no` here preserves Driver
    # 17's legacy behavior so callers who only change the driver string
    # continue to connect against the same server with the same semantics.
    effective_encrypt = encrypt if encrypt is not None else False
    parts.append(f"Encrypt={'yes' if effective_encrypt else 'no'}")
    if trust_server_certificate is not None:
        parts.append(
            f"TrustServerCertificate={'yes' if trust_server_certificate else 'no'}"
        )
    conn_str = ";".join(parts) + ";"

    # --- open reader ---------------------------------------------------------
    try:
        reader = read_arrow_batches_from_odbc(
            connection_string=conn_str,
            query=query,
            batch_size=batch_size,
            user=user,
            password=password,
            parameters=params,
            login_timeout_sec=login_timeout_sec,
            query_timeout_sec=query_timeout_sec,
            max_text_size=max_text_size,
            max_binary_size=max_binary_size,
        )
    except TypeError as exc:
        # arrow_odbc added `max_text_size` / `max_binary_size` /
        # `query_timeout_sec` in different releases; an "unexpected
        # keyword argument" here almost certainly means the installed
        # arrow_odbc is older than this wrapper expects.
        raise SqlIngestError(
            f"arrow_odbc rejected a keyword argument "
            f"(likely a version mismatch): {exc}"
        ) from exc
    except Exception as exc:  # arrow_odbc raises a variety of native errors
        raise SqlIngestError(
            f"failed to open ODBC reader against {server_canon}/{database}: {exc}"
        ) from exc

    if reader is None:
        # arrow_odbc returns None for statements with no result set
        # (INSERT/UPDATE/DDL). sql_query is defined only for SELECT.
        raise SqlIngestError(
            "query produced no result set; sql_query expects a SELECT"
        )

    # All reader usage (schema read, batch streaming) is wrapped so that the
    # ODBC cursor is released synchronously in every exit path — normal
    # return, caller-facing raise, or unexpected raise. In CPython this
    # triggers arrow_odbc's Rust Drop immediately, so the server-side cursor
    # is closed before the SqlIngestError propagates up the stack.
    batches: list[pa.RecordBatch] = []
    total_rows = 0
    over_cap = False
    try:
        try:
            schema = reader.schema
        except Exception as exc:
            raise SqlIngestError(
                f"failed to read arrow schema from ODBC result: {exc}"
            ) from exc
        if schema is None:
            raise SqlIngestError("ODBC result reported a null arrow schema")

        # Pre-conversion: reject duplicate arrow field names with a clear
        # error rather than letting pl.from_arrow raise an opaque
        # ComputeError.
        schema_names: list[str] = list(schema.names)
        schema_dupes = _find_duplicates(schema_names)
        if schema_dupes:
            raise SqlIngestError(
                f"duplicate column names in server result set: {schema_dupes}"
            )

        # Manual iteration (vs list(reader)) so max_rows can abort early
        # before we balloon memory on a runaway result.
        try:
            for batch in reader:
                batches.append(batch)
                total_rows += batch.num_rows
                if max_rows_val is not None and total_rows > max_rows_val:
                    over_cap = True
                    break
        except Exception as exc:
            batches.clear()
            raise SqlIngestError(
                f"failed while streaming batches from {server_canon}/{database}: {exc}"
            ) from exc

        if over_cap:
            # `over_cap = True` is only set inside the streaming loop after
            # the `max_rows_val is not None` guard, so this assert narrows
            # the type for readers and type-checkers without being a
            # load-bearing check.
            assert max_rows_val is not None
            batches.clear()
            raise SqlIngestError(
                f"result exceeded max_rows={max_rows_val}: streamed "
                f"{total_rows} rows before abort against "
                f"{server_canon}/{database}"
            )
    finally:
        del reader

    try:
        table = pa.Table.from_batches(batches=batches, schema=schema)
    except Exception as exc:
        # Release the accumulated batches before propagating; the Table
        # wasn't built, so they'd otherwise linger until GC.
        batches.clear()
        raise SqlIngestError(f"failed to assemble arrow table: {exc}") from exc

    if table.num_rows != total_rows:
        raise SqlIngestError(
            "row-count invariant violated assembling arrow Table: "
            f"sum(batch.num_rows)={total_rows} vs table.num_rows="
            f"{table.num_rows}"
        )

    # Release per-batch Python refs. The Table keeps its own zero-copy
    # references to the underlying buffers.
    batches.clear()

    # pl.from_arrow(Table) returns DataFrame; the Series branch exists for
    # Array/ChunkedArray inputs. Wrap defensively: recent Polars rejects
    # tables with duplicate arrow field names, which we'd rather surface
    # as SqlIngestError than as a raw pl.exceptions error.
    try:
        df_or_series = pl.from_arrow(table)
    except Exception as exc:
        raise SqlIngestError(
            f"failed to convert arrow Table to Polars DataFrame: {exc}"
        ) from exc
    if not isinstance(df_or_series, pl.DataFrame):
        raise SqlIngestError(
            f"expected pl.DataFrame from arrow Table, got "
            f"{type(df_or_series).__name__}"
        )
    df = df_or_series

    # --- schema & row-count preservation across arrow -> polars --------------
    cols = list(df.columns)
    if cols != schema_names:
        raise SqlIngestError(
            "schema mismatch between arrow Table and Polars DataFrame "
            f"(arrow={schema_names} polars={cols}); "
            "pl.from_arrow must not reorder or rename columns"
        )
    if df.height != total_rows:
        raise SqlIngestError(
            "row-count invariant violated in arrow->polars conversion: "
            f"expected {total_rows}, got {df.height}"
        )

    # --- post-load caller-facing checks --------------------------------------
    cols_dupes = _find_duplicates(cols)
    if cols_dupes:
        # Unreachable given the pre-conversion schema check, but kept as a
        # defense-in-depth guard against future arrow/polars behavior drift.
        raise SqlIngestError(
            f"duplicate column names in result set: {cols_dupes}"
        )

    if expected is not None:
        missing = [c for c in expected if c not in cols]
        extra = [c for c in cols if c not in expected]
        if missing or (extra and not allow_extra_columns):
            err_msg = f"schema mismatch: missing={missing}"
            if not allow_extra_columns:
                err_msg += f" extra={extra}"
            err_msg += f" got={cols}"
            raise SqlIngestError(err_msg)

    if require_tz_aware_timestamps:
        naive: list[str] = []
        for name, dtype in df.schema.items():
            if isinstance(dtype, pl.Datetime) and _datetime_tz(dtype) is None:
                naive.append(name)
        if naive:
            raise SqlIngestError(
                "require_tz_aware_timestamps=True but columns lack tz: "
                f"{naive}"
            )

    if require_non_empty and df.height == 0:
        raise SqlIngestError(
            f"query returned 0 rows against {server_canon}/{database} "
            "(require_non_empty=True)"
        )

    return df
