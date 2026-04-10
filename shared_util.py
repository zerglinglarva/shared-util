#################################################################################################
# Import Libraries
import argparse
import datetime
import errno
import os
import re
import stat
import sys
import time
import uuid
import zoneinfo
from pathlib import PurePath

import polars as pl
from filelock import FileLock
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


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
    default_write_mode: str,
    dev_folder: str,
    prod_folder: str,
) -> tuple[datetime.date, datetime.date, str]:

    # ── Step 1: Detect execution mode ────────────────────────────────────
    # Jupyter/IPython loads 'ipykernel'; CLI scripts have sys.argv[0] ending in '.py'.
    # In interactive mode argparse would choke on notebook/IDE argv, so we skip it.
    is_interactive = "ipykernel" in sys.modules or not sys.argv[0].endswith(".py")

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
    # Guard against empty folder strings before resolving.  os.path.realpath("")
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
    write_directory = os.path.realpath(write_directory)

    # Fail early if the resolved directory doesn't exist on disk
    if not os.path.isdir(write_directory):
        raise FileNotFoundError(f"Write directory does not exist: {write_directory}")

    return startdate, enddate, write_directory


###################################################################################################
# Define function to lazy scan parquet files, concatenate them into a single polars lazyframe
# and filter by date range
def lazy_parquet(
    folder_path: str,
    date_column: str,
    start_date: datetime.date,
    end_date: datetime.date,
) -> pl.LazyFrame:

    if not folder_path or not folder_path.strip():
        raise ValueError("folder_path must not be empty or whitespace")
    if not date_column or not date_column.strip():
        raise ValueError("date_column must not be empty or whitespace")

    # Resolve to absolute path so that scan_parquet paths remain valid
    # even if the caller changes the working directory before collect().
    folder_path = os.path.realpath(folder_path)

    if not os.path.isdir(folder_path):
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

    file_names = os.listdir(folder_path)
    parquet_file_names = [
        file
        for file in file_names
        if file.lower().endswith(".parquet")
        and os.path.isfile(os.path.join(folder_path, file))
    ]

    if not parquet_file_names:
        raise ValueError(f"No parquet files found in folder: {folder_path}")

    # Read schemas and check compatibility before concatenating.
    # On a shared NAS, files can vanish between listdir and schema read
    # (e.g. concurrent save_results deleting old date-range files).
    # Skip files that disappear rather than crashing with an opaque ArrowError.
    file_paths = [os.path.join(folder_path, file) for file in parquet_file_names]
    schemas: dict[str, dict[str, pl.DataType]] = {}
    for path in file_paths:
        try:
            schemas[path] = pl.read_parquet_schema(path)
        except OSError:
            continue  # File vanished or became unreadable; skip it
        except Exception as e:
            # Corrupted/truncated parquet files raise ComputeError (not OSError).
            # Warn so the user knows, but don't crash the entire function
            # when other valid files exist.
            print(f"Warning: skipping {os.path.basename(path)}: {e}")
            continue

    # Re-filter file_paths to only those whose schema was successfully read
    file_paths = [p for p in file_paths if p in schemas]

    if not file_paths:
        raise ValueError(f"No readable parquet files found in folder: {folder_path}")
    reference_path = file_paths[0]
    reference_schema = schemas[reference_path]

    ref_name = os.path.basename(reference_path)
    ref_cols = set(reference_schema.keys())

    mismatches: list[str] = []
    for path, schema in schemas.items():
        if schema == reference_schema:
            continue
        file_name = os.path.basename(path)
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
    base_dtype = getattr(col_dtype, "base_type", lambda: col_dtype)()
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
    reference_columns = list(reference_schema.keys())
    dataframes = []
    for path in file_paths:
        try:
            # Cast to Date for day-level comparison.  For Date columns this
            # is a no-op.  For Datetime columns it truncates to date, which
            # prevents the <= filter from silently excluding same-day rows
            # with non-midnight timestamps (e.g. 2024-01-15 23:59:59 would
            # be excluded by <= Date(2024-01-15) without the cast, because
            # Polars supercasts Date to Datetime midnight).
            date_expr = pl.col(date_column).cast(pl.Date)
            df = (
                pl.scan_parquet(path)
                .select(reference_columns)
                .filter(
                    (date_expr >= pl.lit(start_date)) & (date_expr <= pl.lit(end_date))
                )
            )
            dataframes.append(df)
        except Exception:
            continue  # File vanished or became unreadable after schema check

    if not dataframes:
        raise ValueError(
            f"All parquet files in {folder_path} became unreadable "
            f"between schema validation and scan"
        )

    concatenated_df = pl.concat(dataframes)

    return concatenated_df

#################################################################################################
# Chart helper — eliminates repeated boilerplate across all 8 time-series plots.
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
    """
    fig, ax = plt.subplots(1, 1, figsize=(18, 6))

    for dates, values, label in data_series:
        ax.plot(dates, values, label=label)

    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.1, 0.5), borderaxespad=0.5)
    ax.tick_params(axis="x", rotation=90)

    formatter = mtick.FuncFormatter(
        lambda x, _pos: y_format.format(x=x, x_k=x / 1e3, x_b=x / 1e9)
    )
    ax.yaxis.set_major_formatter(formatter)

    # Mirror labels on right y-axis for readability
    ax_right = ax.twinx()
    ax_right.yaxis.set_major_formatter(formatter)
    ax_right.set_ylim(ax.get_ylim())

    plt.subplots_adjust(right=0.6)
    plt.show()
    plt.close()
    return fig



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
    import base64
    import html
    import io

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
    if not os.path.isdir(write_directory):
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

    if (
        not file_name_prefix
        or file_name_prefix != file_name_prefix.strip()
        or os.path.basename(file_name_prefix) != file_name_prefix
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
    file_path = os.path.join(folder_path, file_name)
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
        f"{datetime.datetime.now(tz=zoneinfo.ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S')} US Eastern</p>",
    ]

    for fig in figures:
        axes_list = fig.get_axes()
        raw_title = axes_list[0].get_title() if axes_list else ""
        safe_title = html.escape(raw_title)

        buf = io.BytesIO()
        try:
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode("ascii")
        finally:
            buf.close()

        if safe_title:
            html_parts.append(f"<h2>{safe_title}</h2>")
        html_parts.append(f'<img src="data:image/png;base64,{b64}" alt="{safe_title}">')

    html_parts.append("</body></html>")

    # ── Step 6: Atomic write — tmp then os.replace ──────────────────────
    # Write to a temp file, fsync, then atomically rename. Retries handle
    # transient AV/DLP/SMB locks that can appear between file creation and
    # fsync (same pattern as _write_and_fsync for parquet files).
    html_bytes = "\n".join(html_parts).encode("utf-8")
    expected_size = len(html_bytes)
    rename_done = False

    try:
        # Write phase — retry up to 4× for transient NAS I/O errors.
        for _write_attempt in range(4):
            try:
                # O_BINARY prevents \n→\r\n translation on Windows.
                # On POSIX the constant doesn't exist (all I/O is binary),
                # so fall back to 0 (no-op).
                fd = os.open(
                    tmp_path,
                    os.O_CREAT | os.O_WRONLY | os.O_TRUNC | getattr(os, "O_BINARY", 0),
                )
                try:
                    # os.write may return fewer bytes than requested (POSIX spec).
                    # Loop until all bytes are flushed, or raise on zero-write
                    # (which signals disk-full or broken pipe).
                    view = memoryview(html_bytes)
                    written_total = 0
                    while written_total < expected_size:
                        written = os.write(fd, view[written_total:])
                        if written == 0:
                            raise OSError(
                                f"os.write returned 0 bytes after "
                                f"{written_total}/{expected_size} — disk may be full"
                            )
                        written_total += written
                    os.fsync(fd)
                finally:
                    os.close(fd)
                break  # Write succeeded
            except OSError:
                # Remove partial temp file before retrying.
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass  # File may not have been created; ignore
                if _write_attempt == 3:
                    raise  # Exhausted all retries
                time.sleep(min(15.0, 1.0 * (2**_write_attempt)))

        # Verify written size via fstat on an open handle (bypasses SMB cache).
        fd = os.open(tmp_path, os.O_RDONLY)
        try:
            on_disk = os.fstat(fd).st_size
        finally:
            os.close(fd)
        if on_disk != expected_size:
            raise OSError(
                f"Post-write size mismatch: expected {expected_size} bytes, "
                f"got {on_disk} bytes for {tmp_path}"
            )

        # Rename phase — retry up to 4× for transient SMB sharing violations.
        for _rename_attempt in range(4):
            try:
                os.replace(tmp_path, file_path)
                rename_done = True
                break
            except OSError:
                if _rename_attempt == 3:
                    raise
                time.sleep(min(15.0, 1.0 * (2**_rename_attempt)))

    except BaseException:
        if not rename_done:
            # Best-effort cleanup of partial temp file
            try:
                os.remove(tmp_path)
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

## Windows reserved device names: CON, PRN, AUX, NUL, COM1-9, LPT1-9.
## These are forbidden as basenames regardless of extension (CON.parquet hangs).
_WINDOWS_RESERVED_NAMES = re.compile(
    r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])$", re.IGNORECASE
)

## Safe errno access: these POSIX constants may be absent on some Windows Python builds.
## Using -1 as sentinel ensures they never accidentally match a real errno value.
_ESTALE: int = getattr(errno, "ESTALE", -1)  # NFS stale file handle
_ETXTBSY: int = getattr(errno, "ETXTBSY", -1)  # Text file busy


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
    try:
        # --- Determine the server's current time via a disposable probe file ---
        # Why: Client and NAS clocks can differ by minutes. Using the server's
        # own mtime avoids deleting files that only *appear* old due to drift.
        probe_path = os.path.join(folder_path, f".time_probe_{uuid.uuid4().hex}")
        probe_created = False
        try:
            fd = os.open(probe_path, os.O_CREAT | os.O_WRONLY)
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
                    os.remove(probe_path)
                except OSError:
                    pass

        # --- Sweep the folder for stale temp files and orphaned time probes ---
        for f in os.listdir(folder_path):
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

            fpath = os.path.join(folder_path, f)
            try:
                if server_now - os.path.getmtime(fpath) > max_age_seconds:
                    os.remove(fpath)
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
    if not os.path.isdir(write_directory):
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
    if (
        not file_name_prefix
        or file_name_prefix != file_name_prefix.strip()
        or os.path.basename(file_name_prefix) != file_name_prefix
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

    Resolves symlinks via os.path.realpath and then verifies that the
    resulting path is still inside write_directory.  This prevents a
    crafted subfolder like '../../etc' from escaping the intended root.
    """
    base_dir = os.path.realpath(write_directory)

    # Strip leading slashes so os.path.join treats subfolder as relative,
    # not as an absolute path that would silently override base_dir.
    subfolder_safe = subfolder.lstrip("\\/")
    if not subfolder_safe:
        raise ValueError(
            f"subfolder resolves to empty after stripping leading slashes: {subfolder!r}"
        )
    folder_path = os.path.realpath(os.path.join(base_dir, subfolder_safe))

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

        # base_type() strips parameterization: Datetime("us") → Datetime, so the
        # comparison works for all time-unit variants.  The fallback handles
        # older Polars versions where base_type may not exist: if col_dtype is
        # already a class (e.g. pl.Date), use it directly; if it's an instance
        # (e.g. Datetime("us")), type() recovers the unparameterized class.
        if hasattr(col_dtype, "base_type"):
            base_dtype = col_dtype.base_type()
        else:
            base_dtype = col_dtype if isinstance(col_dtype, type) else type(col_dtype)
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
    try:
        for f in os.listdir(folder_path):
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
                dup_path = os.path.join(folder_path, f)
                try:
                    # On case-insensitive filesystems (NTFS, SMB), the same physical
                    # file can appear with different casing.  We MUST detect this and
                    # skip deletion to avoid destroying our own output.  The check
                    # uses samefile (inode identity) when available, falling back to
                    # case-insensitive name comparison if samefile is absent or fails
                    # on network paths lacking GetFileInformationByHandle support.
                    is_same_file = False
                    if os.path.exists(dup_path):
                        if hasattr(os.path, "samefile"):
                            try:
                                is_same_file = os.path.samefile(dup_path, file_path)
                            except OSError:
                                is_same_file = fl == file_name.lower()
                        else:
                            is_same_file = fl == file_name.lower()
                    if is_same_file:
                        continue  # Same file — do not delete our own output
                    # Clear read-only attribute before removal.
                    # Isilon backup/compliance tools may set this flag on
                    # both Windows (NTFS) and Linux (NFS/SMB).
                    try:
                        dattrs = os.stat(dup_path).st_mode
                        if not (dattrs & stat.S_IWRITE):
                            os.chmod(dup_path, dattrs | stat.S_IWRITE)
                    except OSError:
                        pass  # If we can't clear it, os.remove below will fail (caught)
                    os.remove(dup_path)
                    print(f"Erased duplicate: {f}")
                except OSError:
                    pass  # File may be in use by a reader or locked by AV; skip it
    except OSError as e:
        # os.listdir itself can fail on a network path; warn but don't crash
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
    tmp_basename_bytes = len(os.path.basename(tmp_path).encode("utf-8"))
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
    """Create the target folder, retrying up to 3× for transient NAS errors.

    Uses exponential backoff (0.5s, 1.0s) between retries.  The final
    attempt lets the OSError propagate so the caller sees the real failure.
    """
    for _mkdir_attempt in range(3):
        try:
            os.makedirs(folder_path, exist_ok=True)
            return
        except OSError:
            if _mkdir_attempt == 2:
                raise
            time.sleep(0.5 * (2**_mkdir_attempt))


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
    Raises OSError if the file is zero-byte or unmeasurable after retries.
    """
    # Retry write_parquet for transient NAS I/O errors (network drops, SMB
    # sharing violations, NFS stale handles).  Before each retry, remove the
    # corrupted partial .tmp file — parquet format is not appendable, so
    # write_parquet must start from a clean slate.  Non-I/O errors (schema
    # issues, OOM) are not caught and propagate immediately.
    for _write_attempt in range(4):
        try:
            dataframe.write_parquet(tmp_path)
            break
        except OSError:
            # Remove corrupted partial file before retrying.
            try:
                os.remove(tmp_path)
            except OSError:
                pass  # File may not have been created yet; ignore
            if _write_attempt == 3:
                raise  # Exhausted all retries
            time.sleep(min(15.0, 1.0 * (2**_write_attempt)))  # Backoff: 1s, 2s, 4s

    tmp_size = -1
    for _fsync_attempt in range(12):
        try:
            fd = os.open(tmp_path, os.O_RDWR)
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
            # WinError 5 (Access Denied) or POSIX EACCES on NFS can both
            # indicate a read-only flag rather than a transient sharing lock.
            if win_err == 5 or posix_err == errno.EACCES:
                try:
                    attrs = os.stat(tmp_path).st_mode
                    if not (attrs & stat.S_IWRITE):
                        os.chmod(tmp_path, attrs | stat.S_IWRITE)
                except OSError:
                    pass
            if win_err not in (5, 32, 33) and posix_err not in (
                errno.EACCES,
                errno.EBUSY,
                errno.EAGAIN,
                _ESTALE,
                _ETXTBSY,
            ):
                raise  # Not a transient lock — propagate immediately
            if _fsync_attempt == 11:
                raise  # Exhausted all retries
            time.sleep(min(15.0, 0.5 * (2**_fsync_attempt)))  # Backoff: 0.5s … 15s cap

    # Sanity check: ensure write_parquet actually produced a non-empty file.
    # tmp_size was read via os.fstat(fd) on the open handle, bypassing SMB cache.
    if tmp_size <= 0:
        raise OSError(
            f"write_parquet produced a zero-byte or unmeasured file: {tmp_path}"
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
    for _attempt in range(12):
        try:
            os.replace(tmp_path, file_path)
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
            # Clear it before the next retry so os.replace can overwrite.
            if (win_err == 5 or posix_err == errno.EACCES) and os.path.isfile(
                file_path
            ):
                try:
                    attrs = os.stat(file_path).st_mode
                    if not (attrs & stat.S_IWRITE):
                        os.chmod(file_path, attrs | stat.S_IWRITE)
                except OSError:
                    pass
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
            fd = os.open(file_path, os.O_RDONLY)
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
    file_path = os.path.join(folder_path, file_name)
    tmp_path = (
        f"{file_path}.{uuid.uuid4().hex[:8]}.tmp"  # Random suffix prevents collisions
    )
    lock_path = os.path.join(folder_path, f".{file_name_prefix}.lock")

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
                    os.chmod(lock_path, 0o666)
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
                os.remove(tmp_path)
            except OSError:
                pass  # tmp may already be gone or still locked; stale cleanup handles it

    elapsed = (time.monotonic() - start_time) / 60
    print(f"Results saved to: {file_path}, Time taken (minutes): {elapsed:.2f}")
    return file_path
