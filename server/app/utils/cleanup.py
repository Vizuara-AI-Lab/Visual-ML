"""
Periodic storage cleanup utility.

Removes stale files from the uploads/ and models/ directories so they
do not grow unbounded.  Subdirectory structure is always preserved —
only files are ever deleted.

Age thresholds are controlled via settings:
  UPLOAD_MAX_AGE_HOURS  – uploads/ (csvs, artifacts, tts cache …)
  MODEL_MAX_AGE_HOURS   – models/ (.joblib files …)
  CLEANUP_INTERVAL_HOURS – how often the background loop runs
"""

import asyncio
import time
from pathlib import Path
from typing import NamedTuple

from app.core.config import settings
from app.core.logging import logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class CleanupResult(NamedTuple):
    deleted: int
    freed_bytes: int
    errors: int


def _age_hours(path: Path) -> float:
    """Return how many hours ago the file was last modified."""
    return (time.time() - path.stat().st_mtime) / 3600


def _human_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes //= 1024
    return f"{num_bytes:.1f} TB"


# ---------------------------------------------------------------------------
# Core cleanup
# ---------------------------------------------------------------------------

def _clean_directory(root: Path, max_age_hours: float) -> CleanupResult:
    """
    Walk *root* recursively and delete every **file** older than
    *max_age_hours*.  Directories are never removed.

    Returns a CleanupResult with counts of deleted files, freed bytes,
    and errors.
    """
    deleted = freed = errors = 0

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            age = _age_hours(path)
            if age >= max_age_hours:
                size = path.stat().st_size
                path.unlink()
                deleted += 1
                freed += size
                logger.debug(
                    "Cleanup: deleted {} (age {:.1f}h, {})",
                    path.relative_to(root),
                    age,
                    _human_size(size),
                )
        except Exception as exc:
            errors += 1
            logger.warning("Cleanup: could not delete {}: {}", path, exc)

    return CleanupResult(deleted=deleted, freed_bytes=freed, errors=errors)


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------

async def run_cleanup_once() -> None:
    """
    Run a single cleanup pass over uploads/ and models/.
    Executes the blocking filesystem work in a thread-pool executor so the
    event loop is never blocked.
    """
    upload_dir = Path(settings.UPLOAD_DIR)
    model_dir = Path(settings.MODEL_ARTIFACTS_DIR)

    loop = asyncio.get_running_loop()

    upload_result, model_result = await asyncio.gather(
        loop.run_in_executor(
            None, _clean_directory, upload_dir, settings.UPLOAD_MAX_AGE_HOURS
        ),
        loop.run_in_executor(
            None, _clean_directory, model_dir, settings.MODEL_MAX_AGE_HOURS
        ),
    )

    total_deleted = upload_result.deleted + model_result.deleted
    total_freed = upload_result.freed_bytes + model_result.freed_bytes
    total_errors = upload_result.errors + model_result.errors

    logger.info(
        "Cleanup complete — uploads: {} deleted ({} freed) | "
        "models: {} deleted ({} freed) | errors: {}",
        upload_result.deleted,
        _human_size(upload_result.freed_bytes),
        model_result.deleted,
        _human_size(model_result.freed_bytes),
        total_errors,
    )

    if total_deleted == 0:
        logger.debug("Cleanup: nothing to remove this cycle.")


async def start_cleanup_loop() -> None:
    """
    Background coroutine that wakes up every CLEANUP_INTERVAL_HOURS and
    calls run_cleanup_once().  Designed to run as an asyncio Task.
    """
    interval_seconds = settings.CLEANUP_INTERVAL_HOURS * 3600
    logger.info(
        "Storage cleanup scheduler started — interval: {}h, "
        "upload max age: {}h, model max age: {}h",
        settings.CLEANUP_INTERVAL_HOURS,
        settings.UPLOAD_MAX_AGE_HOURS,
        settings.MODEL_MAX_AGE_HOURS,
    )

    # Run an initial pass shortly after startup so nothing stale lingers
    # even if the server was restarted.
    await asyncio.sleep(60)  # 1-minute grace period after boot
    await run_cleanup_once()

    while True:
        await asyncio.sleep(interval_seconds)
        await run_cleanup_once()
