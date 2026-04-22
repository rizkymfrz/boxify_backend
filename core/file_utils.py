"""
Boxify Backend — File System Utilities

Provides helper functions for:
  - Clearing directory contents safely
  - Securely extracting zip archives (Zip Slip protection)
  - Filtering images by supported extension
"""

import logging
import shutil
import zipfile
from pathlib import Path

from core.config import SUPPORTED_IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


def clear_directory(directory: Path) -> None:
    """
    Remove all files and subdirectories inside *directory*, but keep
    the directory itself.

    Args:
        directory: Path to the directory whose contents should be deleted.
    """
    if not directory.exists():
        logger.warning("clear_directory called on non-existent path: %s", directory)
        return

    for child in directory.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except OSError as exc:
            logger.error("Failed to remove %s: %s", child, exc)


def is_supported_image(filename: str) -> bool:
    """
    Check whether *filename* has a supported image extension.

    The check is case-insensitive.

    Args:
        filename: The filename (or full path) to check.

    Returns:
        True if the extension is in SUPPORTED_IMAGE_EXTENSIONS.
    """
    return Path(filename).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def extract_images_from_zip(
    zip_path: Path,
    destination: Path,
) -> int:
    """
    Securely extract supported image files from a zip archive into
    *destination*.

    Security measures:
      - Absolute paths inside the zip are rejected (Zip Slip protection).
      - Path components like ``..`` that try to escape the destination are
        rejected.
      - Only files with supported image extensions are extracted; everything
        else (directories, metadata files like __MACOSX, etc.) is silently
        skipped.

    Args:
        zip_path: Path to the uploaded .zip file.
        destination: Target directory for extracted images.

    Returns:
        The number of image files successfully extracted.

    Raises:
        zipfile.BadZipFile: If the file is not a valid zip archive.
        ValueError: If a zip entry tries to escape the destination directory.
    """
    extracted_count = 0
    destination = destination.resolve()

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            # Skip directories
            if member.is_dir():
                continue

            # Get just the filename (ignore any directory structure in the zip)
            filename = Path(member.filename).name

            # Skip hidden / macOS metadata files
            if filename.startswith(".") or filename.startswith("__"):
                continue

            # Only extract supported image types
            if not is_supported_image(filename):
                logger.debug("Skipping unsupported file in zip: %s", member.filename)
                continue

            # Resolve the target path and verify it stays within destination
            target_path = (destination / filename).resolve()
            if not str(target_path).startswith(str(destination)):
                raise ValueError(
                    f"Zip Slip detected: {member.filename!r} would extract "
                    f"outside the target directory."
                )

            # Extract the file content and write to destination
            with zf.open(member) as source, open(target_path, "wb") as dest_file:
                shutil.copyfileobj(source, dest_file)

            extracted_count += 1
            logger.info("Extracted: %s", filename)

    return extracted_count
