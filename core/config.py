"""
Boxify Backend — Configuration Settings

Centralizes all path constants and configuration values used across
the backend. All paths are resolved relative to the backend/ directory
to ensure consistent behavior regardless of the working directory.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Base Paths
# ---------------------------------------------------------------------------
# Resolve the project root as the parent of this file's directory (core/)
# i.e., backend/
BASE_DIR = Path(__file__).resolve().parent.parent

# Data storage root
DATA_DIR = BASE_DIR / "data"
PROJECTS_DIR = DATA_DIR / "projects"

# ---------------------------------------------------------------------------
# Project-Aware Path Helpers
# ---------------------------------------------------------------------------

def get_project_dir(project_id: int) -> Path:
    """Return the root directory for a given project."""
    return PROJECTS_DIR / f"project_{project_id}"

def get_images_dir(project_id: int) -> Path:
    return get_project_dir(project_id) / "images"

def get_output_dir(project_id: int) -> Path:
    return get_project_dir(project_id) / "output"

def get_inference_dir(project_id: int) -> Path:
    return get_project_dir(project_id) / "inference"

def get_classes_file(project_id: int) -> Path:
    return get_project_dir(project_id) / "classes.txt"

def ensure_project_dirs(project_id: int) -> None:
    """Create all required subdirectories for a project."""
    get_images_dir(project_id).mkdir(parents=True, exist_ok=True)
    get_output_dir(project_id).mkdir(parents=True, exist_ok=True)
    get_inference_dir(project_id).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Legacy default project (kept for backward compatibility / standalone mode)
# ---------------------------------------------------------------------------
DEFAULT_PROJECT_DIR = PROJECTS_DIR / "default_project"
IMAGES_DIR = DEFAULT_PROJECT_DIR / "images"
OUTPUT_DIR = DEFAULT_PROJECT_DIR / "output"
INFERENCE_DIR = DEFAULT_PROJECT_DIR / "inference"
CLASSES_FILE = DEFAULT_PROJECT_DIR / "classes.txt"

# ---------------------------------------------------------------------------
# Upload Constraints
# ---------------------------------------------------------------------------
# Maximum upload size in bytes (500 MB)
MAX_UPLOAD_SIZE_BYTES: int = 500 * 1024 * 1024  # 524_288_000

# Supported image file extensions (lowercase)
SUPPORTED_IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png"}

# ---------------------------------------------------------------------------
# CORS Settings (permissive for MVP)
# ---------------------------------------------------------------------------
CORS_ALLOW_ORIGINS: list[str] = ["*"]
CORS_ALLOW_METHODS: list[str] = ["*"]
CORS_ALLOW_HEADERS: list[str] = ["*"]

# ---------------------------------------------------------------------------
# Ensure required directories exist at import time
# ---------------------------------------------------------------------------
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
