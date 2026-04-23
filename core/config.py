"""
Boxify Backend — Configuration Settings

Centralizes ALL configuration values used across the backend.
Values are loaded from a `.env` file (via python-dotenv) with sensible
defaults so the app works out-of-the-box for local development.

Usage in other modules:
    from core.config import DATABASE_URL, JWT_SECRET_KEY, ...
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env file (auto-discover from backend/ root)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
_env_path = BASE_DIR / ".env"
load_dotenv(_env_path)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DATABASE_URL: str = os.getenv(
    "MYSQL_URL",
    "mysql+pymysql://samtek_user:samtek123@localhost:3306/boxify",
)

# ---------------------------------------------------------------------------
# JWT Authentication
# ---------------------------------------------------------------------------
JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "boxify-dev-secret-change-in-production")
JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES: int = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))  # 24 h

# ---------------------------------------------------------------------------
# File Storage Paths
# ---------------------------------------------------------------------------
_projects_dir_raw = os.getenv("PROJECTS_DIR", "")
if _projects_dir_raw:
    # Support both absolute and relative paths
    _p = Path(_projects_dir_raw)
    PROJECTS_DIR = _p if _p.is_absolute() else (BASE_DIR / _p).resolve()
else:
    PROJECTS_DIR = BASE_DIR / "data" / "projects"

DATA_DIR = PROJECTS_DIR.parent  # kept for backward-compat


def get_project_dir(project_id: int) -> Path:
    """Return the root directory for a given project."""
    return PROJECTS_DIR / f"project_{project_id}"

def get_images_dir(project_id: int) -> Path:
    return get_project_dir(project_id) / "images"

def get_output_dir(project_id: int) -> Path:
    return get_project_dir(project_id) / "output"

def get_inference_dir(project_id: int) -> Path:
    return get_project_dir(project_id) / "inference"

def get_models_dir(project_id: int) -> Path:
    return get_project_dir(project_id) / "models"

def get_classes_file(project_id: int) -> Path:
    return get_project_dir(project_id) / "classes.txt"

def ensure_project_dirs(project_id: int) -> None:
    """Create all required subdirectories for a project."""
    get_images_dir(project_id).mkdir(parents=True, exist_ok=True)
    get_output_dir(project_id).mkdir(parents=True, exist_ok=True)
    get_inference_dir(project_id).mkdir(parents=True, exist_ok=True)
    get_models_dir(project_id).mkdir(parents=True, exist_ok=True)

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
MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))
MAX_UPLOAD_SIZE_BYTES: int = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# Supported image file extensions (lowercase)
SUPPORTED_IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png"}

# ---------------------------------------------------------------------------
# CORS Settings
# ---------------------------------------------------------------------------
def _parse_csv(value: str) -> list[str]:
    """Parse a comma-separated env var into a list."""
    return [item.strip() for item in value.split(",") if item.strip()]

CORS_ALLOW_ORIGINS: list[str] = _parse_csv(os.getenv("CORS_ALLOW_ORIGINS", "*"))
CORS_ALLOW_METHODS: list[str] = _parse_csv(os.getenv("CORS_ALLOW_METHODS", "*"))
CORS_ALLOW_HEADERS: list[str] = _parse_csv(os.getenv("CORS_ALLOW_HEADERS", "*"))

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT: int = int(os.getenv("APP_PORT", "8000"))

# ---------------------------------------------------------------------------
# Ensure required directories exist at import time
# ---------------------------------------------------------------------------
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
