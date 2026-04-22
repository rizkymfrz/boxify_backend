"""
Boxify Backend — FastAPI Application

The main entry point for the Boxify annotation backend.
Run with:  uvicorn api.main:app --reload --port 8000
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import ErrorResponse
from core.config import (
    CORS_ALLOW_HEADERS,
    CORS_ALLOW_METHODS,
    CORS_ALLOW_ORIGINS,
)

# Database & Routers
from core.database import engine
from core import models
from api.router_auth import router as auth_router
from api.router_projects import router as projects_router
from api.router_classes import router as classes_router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Boxify Annotation API",
    description=(
        "A lightweight, filesystem-based backend for the Boxify bounding "
        "box annotation tool. Handles dataset ingestion, image serving, "
        "YOLO annotation saving, and dataset export."
    ),
    version="1.0.0",
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)

# ---------------------------------------------------------------------------
# CORS Middleware (permissive for MVP)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
)

# ---------------------------------------------------------------------------
# Database — Auto-create tables on startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
def on_startup():
    models.Base.metadata.create_all(bind=engine)
    logger.info("Database tables ensured.")

# ---------------------------------------------------------------------------
# Register Routers (Auth & Projects)
# ---------------------------------------------------------------------------
app.include_router(auth_router)
app.include_router(projects_router)
app.include_router(classes_router)


