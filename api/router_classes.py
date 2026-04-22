"""
Boxify Backend — Class Management Router

Endpoints:
    GET    /api/projects/{project_id}/classes                — List all classes
    POST   /api/projects/{project_id}/classes                — Create a class
    PUT    /api/projects/{project_id}/classes/{class_id}     — Rename / recolor
    DELETE /api/projects/{project_id}/classes/{class_id}     — Purge & re-index
"""

import logging
from sqlalchemy.exc import IntegrityError
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.deps import get_current_user
from api.schemas import (
    ClassCreate,
    ClassListResponse,
    ClassResponse,
    ClassUpdate,
)
from core.database import get_db
from core.models import Project, ProjectClass, User
from core.export_logic import (
    delete_class_and_reindex,
    rename_class_in_xmls,
    sync_classes_txt,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/projects", tags=["Classes"])


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_project_or_404(db: Session, project_id: int, user_id: int) -> Project:
    project = (
        db.query(Project)
        .filter(Project.id == project_id, Project.owner_id == user_id)
        .first()
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")
    return project


def _get_class_or_404(db: Session, project_id: int, class_id: int) -> ProjectClass:
    cls = (
        db.query(ProjectClass)
        .filter(ProjectClass.id == class_id, ProjectClass.project_id == project_id)
        .first()
    )
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found.")
    return cls


def _build_response(cls: ProjectClass, yolo_index: int) -> ClassResponse:
    return ClassResponse(
        id=cls.id,
        project_id=cls.project_id,
        name=cls.name,
        color=cls.color,
        yolo_index=yolo_index,
    )


def _fetch_ordered_classes(db: Session, project_id: int) -> list[ProjectClass]:
    """Return all classes for a project sorted by id ASC (= YOLO ordering)."""
    return (
        db.query(ProjectClass)
        .filter(ProjectClass.project_id == project_id)
        .order_by(ProjectClass.id.asc())
        .all()
    )


# ---------------------------------------------------------------------------
# GET /api/projects/{project_id}/classes
# ---------------------------------------------------------------------------

@router.get("/{project_id}/classes", response_model=ClassListResponse)
def list_classes(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return all classes for a project ordered by DB id (= YOLO index order)."""
    _get_project_or_404(db, project_id, current_user.id)
    classes = _fetch_ordered_classes(db, project_id)
    return ClassListResponse(
        classes=[_build_response(cls, idx) for idx, cls in enumerate(classes)]
    )


# ---------------------------------------------------------------------------
# POST /api/projects/{project_id}/classes
# ---------------------------------------------------------------------------

@router.post("/{project_id}/classes", response_model=ClassResponse, status_code=201)
def create_class(
    project_id: int,
    payload: ClassCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Create a new annotation class for the project.

    After inserting the DB row the ``classes.txt`` file is regenerated so the
    filesystem always stays in sync with the database.
    """
    _get_project_or_404(db, project_id, current_user.id)

    new_class = ProjectClass(
        project_id=project_id,
        name=payload.name,
        color=payload.color,
    )
    db.add(new_class)

    try:
        db.commit()
        db.refresh(new_class)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail=f"A class named '{payload.name}' already exists in this project.",
        )

    # Sync filesystem
    classes = _fetch_ordered_classes(db, project_id)
    sync_classes_txt(project_id, classes)

    # YOLO index = position in the freshly sorted list
    yolo_index = next(i for i, c in enumerate(classes) if c.id == new_class.id)

    logger.info(
        "[project %d] Created class '%s' (id=%d, yolo_index=%d, color=%s)",
        project_id, new_class.name, new_class.id, yolo_index, new_class.color,
    )
    return _build_response(new_class, yolo_index)


# ---------------------------------------------------------------------------
# PUT /api/projects/{project_id}/classes/{class_id}
# ---------------------------------------------------------------------------

@router.put("/{project_id}/classes/{class_id}", response_model=ClassResponse)
def update_class(
    project_id: int,
    class_id: int,
    payload: ClassUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Rename and / or recolor an existing class.

    If the name changes:
    1. All ``.xml`` annotation files are patched in-place (rename_class_in_xmls).
    2. ``classes.txt`` is regenerated (sync_classes_txt).

    If only the color changes no filesystem writes are needed beyond
    ``classes.txt`` (which doesn't store color, but we sync anyway for safety).
    """
    _get_project_or_404(db, project_id, current_user.id)
    cls = _get_class_or_404(db, project_id, class_id)

    old_name = cls.name
    name_changed = payload.name is not None and payload.name != old_name

    if payload.name is not None:
        cls.name = payload.name
    if payload.color is not None:
        cls.color = payload.color

    try:
        db.commit()
        db.refresh(cls)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail=f"A class named '{payload.name}' already exists in this project.",
        )

    # Filesystem updates
    if name_changed:
        rename_class_in_xmls(project_id, old_name, cls.name)

    classes = _fetch_ordered_classes(db, project_id)
    sync_classes_txt(project_id, classes)

    yolo_index = next(i for i, c in enumerate(classes) if c.id == cls.id)

    logger.info(
        "[project %d] Updated class id=%d: name='%s'->'%s', color='%s'",
        project_id, cls.id, old_name, cls.name, cls.color,
    )
    return _build_response(cls, yolo_index)


# ---------------------------------------------------------------------------
# DELETE /api/projects/{project_id}/classes/{class_id}
# ---------------------------------------------------------------------------

@router.delete("/{project_id}/classes/{class_id}", status_code=204)
def delete_class(
    project_id: int,
    class_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Delete a class using the Purge & Re-Index strategy:

    1. Determine the YOLO index of the class **before** it is removed from the DB.
    2. Remove the DB row.
    3. Call ``delete_class_and_reindex`` to:
       a. Purge matching ``<object>`` nodes from all ``.xml`` files.
       b. Drop lines with the deleted index and decrement higher indices in all
          ``.txt`` YOLO annotation files.
    4. Regenerate ``classes.txt`` from the updated (post-delete) class list.
    """
    _get_project_or_404(db, project_id, current_user.id)
    cls = _get_class_or_404(db, project_id, class_id)

    # Determine YOLO index BEFORE deletion so re-index logic is correct
    classes_before = _fetch_ordered_classes(db, project_id)
    deleted_yolo_index = next(
        (i for i, c in enumerate(classes_before) if c.id == cls.id), None
    )
    if deleted_yolo_index is None:
        raise HTTPException(status_code=500, detail="Could not determine class YOLO index.")

    class_name = cls.name

    # Remove from DB
    db.delete(cls)
    db.commit()

    # Purge annotations and re-index remaining classes in filesystem files
    stats = delete_class_and_reindex(project_id, class_name, deleted_yolo_index)

    # Regenerate classes.txt from the surviving classes
    classes_after = _fetch_ordered_classes(db, project_id)
    sync_classes_txt(project_id, classes_after)

    logger.info(
        "[project %d] Deleted class '%s' (yolo_index=%d) | "
        "xmls_modified=%d txts_modified=%d | remaining_classes=%d",
        project_id,
        class_name,
        deleted_yolo_index,
        stats["xmls_modified"],
        stats["txts_modified"],
        len(classes_after),
    )
    # 204 No Content — FastAPI sends no body automatically
