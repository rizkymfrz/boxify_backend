"""
Boxify Backend — Projects Router

Endpoints:
    POST /api/projects          — Create project with ZIP upload
    GET  /api/projects          — List projects for the authenticated user
    POST /api/projects/{id}/models — Upload a YOLO .pt model
    POST /api/projects/{id}/images/{filename}/auto-label — AI auto-labeling
"""

import logging
import tempfile
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from PIL import Image

from api.deps import get_current_user
from api.schemas import (
    AnnotationRequest,
    AnnotationResponse,
    AutoLabelRequest,
    AutoLabelResponse,
    ImageItem,
    ImageListResponse,
    ProjectCreateResponse,
    ProjectListItem,
    ProjectListResponse,
)
from core.config import (
    MAX_UPLOAD_SIZE_BYTES,
    SUPPORTED_IMAGE_EXTENSIONS,
    ensure_project_dirs,
    get_images_dir,
    get_inference_dir,
    get_models_dir,
    get_output_dir,
    get_project_dir,
)
from core.database import get_db
from core.models import ImageRecord, ModelRecord, Project, User
from core.file_utils import extract_images_from_zip
from core.export_logic import BoundingBox, save_annotations, load_yolo_annotations, get_index_to_label_map

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/projects", tags=["Projects"])


@router.post("", response_model=ProjectCreateResponse, status_code=201)
async def create_project(
    name: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new project by uploading a ZIP of images."""
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are accepted.")

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_zip_path = tmp_dir / "upload.zip"

    try:
        # Stream upload to temp file with size check
        total_bytes = 0
        with open(tmp_zip_path, "wb") as tmp_file:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(status_code=413, detail="File exceeds 500 MB limit.")
                tmp_file.write(chunk)

        if not zipfile.is_zipfile(tmp_zip_path):
            raise HTTPException(status_code=400, detail="Invalid zip archive.")

        # Create DB record first to get the project ID
        project = Project(name=name, owner_id=current_user.id)
        db.add(project)
        db.commit()
        db.refresh(project)

        # Create project directories
        ensure_project_dirs(project.id)
        images_dir = get_images_dir(project.id)

        # Extract images
        image_count = extract_images_from_zip(tmp_zip_path, images_dir)

        if image_count == 0:
            db.delete(project)
            db.commit()
            raise HTTPException(
                status_code=400,
                detail=f"No supported images found. Supported: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}",
            )

        # Create ImageRecord rows for each extracted image
        for img_file in sorted(images_dir.iterdir()):
            if img_file.is_file() and img_file.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                db.add(ImageRecord(project_id=project.id, filename=img_file.name, status="pending"))
        db.commit()

        logger.info("Created project '%s' (id=%d) with %d images", name, project.id, image_count)

        return ProjectCreateResponse(
            id=project.id,
            name=project.name,
            image_count=image_count,
            created_at=project.created_at,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error creating project")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.get("", response_model=ProjectListResponse)
def list_projects(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all projects belonging to the authenticated user."""
    projects = db.query(Project).filter(Project.owner_id == current_user.id).order_by(Project.created_at.desc()).all()

    items = []
    for p in projects:
        total = db.query(ImageRecord).filter(ImageRecord.project_id == p.id).count()
        done = db.query(ImageRecord).filter(ImageRecord.project_id == p.id, ImageRecord.status == "done").count()
        items.append(ProjectListItem(
            id=p.id,
            name=p.name,
            image_count=total,
            annotated_count=done,
            created_at=p.created_at,
        ))

    return ProjectListResponse(projects=items)


def get_project_or_404(db: Session, project_id: int, user_id: int) -> Project:
    project = db.query(Project).filter(Project.id == project_id, Project.owner_id == user_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")
    return project


@router.get("/{project_id}/images", response_model=ImageListResponse)
def list_project_images(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    get_project_or_404(db, project_id, current_user.id)
    images_dir = get_images_dir(project_id)
    inference_dir = get_inference_dir(project_id)

    images_data = []
    if images_dir.exists():
        for f in sorted(images_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                base_name = f.stem
                txt_path = inference_dir / f"{base_name}.txt"
                annotation_count = 0
                if txt_path.exists():
                    with open(txt_path, "r", encoding="utf-8") as txt_f:
                        annotation_count = sum(1 for line in txt_f if line.strip())
                images_data.append(ImageItem(filename=f.name, annotation_count=annotation_count))

    return ImageListResponse(images=images_data)


@router.get("/{project_id}/images/{filename}")
def get_project_image(
    project_id: int,
    filename: str,
    db: Session = Depends(get_db),
):
    # Publicly accessible for <img> tags without needing complex JWT passing
    images_dir = get_images_dir(project_id)
    image_path = images_dir / filename

    resolved = image_path.resolve()
    if not str(resolved).startswith(str(images_dir.resolve())):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")

    suffix = resolved.suffix.lower()
    media_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
    return FileResponse(resolved, media_type=media_types.get(suffix, "application/octet-stream"))


@router.get("/{project_id}/annotations/{filename}", response_model=AnnotationRequest)
def get_project_annotation(
    project_id: int,
    filename: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    get_project_or_404(db, project_id, current_user.id)
    
    images_dir = get_images_dir(project_id)
    inference_dir = get_inference_dir(project_id)
    
    image_path = images_dir / filename
    resolved_image = image_path.resolve()
    
    if not resolved_image.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")

    try:
        with Image.open(resolved_image) as img:
            img_width, img_height = img.size
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read image dimensions: {exc}")

    base_name = Path(filename).stem
    yolo_path = inference_dir / f"{base_name}.txt"
    
    classes_file = get_project_dir(project_id) / "classes.txt"
    index_to_label = get_index_to_label_map(classes_file)

    bboxes = load_yolo_annotations(yolo_path, img_width, img_height, index_to_label)
    
    boxes_out = [
        {
            "x": bbox.x,
            "y": bbox.y,
            "width": bbox.width,
            "height": bbox.height,
            "label": bbox.label,
            "type": bbox.type,
            "points": bbox.points
        }
        for bbox in bboxes
    ]

    return AnnotationRequest(
        image_width=img_width,
        image_height=img_height,
        filename=filename,
        boxes=boxes_out,
    )


@router.post("/{project_id}/annotations/{filename}", response_model=AnnotationResponse)
def save_project_annotation(
    project_id: int,
    filename: str,
    payload: AnnotationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    get_project_or_404(db, project_id, current_user.id)
    
    images_dir = get_images_dir(project_id)
    inference_dir = get_inference_dir(project_id)
    output_dir = get_output_dir(project_id)
    
    image_path = images_dir / filename
    resolved_image = image_path.resolve()
    
    if not resolved_image.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")

    base_name = Path(filename).stem
    yolo_path = inference_dir / f"{base_name}.txt"
    xml_path = output_dir / f"{base_name}.xml"
    classes_file = get_project_dir(project_id) / "classes.txt"

    bboxes = [
        BoundingBox(
            x=box.x,
            y=box.y,
            width=box.width,
            height=box.height,
            label=box.label,
            type=box.type,
            points=[{"x": p.x, "y": p.y} for p in box.points] if box.points else None
        )
        for box in payload.boxes
    ]

    try:
        box_count = save_annotations(
            bboxes=bboxes,
            image_width=payload.image_width,
            image_height=payload.image_height,
            image_filename=filename,
            yolo_output_path=yolo_path,
            xml_output_path=xml_path,
            classes_file=classes_file,
        )
        
        # Update ImageRecord status to done
        record = db.query(ImageRecord).filter(ImageRecord.project_id == project_id, ImageRecord.filename == filename).first()
        if record:
            record.status = "done"
            db.commit()

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return AnnotationResponse(
        message=f"Annotations saved for {filename}",
        label_file=f"{base_name}.txt",
        box_count=box_count,
    )


@router.get("/{project_id}/export")
def export_project_dataset(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    project = get_project_or_404(db, project_id, current_user.id)
    
    images_dir = get_images_dir(project_id)
    if not images_dir.exists() or not list(images_dir.iterdir()):
        raise HTTPException(status_code=404, detail="No images found in the project.")

    tmp_dir = Path(tempfile.mkdtemp())
    zip_path = tmp_dir / f"project_{project_id}.zip"

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            if images_dir.exists():
                for f in sorted(images_dir.iterdir()):
                    if f.is_file(): zf.write(f, f"images/{f.name}")
                    
            output_dir = get_output_dir(project_id)
            if output_dir.exists():
                for f in sorted(output_dir.iterdir()):
                    if f.is_file(): zf.write(f, f"output/{f.name}")
                    
            inference_dir = get_inference_dir(project_id)
            if inference_dir.exists():
                for f in sorted(inference_dir.iterdir()):
                    if f.is_file(): zf.write(f, f"inference/{f.name}")
                    
            classes_file = get_project_dir(project_id) / "classes.txt"
            if classes_file.exists():
                zf.write(classes_file, "classes.txt")

        return FileResponse(
            path=zip_path,
            filename=f"{project.name.replace(' ', '_')}.zip",
            media_type="application/zip",
        )
    except Exception as exc:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Export failed: {exc}")


@router.delete("/{project_id}/images/{filename}")
def delete_project_image(
    project_id: int,
    filename: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    get_project_or_404(db, project_id, current_user.id)
    
    images_dir = get_images_dir(project_id)
    inference_dir = get_inference_dir(project_id)
    output_dir = get_output_dir(project_id)
    
    image_path = images_dir / filename
    resolved_image = image_path.resolve()
    
    if not resolved_image.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
        
    resolved_image.unlink()
    
    base_name = Path(filename).stem
    
    yolo_path = inference_dir / f"{base_name}.txt"
    if yolo_path.exists():
        yolo_path.unlink()
        
    xml_path = output_dir / f"{base_name}.xml"
    if xml_path.exists():
        xml_path.unlink()
        
    record = db.query(ImageRecord).filter(ImageRecord.project_id == project_id, ImageRecord.filename == filename).first()
    if record:
        db.delete(record)
        db.commit()
        
    return {"message": f"Image {filename} deleted successfully."}


# ---------------------------------------------------------------------------
# Feature F: Model Upload & AI Auto-Labeling
# ---------------------------------------------------------------------------


@router.post("/{project_id}/models", status_code=201)
async def upload_model(
    project_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Upload a YOLO .pt model file to a project."""
    get_project_or_404(db, project_id, current_user.id)

    if not file.filename or not file.filename.lower().endswith(".pt"):
        raise HTTPException(status_code=400, detail="Only .pt model files are accepted.")

    models_dir = get_models_dir(project_id)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / file.filename

    try:
        with open(model_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)
    except Exception as exc:
        logger.exception("Error saving model file")
        raise HTTPException(status_code=500, detail=f"Failed to save model: {exc}") from exc

    logger.info(
        "[project %d] Uploaded model '%s' (%d bytes)",
        project_id, file.filename, model_path.stat().st_size,
    )

    # Persist metadata to the database
    record = ModelRecord(
        project_id=project_id,
        name=file.filename,
        file_path=str(model_path),
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return {
        "message": f"Model '{file.filename}' uploaded successfully.",
        "model_name": file.filename,
        "id": record.id,
        "uploaded_at": record.uploaded_at.isoformat(),
    }


@router.get("/{project_id}/models")
def list_project_models(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return all model records for a given project, ordered newest first."""
    get_project_or_404(db, project_id, current_user.id)

    records = (
        db.query(ModelRecord)
        .filter(ModelRecord.project_id == project_id)
        .order_by(ModelRecord.uploaded_at.desc())
        .all()
    )

    return {
        "models": [
            {
                "id": r.id,
                "name": r.name,
                "file_path": r.file_path,
                "uploaded_at": r.uploaded_at.isoformat(),
            }
            for r in records
        ]
    }


@router.post(
    "/{project_id}/images/{filename}/auto-label",
    response_model=AutoLabelResponse,
)
def auto_label_image(
    project_id: int,
    filename: str,
    payload: AutoLabelRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Run YOLO inference on a single image and merge results with existing annotations."""
    get_project_or_404(db, project_id, current_user.id)

    from core.inference_service import run_auto_labeling

    try:
        result = run_auto_labeling(
            project_id=project_id,
            filename=filename,
            model_name=payload.model_name,
            db=db,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Auto-labeling failed for '%s'", filename)
        raise HTTPException(status_code=500, detail=f"Auto-labeling failed: {exc}") from exc

    # Mark image as annotated
    record = db.query(ImageRecord).filter(
        ImageRecord.project_id == project_id,
        ImageRecord.filename == filename,
    ).first()
    if record and result["boxes_added"] > 0:
        record.status = "done"
        db.commit()

    return AutoLabelResponse(
        message=f"Auto-labeling completed for {filename}.",
        boxes_added=result["boxes_added"],
        classes_created=result["classes_created"],
    )
