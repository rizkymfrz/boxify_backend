"""
Boxify Backend — AI Inference Service

Handles YOLO model loading and auto-labeling of images.
Integrates with the existing annotation pipeline (save_annotations)
so that both .txt (YOLO) and .xml (VOC) files are updated atomically.
"""

import logging
from pathlib import Path

from PIL import Image
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from ultralytics import YOLO

from core.config import (
    get_images_dir,
    get_inference_dir,
    get_models_dir,
    get_output_dir,
    get_project_dir,
)
from core.export_logic import (
    BoundingBox,
    get_index_to_label_map,
    load_yolo_annotations,
    save_annotations,
    sync_classes_txt,
)
from core.models import ProjectClass

logger = logging.getLogger(__name__)

# Default colours assigned to AI-discovered classes in rotation.
# Matches the colour palette used by the frontend class manager.
_DEFAULT_COLORS = [
    "#ef4444",  # red
    "#f97316",  # orange
    "#eab308",  # yellow
    "#22c55e",  # green
    "#3b82f6",  # blue
    "#a855f7",  # purple
    "#ec4899",  # pink
]


def run_auto_labeling(
    project_id: int,
    filename: str,
    model_name: str,
    db: Session,
) -> dict:
    """
    Run YOLO inference on a single image and merge the results with any
    existing manual annotations.

    Args:
        project_id:  Target project ID.
        filename:    Image filename (e.g. ``Image_1.jpg``).
        model_name:  Name of the ``.pt`` model file stored in the project's
                     ``models/`` directory.
        db:          Active SQLAlchemy session.

    Returns:
        A dict with keys ``boxes_added`` (int) and ``classes_created``
        (list[str]).
    """

    # ------------------------------------------------------------------
    # 1. Resolve paths & validate
    # ------------------------------------------------------------------
    model_path = get_models_dir(project_id) / model_name
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_name}")

    images_dir = get_images_dir(project_id)
    image_path = images_dir / filename
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {filename}")

    # ------------------------------------------------------------------
    # 2. Load model & run inference
    # ------------------------------------------------------------------
    logger.info(
        "[project %d] Loading model '%s' for image '%s'",
        project_id, model_name, filename,
    )
    model = YOLO(str(model_path))
    results = model(str(image_path), verbose=False)

    # Read actual image dimensions
    with Image.open(image_path) as img:
        img_width, img_height = img.size

    # ------------------------------------------------------------------
    # 3. Extract predictions & ensure project classes exist
    # ------------------------------------------------------------------
    # Build a lookup of existing project classes
    existing_classes: list[ProjectClass] = (
        db.query(ProjectClass)
        .filter(ProjectClass.project_id == project_id)
        .order_by(ProjectClass.id)
        .all()
    )
    
    # Map lowercase class names to the exact casing stored in DB to avoid MySQL case-insensitivity issues
    class_name_map = {c.name.lower(): c.name for c in existing_classes}

    new_classes_created: list[str] = []
    new_bboxes: list[BoundingBox] = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for i in range(len(boxes)):
            # xyxy format: [x1, y1, x2, y2]
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            cls_id = int(boxes.cls[i].item())
            raw_class_name = result.names.get(cls_id, f"class_{cls_id}")
            lower_class_name = raw_class_name.lower()

            # Auto-create class in DB if missing (case-insensitive check)
            if lower_class_name in class_name_map:
                final_class_name = class_name_map[lower_class_name]
            else:
                final_class_name = raw_class_name
                color = _DEFAULT_COLORS[len(existing_classes) % len(_DEFAULT_COLORS)]
                
                try:
                    new_class = ProjectClass(
                        project_id=project_id,
                        name=final_class_name,
                        color=color,
                    )
                    db.add(new_class)
                    db.flush()  # get the ID assigned
                    
                    existing_classes.append(new_class)
                    class_name_map[lower_class_name] = final_class_name
                    new_classes_created.append(final_class_name)
                    logger.info(
                        "[project %d] Auto-created class '%s' (color=%s)",
                        project_id, final_class_name, color,
                    )
                except IntegrityError:
                    db.rollback()
                    # It might have been created by a concurrent request, fetch it
                    existing_class = db.query(ProjectClass).filter(
                        ProjectClass.project_id == project_id,
                        ProjectClass.name == final_class_name
                    ).first()
                    
                    if existing_class:
                        final_class_name = existing_class.name
                        class_name_map[lower_class_name] = final_class_name
                    else:
                        class_name_map[lower_class_name] = final_class_name

            # Convert xyxy → absolute x, y, width, height
            abs_x = x1
            abs_y = y1
            abs_w = x2 - x1
            abs_h = y2 - y1

            new_bboxes.append(BoundingBox(
                x=abs_x, y=abs_y, width=abs_w, height=abs_h, label=final_class_name,
                type="bbox"
            ))

    # ------------------------------------------------------------------
    # 4. Sync classes.txt after potential new class additions
    # ------------------------------------------------------------------
    if new_classes_created:
        db.commit()
        # Re-fetch ordered list after commit
        all_classes = (
            db.query(ProjectClass)
            .filter(ProjectClass.project_id == project_id)
            .order_by(ProjectClass.id)
            .all()
        )
        sync_classes_txt(project_id, all_classes)

    # ------------------------------------------------------------------
    # 5. Merge with existing annotations & save
    # ------------------------------------------------------------------
    base_name = Path(filename).stem
    inference_dir = get_inference_dir(project_id)
    output_dir = get_output_dir(project_id)
    classes_file = get_project_dir(project_id) / "classes.txt"

    yolo_path = inference_dir / f"{base_name}.txt"
    xml_path = output_dir / f"{base_name}.xml"

    # Load any existing annotations
    index_to_label = get_index_to_label_map(classes_file)
    existing_bboxes = load_yolo_annotations(yolo_path, img_width, img_height, index_to_label)

    # Combine: existing manual + new AI predictions
    merged = existing_bboxes + new_bboxes

    # Save both .txt and .xml atomically
    save_annotations(
        bboxes=merged,
        image_width=img_width,
        image_height=img_height,
        image_filename=filename,
        yolo_output_path=yolo_path,
        xml_output_path=xml_path,
        classes_file=classes_file,
    )

    logger.info(
        "[project %d] Auto-labeling done for '%s' — %d new box(es), %d class(es) created",
        project_id, filename, len(new_bboxes), len(new_classes_created),
    )

    return {
        "boxes_added": len(new_bboxes),
        "classes_created": new_classes_created,
    }
