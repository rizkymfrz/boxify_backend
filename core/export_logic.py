"""
Boxify Backend — YOLO & XML Export Logic

Converts bounding box annotations from absolute pixel coordinates
(as sent by the React-Konva frontend) into two formats simultaneously:
1. Standard YOLO `.txt` format with normalized coordinates.
2. Custom XML format with a `<polygon>` representation of the bounding box.

IMPORTANT — Global Class Mapping:
    YOLO requires class IDs to be globally consistent across all images
    in a dataset. This module manages a single ``classes.txt`` file in the
    project directory.
"""

import logging
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path

from core.config import CLASSES_FILE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global classes.txt Manager
# ---------------------------------------------------------------------------


def load_label_map(classes_file: Path | None = None) -> dict[str, int]:
    if classes_file is None:
        from core.config import CLASSES_FILE
        classes_file = CLASSES_FILE

    if not classes_file.exists():
        return {}

    label_map: dict[str, int] = {}
    with open(classes_file, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            label = line.strip()
            if label:
                label_map[label] = index

    logger.debug("Loaded %d class(es) from %s", len(label_map), classes_file)
    return label_map


def save_label_map(label_map: dict[str, int], classes_file: Path | None = None) -> None:
    if classes_file is None:
        from core.config import CLASSES_FILE
        classes_file = CLASSES_FILE

    classes_file.parent.mkdir(parents=True, exist_ok=True)

    sorted_labels = sorted(label_map.items(), key=lambda item: item[1])

    with open(classes_file, "w", encoding="utf-8") as f:
        for label, _ in sorted_labels:
            f.write(f"{label}\n")

    logger.info("Saved %d class(es) to %s", len(label_map), classes_file)


def register_labels(labels: list[str], classes_file: Path | None = None) -> dict[str, int]:
    label_map = load_label_map(classes_file)
    updated = False

    for label in labels:
        if label not in label_map:
            new_index = len(label_map)
            label_map[label] = new_index
            logger.info(
                "Registered new class: %r → index %d", label, new_index
            )
            updated = True

    if updated:
        save_label_map(label_map, classes_file)

    return label_map


# ---------------------------------------------------------------------------
# Bounding Box Data
# ---------------------------------------------------------------------------


class BoundingBox:
    __slots__ = ("x", "y", "width", "height", "label", "type", "points")

    def __init__(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        label: str,
        type: str = "bbox",
        points: list[dict[str, float]] | None = None,
    ) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.label = label
        self.type = type
        self.points = points


# ---------------------------------------------------------------------------
# Format Conversions
# ---------------------------------------------------------------------------


def convert_to_yolo(
    bbox: BoundingBox,
    image_width: int,
    image_height: int,
    label_to_index: dict[str, int],
) -> str:
    """Convert to standard YOLO normalized coordinates."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Image dimensions must be positive.")

    class_index = label_to_index[bbox.label]
    
    if bbox.type == "polygon" and bbox.points:
        # YOLO segmentation format: <class> <x1> <y1> <x2> <y2> ...
        coords = []
        for p in bbox.points:
            # Handle both dict and object-like access for robustness
            px = p["x"] if isinstance(p, dict) else getattr(p, "x")
            py = p["y"] if isinstance(p, dict) else getattr(p, "y")
            coords.append(f"{max(0.0, min(1.0, px)):.6f}")
            coords.append(f"{max(0.0, min(1.0, py)):.6f}")
        return f"{class_index} {' '.join(coords)}"

    x_center = (bbox.x + bbox.width / 2.0) / image_width
    y_center = (bbox.y + bbox.height / 2.0) / image_height
    norm_width = bbox.width / image_width
    norm_height = bbox.height / image_height

    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    norm_width = max(0.0, min(1.0, norm_width))
    norm_height = max(0.0, min(1.0, norm_height))

    return f"{class_index} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"


def convert_to_xml(
    bboxes: list[BoundingBox], 
    image_filename: str,
    image_width: int,
    image_height: int,
) -> str:
    """Convert to Pascal VOC XML format with polygon support."""
    annotation = ET.Element("annotation")

    # Add some basic metadata
    folder = ET.SubElement(annotation, "folder")
    folder.text = "default_project"

    filename_elem = ET.SubElement(annotation, "filename")
    filename_elem.text = image_filename

    # Image size metadata
    size_elem = ET.SubElement(annotation, "size")
    ET.SubElement(size_elem, "width").text = str(image_width)
    ET.SubElement(size_elem, "height").text = str(image_height)
    ET.SubElement(size_elem, "depth").text = "3"

    for bbox in bboxes:
        obj = ET.SubElement(annotation, "object")

        name = ET.SubElement(obj, "name")
        name.text = bbox.label

        type_elem = ET.SubElement(obj, "type")
        type_elem.text = bbox.type

        if bbox.type == "polygon" and bbox.points:
            # True polygon: un-normalize each point back to absolute pixels
            # Format: <polygon><x1>..</x1><y1>..</y1>...</polygon>
            polygon_elem = ET.SubElement(obj, "polygon")
            for i, p in enumerate(bbox.points, 1):
                # Handle both dict and object-like access for robustness
                px = p["x"] if isinstance(p, dict) else getattr(p, "x")
                py = p["y"] if isinstance(p, dict) else getattr(p, "y")
                
                abs_x = int(round(px * image_width))
                abs_y = int(round(py * image_height))
                
                ET.SubElement(polygon_elem, f"x{i}").text = str(abs_x)
                ET.SubElement(polygon_elem, f"y{i}").text = str(abs_y)
        else:
            # Standard BBox: <bndbox><xmin>..</xmin>...
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(round(bbox.x)))
            ET.SubElement(bndbox, "ymin").text = str(int(round(bbox.y)))
            ET.SubElement(bndbox, "xmax").text = str(int(round(bbox.x + bbox.width)))
            ET.SubElement(bndbox, "ymax").text = str(int(round(bbox.y + bbox.height)))

    # Pretty print the XML
    xml_str = ET.tostring(annotation, encoding="utf-8")
    parsed_xml = minidom.parseString(xml_str)
    return parsed_xml.toprettyxml(indent="  ")


def save_annotations(
    bboxes: list[BoundingBox],
    image_width: int,
    image_height: int,
    image_filename: str,
    yolo_output_path: Path,
    xml_output_path: Path,
    classes_file: Path | None = None,
) -> int:
    """
    Saves BOTH the standard YOLO .txt and the custom XML files.
    Registers labels globally first.
    """
    # 1. Register labels
    unique_labels = list({bbox.label for bbox in bboxes})
    label_to_index = register_labels(unique_labels, classes_file)

    # 2. Generate and save YOLO (.txt)
    yolo_lines: list[str] = []
    for bbox in bboxes:
        try:
            line = convert_to_yolo(bbox, image_width, image_height, label_to_index)
            yolo_lines.append(line)
        except KeyError:
            logger.warning("Unknown label %r encountered", bbox.label)
        except ValueError as exc:
            logger.error("Invalid parameters: %s", exc)
            raise

    yolo_output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_yolo = yolo_output_path.with_suffix(".txt.tmp")
    with open(tmp_yolo, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines))
        if yolo_lines:
            f.write("\n")
    tmp_yolo.replace(yolo_output_path)

    # 3. Generate and save Custom XML (.xml)
    xml_content = convert_to_xml(
        bboxes, image_filename, image_width, image_height
    )
    xml_output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_xml = xml_output_path.with_suffix(".xml.tmp")
    with open(tmp_xml, "w", encoding="utf-8") as f:
        f.write(xml_content)
    tmp_xml.replace(xml_output_path)

    logger.info(
        "Saved %d annotation(s) to %s AND %s",
        len(bboxes),
        yolo_output_path.name,
        xml_output_path.name,
    )

    return len(bboxes)


def get_index_to_label_map(classes_file: Path | None = None) -> dict[int, str]:
    """Return a reverse mapping from class index to label string."""
    label_map = load_label_map(classes_file)
    return {v: k for k, v in label_map.items()}


# ---------------------------------------------------------------------------
# Project-Scoped Class Management Helpers
# ---------------------------------------------------------------------------


def sync_classes_txt(project_id: int, classes: list) -> None:
    """
    Recreate ``classes.txt`` for *project_id* from the canonical list of
    ``ProjectClass`` ORM objects sorted by their DB ``id`` (ascending).

    The 0-based array position of each class in this sorted list is the
    YOLO class index written into the ``.txt`` annotation files.

    Args:
        project_id: The target project.
        classes:    An ordered (by id ASC) iterable of ProjectClass ORM rows.
    """
    from core.config import get_project_dir

    classes_file = get_project_dir(project_id) / "classes.txt"
    classes_file.parent.mkdir(parents=True, exist_ok=True)

    with open(classes_file, "w", encoding="utf-8") as f:
        for cls in classes:
            f.write(f"{cls.name}\n")

    logger.info(
        "[project %d] Synced classes.txt → %d class(es): %s",
        project_id,
        len(classes),
        [c.name for c in classes],
    )


def rename_class_in_xmls(project_id: int, old_name: str, new_name: str) -> int:
    """
    Walk every ``.xml`` file in the project's ``output/`` directory and
    replace every ``<name>`` text node equal to *old_name* with *new_name*.

    Returns:
        Number of XML files that were modified.
    """
    from core.config import get_output_dir

    output_dir = get_output_dir(project_id)
    if not output_dir.exists():
        logger.info("[project %d] No output/ dir — rename_class_in_xmls is a no-op.", project_id)
        return 0

    modified = 0
    for xml_file in sorted(output_dir.glob("*.xml")):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            changed = False

            for obj in root.findall("object"):
                name_el = obj.find("name")
                if name_el is not None and name_el.text == old_name:
                    name_el.text = new_name
                    changed = True

            if changed:
                # Overwrite atomically via a temp-then-replace pattern
                tmp = xml_file.with_suffix(".xml.tmp")
                tree.write(tmp, encoding="utf-8", xml_declaration=True)
                tmp.replace(xml_file)
                modified += 1
                logger.debug("[project %d] Renamed '%s'→'%s' in %s", project_id, old_name, new_name, xml_file.name)

        except ET.ParseError:
            logger.warning("[project %d] Skipping malformed XML: %s", project_id, xml_file.name)

    logger.info(
        "[project %d] rename_class_in_xmls: '%s'→'%s' in %d file(s).",
        project_id, old_name, new_name, modified,
    )
    return modified


def delete_class_and_reindex(
    project_id: int,
    class_name_to_delete: str,
    deleted_class_index: int,
) -> dict[str, int]:
    """
    Purge & Re-Index pipeline — three-phase operation:

    Phase 1 — XML Purge
        Remove every ``<object>`` whose ``<name>`` matches *class_name_to_delete*
        from all ``.xml`` files in ``output/``. An empty ``<annotation>`` is kept
        so the image acts as a YOLO background-training sample.

    Phase 2 — YOLO TXT Re-Index
        For each ``.txt`` file in ``inference/``:
        * Lines whose class index == *deleted_class_index* are dropped.
        * Lines whose class index >  *deleted_class_index* are decremented by 1.
        Empty files are kept (YOLO background images).

    Args:
        project_id:            Target project.
        class_name_to_delete:  The human-readable class name to purge.
        deleted_class_index:   The 0-based YOLO index of the class being deleted.

    Returns:
        A dict with keys ``xmls_modified`` and ``txts_modified`` for logging.
    """
    from core.config import get_output_dir, get_inference_dir

    stats = {"xmls_modified": 0, "txts_modified": 0}

    # ------------------------------------------------------------------ #
    # Phase 1: XML Purge                                                   #
    # ------------------------------------------------------------------ #
    output_dir = get_output_dir(project_id)
    if output_dir.exists():
        for xml_file in sorted(output_dir.glob("*.xml")):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                # Collect objects to remove *before* mutating the tree
                to_remove = [
                    obj for obj in root.findall("object")
                    if (name_el := obj.find("name")) is not None
                    and name_el.text == class_name_to_delete
                ]

                if to_remove:
                    for obj in to_remove:
                        root.remove(obj)

                    tmp = xml_file.with_suffix(".xml.tmp")
                    tree.write(tmp, encoding="utf-8", xml_declaration=True)
                    tmp.replace(xml_file)
                    stats["xmls_modified"] += 1
                    logger.debug(
                        "[project %d] Purged %d object(s) from %s",
                        project_id, len(to_remove), xml_file.name,
                    )

            except ET.ParseError:
                logger.warning("[project %d] Skipping malformed XML: %s", project_id, xml_file.name)
    else:
        logger.info("[project %d] No output/ dir — XML purge is a no-op.", project_id)

    # ------------------------------------------------------------------ #
    # Phase 2: YOLO TXT Re-Index                                           #
    # ------------------------------------------------------------------ #
    inference_dir = get_inference_dir(project_id)
    if inference_dir.exists():
        for txt_file in sorted(inference_dir.glob("*.txt")):
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    original_lines = f.readlines()

                new_lines: list[str] = []
                changed = False

                for line in original_lines:
                    stripped = line.strip()
                    if not stripped:          # preserve blank lines (empty annotations)
                        new_lines.append(line)
                        continue

                    parts = stripped.split()
                    try:
                        class_idx = int(parts[0])
                    except (ValueError, IndexError):
                        # Malformed line — keep as-is
                        new_lines.append(line)
                        continue

                    if class_idx == deleted_class_index:
                        # Drop this annotation entirely
                        changed = True
                        continue
                    elif class_idx > deleted_class_index:
                        # Decrement the index by 1
                        parts[0] = str(class_idx - 1)
                        new_lines.append(" ".join(parts) + "\n")
                        changed = True
                    else:
                        new_lines.append(line)

                if changed:
                    with open(txt_file, "w", encoding="utf-8") as f:
                        f.writelines(new_lines)
                    stats["txts_modified"] += 1
                    logger.debug(
                        "[project %d] Re-indexed %s (deleted idx=%d)",
                        project_id, txt_file.name, deleted_class_index,
                    )

            except OSError:
                logger.warning("[project %d] Could not process %s", project_id, txt_file.name)
    else:
        logger.info("[project %d] No inference/ dir — TXT re-index is a no-op.", project_id)

    logger.info(
        "[project %d] Purge complete — deleted='%s' (idx=%d) | "
        "xmls_modified=%d txts_modified=%d",
        project_id,
        class_name_to_delete,
        deleted_class_index,
        stats["xmls_modified"],
        stats["txts_modified"],
    )
    return stats


def load_yolo_annotations(
    yolo_file_path: Path,
    image_width: int,
    image_height: int,
    index_to_label: dict[int, str],
) -> list[BoundingBox]:
    """
    Read a YOLO .txt file and convert normalized coordinates back to
    absolute pixel coordinates.
    """
    bboxes: list[BoundingBox] = []
    if not yolo_file_path.exists():
        return bboxes

    with open(yolo_file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
                
            class_index = int(parts[0])
            label = index_to_label.get(class_index, f"class_{class_index}")
            
            # Case 1: Standard YOLO BBox (5 values)
            if len(parts) == 5:
                x_center = float(parts[1])
                y_center = float(parts[2])
                norm_width = float(parts[3])
                norm_height = float(parts[4])

                abs_width = norm_width * image_width
                abs_height = norm_height * image_height
                abs_x = (x_center * image_width) - (abs_width / 2.0)
                abs_y = (y_center * image_height) - (abs_height / 2.0)

                bboxes.append(BoundingBox(
                    x=abs_x,
                    y=abs_y,
                    width=abs_width,
                    height=abs_height,
                    label=label,
                    type="bbox"
                ))
            
            # Case 2: YOLO Segmentation / Polygon (> 5 values, must be even # of coords)
            elif len(parts) > 5 and (len(parts) - 1) % 2 == 0:
                normalized_points = []
                for i in range(1, len(parts), 2):
                    normalized_points.append({
                        "x": float(parts[i]),
                        "y": float(parts[i+1])
                    })
                
                # Calculate bounding box from points for UI compatibility
                if normalized_points:
                    xs = [p["x"] * image_width for p in normalized_points]
                    ys = [p["y"] * image_height for p in normalized_points]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    
                    bboxes.append(BoundingBox(
                        x=min_x,
                        y=min_y,
                        width=max_x - min_x,
                        height=max_y - min_y,
                        label=label,
                        type="polygon",
                        points=normalized_points
                    ))
    return bboxes
