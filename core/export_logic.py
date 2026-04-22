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
    __slots__ = ("x", "y", "width", "height", "label")

    def __init__(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        label: str,
    ) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.label = label


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

    x_center = (bbox.x + bbox.width / 2.0) / image_width
    y_center = (bbox.y + bbox.height / 2.0) / image_height
    norm_width = bbox.width / image_width
    norm_height = bbox.height / image_height

    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    norm_width = max(0.0, min(1.0, norm_width))
    norm_height = max(0.0, min(1.0, norm_height))

    return f"{class_index} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"


def convert_to_xml_polygon(bboxes: list[BoundingBox], image_filename: str) -> str:
    """Convert to custom XML format with a 4-point polygon representation."""
    annotation = ET.Element("annotation")

    # Add some basic metadata
    folder = ET.SubElement(annotation, "folder")
    folder.text = "default_project"

    filename_elem = ET.SubElement(annotation, "filename")
    filename_elem.text = image_filename

    for bbox in bboxes:
        obj = ET.SubElement(annotation, "object")

        name = ET.SubElement(obj, "name")
        name.text = bbox.label

        type_elem = ET.SubElement(obj, "type")
        type_elem.text = "polygon"

        polygon = ET.SubElement(obj, "polygon")

        # Top-Left (x, y)
        p1 = ET.SubElement(polygon, "point")
        ET.SubElement(p1, "x").text = str(int(round(bbox.x)))
        ET.SubElement(p1, "y").text = str(int(round(bbox.y)))

        # Top-Right (x + width, y)
        p2 = ET.SubElement(polygon, "point")
        ET.SubElement(p2, "x").text = str(int(round(bbox.x + bbox.width)))
        ET.SubElement(p2, "y").text = str(int(round(bbox.y)))

        # Bottom-Right (x + width, y + height)
        p3 = ET.SubElement(polygon, "point")
        ET.SubElement(p3, "x").text = str(int(round(bbox.x + bbox.width)))
        ET.SubElement(p3, "y").text = str(int(round(bbox.y + bbox.height)))

        # Bottom-Left (x, y + height)
        p4 = ET.SubElement(polygon, "point")
        ET.SubElement(p4, "x").text = str(int(round(bbox.x)))
        ET.SubElement(p4, "y").text = str(int(round(bbox.y + bbox.height)))

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
    with open(yolo_output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines))
        if yolo_lines:
            f.write("\n")

    # 3. Generate and save Custom XML (.xml)
    xml_content = convert_to_xml_polygon(bboxes, image_filename)
    xml_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(xml_output_path, "w", encoding="utf-8") as f:
        f.write(xml_content)

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
            if len(parts) == 5:
                class_index = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                norm_width = float(parts[3])
                norm_height = float(parts[4])

                abs_width = norm_width * image_width
                abs_height = norm_height * image_height
                abs_x = (x_center * image_width) - (abs_width / 2.0)
                abs_y = (y_center * image_height) - (abs_height / 2.0)

                label = index_to_label.get(class_index, f"class_{class_index}")
                bboxes.append(BoundingBox(
                    x=abs_x,
                    y=abs_y,
                    width=abs_width,
                    height=abs_height,
                    label=label
                ))
    return bboxes
