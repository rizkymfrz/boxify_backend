"""
Boxify Backend — Pydantic Schemas

Defines request and response models for the FastAPI endpoints.
All validation, type coercion, and documentation are handled here
using Pydantic v2.
"""

from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# Auth Schemas
# ---------------------------------------------------------------------------

class AuthRegisterRequest(BaseModel):
    """Payload for user registration."""
    username: str = Field(..., min_length=3, max_length=50, examples=["admin"])
    password: str = Field(..., min_length=6, examples=["secret123"])


class AuthLoginRequest(BaseModel):
    """Payload for user login."""
    username: str = Field(..., examples=["admin"])
    password: str = Field(..., examples=["secret123"])


class AuthResponse(BaseModel):
    """Response after successful register or login."""
    access_token: str = Field(..., description="JWT access token.")
    token_type: str = Field(default="bearer")
    user_id: int = Field(..., description="The authenticated user's ID.")
    username: str = Field(..., description="The authenticated user's username.")


# ---------------------------------------------------------------------------
# Project Schemas
# ---------------------------------------------------------------------------

class ProjectCreateResponse(BaseModel):
    """Response after creating a new project via ZIP upload."""
    id: int
    name: str
    image_count: int
    created_at: datetime


class ProjectListItem(BaseModel):
    """A single project in the list response."""
    id: int
    name: str
    image_count: int = Field(default=0, description="Total images in this project.")
    annotated_count: int = Field(default=0, description="Images with status 'done'.")
    created_at: datetime


class ProjectListResponse(BaseModel):
    """Response for listing all user projects."""
    projects: list[ProjectListItem]


# ---------------------------------------------------------------------------
# Class Management Schemas
# ---------------------------------------------------------------------------

class ClassCreate(BaseModel):
    """Payload for creating a new project class."""
    name: str = Field(..., min_length=1, max_length=100, examples=["person"])
    color: str = Field(
        default="#ef4444",
        pattern=r"^#[0-9a-fA-F]{6}$",
        description="Hex colour string, e.g. '#ef4444'. Defaults to Tailwind red.",
        examples=["#ef4444"],
    )


class ClassUpdate(BaseModel):
    """Payload for renaming or recolouring a class. All fields are optional."""
    name: str | None = Field(default=None, min_length=1, max_length=100, examples=["vehicle"])
    color: str | None = Field(
        default=None,
        pattern=r"^#[0-9a-fA-F]{6}$",
        examples=["#3b82f6"],
    )


class ClassResponse(BaseModel):
    """A single class in the API response."""
    id: int
    project_id: int
    name: str
    color: str
    yolo_index: int = Field(..., description="0-based YOLO class index (position in id-sorted list).")

    class Config:
        from_attributes = True


class ClassListResponse(BaseModel):
    """Response for listing all classes within a project."""
    classes: list[ClassResponse]


# ---------------------------------------------------------------------------
# Feature A: Dataset Upload (Legacy — kept for standalone mode)
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    """Response returned after a successful dataset upload."""

    message: str = Field(
        ...,
        description="Human-readable success message.",
        examples=["Dataset uploaded successfully."],
    )
    image_count: int = Field(
        ...,
        ge=0,
        description="Total number of images extracted from the zip archive.",
        examples=[42],
    )


# ---------------------------------------------------------------------------
# Feature B: Image Listing
# ---------------------------------------------------------------------------

class ImageItem(BaseModel):
    """Detailed image info including annotation count."""

    filename: str = Field(
        ...,
        description="Image filename.",
        examples=["img_001.jpg"],
    )
    annotation_count: int = Field(
        default=0,
        ge=0,
        description="Number of bounding box annotations for this image (lines in the YOLO .txt file).",
        examples=[3],
    )

class ImageListResponse(BaseModel):
    """Response containing all image filenames in the project."""

    images: list[ImageItem] = Field(
        ...,
        description="List of images available in the project.",
    )


# ---------------------------------------------------------------------------
# Feature C: Save Annotation
# ---------------------------------------------------------------------------

class Point(BaseModel):
    """A single normalized [0-1] coordinate point."""
    x: float
    y: float

class BoundingBoxSchema(BaseModel):
    """
    A single bounding box or polygon annotation.
    BBoxes are in absolute pixels; Polygons are in normalized [0-1] coordinates.
    """

    x: float = Field(
        ...,
        description="Left edge of the bounding box in pixels.",
        examples=[120.5],
    )
    y: float = Field(
        ...,
        description="Top edge of the bounding box in pixels.",
        examples=[80.0],
    )
    width: float = Field(
        ...,
        gt=0,
        description="Width of the bounding box in pixels. Must be > 0.",
        examples=[200.0],
    )
    height: float = Field(
        ...,
        gt=0,
        description="Height of the bounding box in pixels. Must be > 0.",
        examples=[150.0],
    )
    label: str = Field(
        ...,
        min_length=1,
        description="Class label for this annotation (e.g., 'person', 'car').",
        examples=["person"],
    )
    type: str = Field(
        default="bbox",
        description="Annotation type: 'bbox' or 'polygon'.",
        examples=["polygon"],
    )
    points: list[Point] | None = Field(
        default=None,
        description="Normalized [0-1] coordinates for polygon vertices. Only for type='polygon'.",
        examples=[[{"x": 0.5, "y": 0.5}, {"x": 0.6, "y": 0.6}, {"x": 0.5, "y": 0.6}]],
    )


class AnnotationRequest(BaseModel):
    """
    Payload for saving annotations for a single image.

    The frontend sends absolute pixel coordinates along with the
    actual image dimensions so the backend can compute YOLO
    normalized values.
    """

    image_width: int = Field(
        ...,
        gt=0,
        description="Width of the source image in pixels.",
        examples=[1920],
    )
    image_height: int = Field(
        ...,
        gt=0,
        description="Height of the source image in pixels.",
        examples=[1080],
    )
    filename: str | None = Field(
        default=None,
        description="The filename of the image being annotated (used for validation).",
    )
    boxes: list[BoundingBoxSchema] = Field(
        default_factory=list,
        description=(
            "Array of bounding boxes. An empty array is valid and will "
            "create an empty YOLO annotation file (no objects)."
        ),
    )


class AnnotationResponse(BaseModel):
    """Response returned after annotations are saved."""

    message: str = Field(
        ...,
        description="Human-readable success message.",
        examples=["Annotations saved for image_001.jpg"],
    )
    label_file: str = Field(
        ...,
        description="Filename of the generated YOLO .txt file.",
        examples=["image_001.txt"],
    )
    box_count: int = Field(
        ...,
        ge=0,
        description="Number of bounding boxes written.",
        examples=[3],
    )


# ---------------------------------------------------------------------------
# Feature D: Export Dataset
# ---------------------------------------------------------------------------

class ExportResponse(BaseModel):
    """
    Response metadata for dataset export.

    Note: The actual endpoint returns a file download (StreamingResponse),
    but this schema is useful for error responses and documentation.
    """

    message: str = Field(
        ...,
        description="Human-readable status message.",
        examples=["Export ready for download."],
    )
    filename: str = Field(
        ...,
        description="Name of the exported zip file.",
        examples=["default_project.zip"],
    )


# ---------------------------------------------------------------------------
# Feature F: AI Auto-Labeling
# ---------------------------------------------------------------------------

class AutoLabelRequest(BaseModel):
    """Payload for triggering AI auto-labeling on a single image."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str = Field(
        ...,
        min_length=1,
        description="The filename of the YOLO .pt model to use for inference.",
        examples=["yolov8n.pt"],
    )


class AutoLabelResponse(BaseModel):
    """Response returned after auto-labeling completes."""

    message: str = Field(
        ...,
        description="Human-readable success message.",
        examples=["Auto-labeling completed successfully."],
    )
    boxes_added: int = Field(
        ...,
        ge=0,
        description="Number of new bounding boxes added by the AI model.",
        examples=[12],
    )
    classes_created: list[str] = Field(
        default_factory=list,
        description="Names of any new classes that were auto-created.",
        examples=[["person", "car"]],
    )


# ---------------------------------------------------------------------------
# General / Error
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """Standard error response body."""

    detail: str = Field(
        ...,
        description="Human-readable error description.",
        examples=["File not found: image_999.jpg"],
    )
