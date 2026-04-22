"""
Boxify Backend — Database Models

SQLAlchemy models for Users, Projects, ImageRecords, and ModelRecords.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum, UniqueConstraint
from sqlalchemy.orm import relationship

from core.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)

    projects = relationship("Project", back_populates="owner", cascade="all, delete-orphan")


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    owner = relationship("User", back_populates="projects")
    images = relationship("ImageRecord", back_populates="project", cascade="all, delete-orphan")
    classes = relationship("ProjectClass", back_populates="project", cascade="all, delete-orphan", order_by="ProjectClass.id")
    models = relationship("ModelRecord", back_populates="project", cascade="all, delete-orphan", order_by="ModelRecord.uploaded_at")


class ImageRecord(Base):
    __tablename__ = "image_records"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    status = Column(Enum("pending", "done", name="image_status_enum"), default="pending", nullable=False)

    project = relationship("Project", back_populates="images")


class ProjectClass(Base):
    """
    Represents a single annotation class belonging to a project.
    YOLO class index = 0-based position in the list sorted by ``id`` ASC.
    """
    __tablename__ = "project_classes"
    __table_args__ = (
        UniqueConstraint("project_id", "name", name="uq_project_class_name"),
    )

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    color = Column(String(7), nullable=False, default="#ef4444")  # Hex colour

    project = relationship("Project", back_populates="classes")


class ModelRecord(Base):
    """
    Tracks YOLO .pt model files uploaded to a project.
    """
    __tablename__ = "model_records"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    project = relationship("Project", back_populates="models")
