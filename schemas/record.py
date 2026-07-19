"""Single source of truth for a generated/digitized question record.

Mirrors TRD.md Section 5 (Data Requirements). Every later component
(engine, render, distractors, verify, textgen, index, api) reads or
writes against this schema — do not duplicate field definitions
elsewhere.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Family(str, Enum):
    FOLD_PUNCH = "fold_punch"
    MIRROR_WATER = "mirror_water"
    ROTATION_SERIES = "rotation_series"


class ExamStyle(str, Enum):
    SSC = "SSC"
    UPSC = "UPSC"
    CAT = "CAT"
    DRDO_PILOT = "DRDO_pilot"
    GENERIC = "generic"


class Source(str, Enum):
    SYNTHETIC = "synthetic"
    SCANNED_BOOK = "scanned_book"


class ImagePaths(BaseModel):
    """Question image plus each option image (A-D), by option label."""

    question: str
    options: dict[str, str] = Field(default_factory=dict)


class QuestionText(BaseModel):
    stem: str
    option_labels: dict[str, str] = Field(default_factory=dict)
    explanation: str


class Verification(BaseModel):
    vlm_model: str | None = None
    vlm_answer: str | None = None
    agree: bool | None = None
    verified_at: datetime | None = None


class Record(BaseModel):
    question_id: UUID = Field(default_factory=uuid4)
    family: Family
    sub_type: str
    difficulty: int = Field(ge=1, le=5)
    params: dict = Field(default_factory=dict)
    image_paths: ImagePaths
    correct_option: str
    distractor_rules: list[str] = Field(default_factory=list)
    exam_style: ExamStyle = ExamStyle.GENERIC
    text: QuestionText
    tags: list[str] = Field(default_factory=list)
    embedding_text: str = ""
    verification: Verification = Field(default_factory=Verification)
    source: Source = Source.SYNTHETIC
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1
