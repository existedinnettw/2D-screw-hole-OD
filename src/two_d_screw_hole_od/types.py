from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class ModelResult:
    model: str
    status: str
    confidence_threshold: float | None
    elapsed_seconds: float | None
    inference_seconds: float | None
    detections: list[Detection]
    notes: str = ""
    annotated_image: str | None = None


@dataclass(frozen=True)
class YoloDiagnostic:
    prompt_set: str
    confidence_threshold: float
    inference_seconds: float
    detections: list[Detection]


@dataclass(frozen=True)
class ExperimentReport:
    image_url: str
    prompts: tuple[str, ...]
    results: list[ModelResult]
    yolo_diagnostics: list[YoloDiagnostic]


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return to_jsonable(asdict(value))
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    return value
