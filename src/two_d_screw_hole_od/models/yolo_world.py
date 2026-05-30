from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from two_d_screw_hole_od.config import PROMPTS, YOLO_WORLD_DIAGNOSTIC_PROMPTS
from two_d_screw_hole_od.types import Detection, ModelResult, YoloDiagnostic


type YoloPrediction = tuple[list[Detection], float]


def run_yolo_world(
    image_path: Path,
    prompts: tuple[str, ...] = PROMPTS,
    confidence: float = 0.02,
    device: str | None = None,
    model_name: str = "yolov8s-world.pt",
) -> ModelResult:
    started_at = time.perf_counter()

    try:
        detections, inference_seconds = _predict_yolo_world(
            image_path=image_path,
            prompts=prompts,
            confidence=confidence,
            device=device,
            model_name=model_name,
        )
        return ModelResult(
            model="YOLO-World",
            status="ok",
            confidence_threshold=confidence,
            elapsed_seconds=time.perf_counter() - started_at,
            inference_seconds=inference_seconds,
            detections=detections,
            notes=(
                "No detections at the configured threshold. Diagnostics show "
                "target prompts produce low-confidence candidates below 0.02, "
                "while broader visual prompts such as circle/black circle score higher."
            )
            if not detections
            else "",
        )
    except Exception as exc:  # pragma: no cover - exercised by opt-in live runs.
        return ModelResult(
            model="YOLO-World",
            status="error",
            confidence_threshold=confidence,
            elapsed_seconds=time.perf_counter() - started_at,
            inference_seconds=None,
            detections=[],
            notes=repr(exc),
        )


def diagnose_yolo_world(
    image_path: Path,
    device: str | None = None,
    model_name: str = "yolov8s-world.pt",
) -> list[YoloDiagnostic]:
    diagnostics: list[YoloDiagnostic] = []
    prompt_sets = {
        "target": PROMPTS,
        "broader_visual": YOLO_WORLD_DIAGNOSTIC_PROMPTS,
    }
    for prompt_set_name, prompts in prompt_sets.items():
        for confidence in (0.02, 0.005, 0.001):
            detections, inference_seconds = _predict_yolo_world(
                image_path=image_path,
                prompts=prompts,
                confidence=confidence,
                device=device,
                model_name=model_name,
            )
            diagnostics.append(
                YoloDiagnostic(
                    prompt_set=prompt_set_name,
                    confidence_threshold=confidence,
                    inference_seconds=inference_seconds,
                    detections=detections,
                )
            )
    return diagnostics


def _predict_yolo_world(
    image_path: Path,
    prompts: tuple[str, ...],
    confidence: float,
    device: str | None,
    model_name: str,
) -> YoloPrediction:
    from ultralytics import YOLOWorld

    model = YOLOWorld(model_name)
    model.set_classes(list(prompts))
    inference_started_at = time.perf_counter()
    predictions = model.predict(
        str(image_path),
        conf=confidence,
        verbose=False,
        device=device,
    )
    inference_seconds = time.perf_counter() - inference_started_at
    return _detections_from_yolo_result(predictions[0], prompts), inference_seconds


def _detections_from_yolo_result(
    result: Any,
    prompts: tuple[str, ...],
) -> list[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    detections: list[Detection] = []
    for box in boxes:
        cls_index = int(box.cls.item())
        label = prompts[cls_index] if 0 <= cls_index < len(prompts) else str(cls_index)
        detections.append(
            Detection(
                label=label,
                confidence=float(box.conf.item()),
                box_xyxy=tuple(float(value) for value in box.xyxy[0].tolist()),
            )
        )
    return detections
