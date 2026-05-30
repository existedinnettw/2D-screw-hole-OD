from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from two_d_screw_hole_od.config import (
    YOLO_WORLD_DIAGNOSTIC_CONFIDENCE_THRESHOLDS,
    YOLO_WORLD_DIAGNOSTIC_PROMPTS,
    YOLO_WORLD_TARGET_PROMPTS,
)
from two_d_screw_hole_od.types import Detection, ModelResult, YoloDiagnostic


type YoloPrediction = tuple[list[Detection], float]


class YoloWorldDetector:
    def __init__(self, model_name: str = "yolov8s-world.pt") -> None:
        self._model_name = model_name
        self._model: Any | None = None

    def predict(
        self,
        image_path: Path,
        prompts: tuple[str, ...],
        confidence: float,
        device: str | None,
    ) -> YoloPrediction:
        model = self._load_model()
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

    def _load_model(self) -> Any:
        if self._model is None:
            from ultralytics import YOLOWorld

            self._model = YOLOWorld(self._model_name)
        return self._model


def run_yolo_world(
    image_path: Path,
    *,
    confidence: float,
    prompts: tuple[str, ...] = YOLO_WORLD_TARGET_PROMPTS,
    device: str | None = None,
    model_name: str = "yolov8s-world.pt",
    detector: YoloWorldDetector | None = None,
) -> ModelResult:
    started_at = time.perf_counter()

    try:
        yolo_detector = detector or YoloWorldDetector(model_name)
        detections, inference_seconds = yolo_detector.predict(
            image_path, prompts, confidence, device
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


def run_yolo_world_with_diagnostics(
    image_path: Path,
    *,
    confidence: float,
    prompts: tuple[str, ...] = YOLO_WORLD_TARGET_PROMPTS,
    device: str | None = None,
    model_name: str = "yolov8s-world.pt",
) -> tuple[ModelResult, list[YoloDiagnostic]]:
    detector = YoloWorldDetector(model_name)
    diagnostic_thresholds = YOLO_WORLD_DIAGNOSTIC_CONFIDENCE_THRESHOLDS
    lowest_target_confidence = min(confidence, *diagnostic_thresholds)
    combined_prompts = _combined_prompts(prompts, YOLO_WORLD_DIAGNOSTIC_PROMPTS)
    started_at = time.perf_counter()

    try:
        all_detections, inference_seconds = detector.predict(
            image_path, combined_prompts, lowest_target_confidence, device
        )
        target_detections = _filter_detections_by_label(
            all_detections, allowed_labels=prompts
        )
        detections = _filter_detections(target_detections, confidence)
        result = ModelResult(
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
        return (
            ModelResult(
                model="YOLO-World",
                status="error",
                confidence_threshold=confidence,
                elapsed_seconds=time.perf_counter() - started_at,
                inference_seconds=None,
                detections=[],
                notes=repr(exc),
            ),
            [],
        )

    diagnostics = _diagnostics_from_prediction(
        all_detections, inference_seconds, diagnostic_thresholds, target_prompts=prompts
    )
    return result, diagnostics


def diagnose_yolo_world(
    image_path: Path,
    device: str | None = None,
    model_name: str = "yolov8s-world.pt",
) -> list[YoloDiagnostic]:
    thresholds = YOLO_WORLD_DIAGNOSTIC_CONFIDENCE_THRESHOLDS
    combined_prompts = _combined_prompts(
        YOLO_WORLD_TARGET_PROMPTS,
        YOLO_WORLD_DIAGNOSTIC_PROMPTS,
    )
    detections, inference_seconds = YoloWorldDetector(model_name).predict(
        image_path, combined_prompts, min(thresholds), device
    )
    return _diagnostics_from_prediction(
        detections,
        inference_seconds,
        thresholds,
        target_prompts=YOLO_WORLD_TARGET_PROMPTS,
    )


def _diagnostics_from_prediction(
    detections: list[Detection],
    inference_seconds: float,
    confidence_thresholds: tuple[float, ...],
    target_prompts: tuple[str, ...],
) -> list[YoloDiagnostic]:
    diagnostics: list[YoloDiagnostic] = []
    prompt_sets = {
        "target": target_prompts,
        "broader_visual": YOLO_WORLD_DIAGNOSTIC_PROMPTS,
    }
    for prompt_set_name, prompts in prompt_sets.items():
        prompt_detections = _filter_detections_by_label(detections, prompts)
        for confidence in confidence_thresholds:
            diagnostics.append(
                YoloDiagnostic(
                    prompt_set=prompt_set_name,
                    confidence_threshold=confidence,
                    inference_seconds=inference_seconds,
                    detections=_filter_detections(prompt_detections, confidence),
                )
            )
    return diagnostics


def _filter_detections(
    detections: list[Detection],
    confidence: float,
) -> list[Detection]:
    return [detection for detection in detections if detection.confidence >= confidence]


def _filter_detections_by_label(
    detections: list[Detection],
    allowed_labels: tuple[str, ...],
) -> list[Detection]:
    labels = set(allowed_labels)
    return [detection for detection in detections if detection.label in labels]


def _combined_prompts(*prompt_sets: tuple[str, ...]) -> tuple[str, ...]:
    prompts: list[str] = []
    for prompt_set in prompt_sets:
        for prompt in prompt_set:
            if prompt not in prompts:
                prompts.append(prompt)
    return tuple(prompts)


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
                box_xyxy=_box_xyxy(box.xyxy[0].tolist()),
            )
        )
    return detections


def _box_xyxy(values: Any) -> tuple[float, float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))
