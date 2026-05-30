from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PIL import Image

from two_d_screw_hole_od.config import PROMPTS
from two_d_screw_hole_od.types import Detection, ModelResult


def run_grounding_dino(
    image_path: Path,
    prompts: tuple[str, ...] = PROMPTS,
    confidence: float = 0.25,
    text_threshold: float = 0.25,
    device: str | None = None,
    model_id: str = "IDEA-Research/grounding-dino-tiny",
) -> ModelResult:
    started_at = time.perf_counter()

    try:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        image = Image.open(image_path).convert("RGB")
        text = ". ".join(prompts) + "."

        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
            selected_device
        )
        inputs = processor(images=image, text=text, return_tensors="pt").to(
            selected_device
        )

        inference_started_at = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
        inference_seconds = time.perf_counter() - inference_started_at

        target_sizes = [image.size[::-1]]
        processed = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=confidence,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
            text_labels=[list(prompts)],
        )[0]
        detections = _detections_from_grounding_dino_result(processed)
        return ModelResult(
            model="Grounding DINO",
            status="ok",
            confidence_threshold=confidence,
            elapsed_seconds=time.perf_counter() - started_at,
            inference_seconds=inference_seconds,
            detections=detections,
        )
    except Exception as exc:  # pragma: no cover - exercised by opt-in live runs.
        return ModelResult(
            model="Grounding DINO",
            status="error",
            confidence_threshold=confidence,
            elapsed_seconds=time.perf_counter() - started_at,
            inference_seconds=None,
            detections=[],
            notes=repr(exc),
        )


def _detections_from_grounding_dino_result(result: dict[str, Any]) -> list[Detection]:
    labels = result.get("text_labels", result.get("labels", []))
    scores = result.get("scores", [])
    boxes = result.get("boxes", [])

    detections: list[Detection] = []
    for label, score, box in zip(labels, scores, boxes, strict=False):
        detections.append(
            Detection(
                label=str(label),
                confidence=float(score.item() if hasattr(score, "item") else score),
                box_xyxy=tuple(
                    float(value)
                    for value in (box.tolist() if hasattr(box, "tolist") else box)
                ),
            )
        )
    return detections
