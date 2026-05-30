from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

from two_d_screw_hole_od.config import PROMPTS, TARGET_IMAGE_URL
from two_d_screw_hole_od.models import (
    diagnose_yolo_world,
    fs_detr_status,
    run_grounding_dino,
    run_yolo_world,
)
from two_d_screw_hole_od.types import ExperimentReport, to_jsonable
from two_d_screw_hole_od.visualization import draw_detections


LOGGER = logging.getLogger(__name__)


def run_zero_shot_experiment(
    image_path: Path,
    output_dir: Path = Path("artifacts"),
    device: str | None = None,
) -> ExperimentReport:
    LOGGER.info("text prompts: %s", ", ".join(PROMPTS))
    yolo_result = run_yolo_world(image_path, confidence=0.006, device=device)
    grounding_dino_result = run_grounding_dino(image_path, device=device)
    fs_detr_result = fs_detr_status()
    for result in (yolo_result, grounding_dino_result, fs_detr_result):
        LOGGER.info(
            "%s status=%s elapsed=%s inference=%s confidence_threshold=%s detections=%d",
            result.model,
            result.status,
            _format_seconds(result.elapsed_seconds),
            _format_seconds(result.inference_seconds),
            result.confidence_threshold,
            len(result.detections),
        )

    yolo_image = draw_detections(
        image_path=image_path,
        detections=yolo_result.detections,
        output_path=output_dir / "yolo_world_boxes.png",
        title="YOLO-World detections",
    )
    grounding_dino_image = draw_detections(
        image_path=image_path,
        detections=grounding_dino_result.detections,
        output_path=output_dir / "grounding_dino_boxes.png",
        title="Grounding DINO detections",
    )

    return ExperimentReport(
        image_url=TARGET_IMAGE_URL,
        prompts=PROMPTS,
        results=[
            replace(yolo_result, annotated_image=str(yolo_image)),
            replace(grounding_dino_result, annotated_image=str(grounding_dino_image)),
            fs_detr_result,
        ],
        yolo_diagnostics=diagnose_yolo_world(image_path, device=device),
    )


def save_report(report: ExperimentReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(report), indent=2), encoding="utf-8")
    return path


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}s"
