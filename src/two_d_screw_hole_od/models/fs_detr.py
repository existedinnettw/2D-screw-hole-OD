from __future__ import annotations

from two_d_screw_hole_od.types import ModelResult


def fs_detr_status() -> ModelResult:
    return ModelResult(
        model="FS-DETR",
        status="not-run",
        confidence_threshold=None,
        elapsed_seconds=None,
        inference_seconds=None,
        detections=[],
        notes=(
            "FS-DETR is a few-shot detector and needs labeled support images. "
            "This single-image zero-shot experiment records it as not comparable."
        ),
    )
