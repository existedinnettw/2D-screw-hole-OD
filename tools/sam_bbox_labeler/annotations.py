from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BoxAnnotation:
    label: str
    box_xyxy: tuple[float, float, float, float]
    confidence: float | None = None
    source: str = "manual"


@dataclass(frozen=True)
class PromptPoint:
    x: float
    y: float
    label: int
    group: int = 1


def prompt_points_to_payload(points: list[PromptPoint]) -> list[dict[str, Any]]:
    return [
        {
            "x": point.x,
            "y": point.y,
            "label": point.label,
            "kind": "positive" if point.label == 1 else "negative",
            "group": point.group,
        }
        for point in points
    ]


def group_prompt_points(points: list[PromptPoint]) -> list[list[PromptPoint]]:
    groups: dict[int, list[PromptPoint]] = {}
    for point in points:
        groups.setdefault(point.group, []).append(point)
    return [groups[group] for group in sorted(groups)]


def annotations_to_payload(
    *,
    image_path: Path,
    image_size: tuple[int, int],
    annotations: list[BoxAnnotation],
) -> dict[str, Any]:
    return {
        "image": str(image_path),
        "width": image_size[0],
        "height": image_size[1],
        "annotations": [asdict(annotation) for annotation in annotations],
    }


def save_annotations(
    *,
    image_path: Path,
    image_size: tuple[int, int],
    annotations: list[BoxAnnotation],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            annotations_to_payload(
                image_path=image_path,
                image_size=image_size,
                annotations=annotations,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path


def coerce_sam3_boxes(output: dict[str, Any], prompt: str) -> list[BoxAnnotation]:
    boxes = _to_python_list(output.get("boxes", []))
    scores = _to_python_list(output.get("scores", []))
    annotations: list[BoxAnnotation] = []

    for index, raw_box in enumerate(boxes):
        if len(raw_box) != 4:
            continue
        score = scores[index] if index < len(scores) else None
        annotations.append(
            BoxAnnotation(
                label=prompt,
                box_xyxy=tuple(float(value) for value in raw_box),
                confidence=float(score) if score is not None else None,
                source="sam3",
            )
        )
    return annotations


def coerce_sam2_masks(masks: list[dict[str, Any]], prompt: str) -> list[BoxAnnotation]:
    annotations: list[BoxAnnotation] = []
    for mask in masks:
        raw_bbox = mask.get("bbox")
        if raw_bbox is None or len(raw_bbox) != 4:
            continue
        x, y, width, height = (float(value) for value in raw_bbox)
        score = mask.get("predicted_iou", mask.get("stability_score"))
        annotations.append(
            BoxAnnotation(
                label=prompt,
                box_xyxy=(x, y, x + width, y + height),
                confidence=float(score) if score is not None else None,
                source="sam2",
            )
        )
    return annotations


def coerce_sam2_prompt_masks(
    masks: Any, scores: Any, prompt: str
) -> list[BoxAnnotation]:
    masks_list = _to_python_list(masks)
    scores_list = _to_python_list(scores)
    if not masks_list:
        return []

    best_index = _best_score_index(scores_list, len(masks_list))
    box = _mask_to_box_xyxy(masks_list[best_index])
    if box is None:
        return []

    score = scores_list[best_index] if best_index < len(scores_list) else None
    return [
        BoxAnnotation(
            label=prompt,
            box_xyxy=box,
            confidence=float(score) if score is not None else None,
            source="sam2-points",
        )
    ]


def _to_python_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value is None:
        return []
    return list(value)


def _best_score_index(scores: list[Any], mask_count: int) -> int:
    if not scores:
        return 0
    scored = [(float(score), index) for index, score in enumerate(scores[:mask_count])]
    return max(scored)[1]


def _mask_to_box_xyxy(mask: Any) -> tuple[float, float, float, float] | None:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "SAM2 point-prompt bbox conversion requires numpy."
        ) from error

    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        return None
    y_indices, x_indices = np.where(mask_array)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None
    return (
        float(x_indices.min()),
        float(y_indices.min()),
        float(x_indices.max() + 1),
        float(y_indices.max() + 1),
    )
