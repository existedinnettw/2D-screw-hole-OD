from __future__ import annotations

from .annotations import (
    BoxAnnotation,
    PromptPoint,
    annotations_to_payload,
    coerce_sam2_masks,
    coerce_sam2_prompt_masks,
    coerce_sam3_boxes,
    group_prompt_points,
    prompt_points_to_payload,
    save_annotations,
)
from .cli import main, parse_args
from .constants import (
    DEFAULT_IMAGE_DIR,
    DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_SAM2_CONFIG,
    IMAGE_EXTENSIONS,
    SAM2_TINY_CHECKPOINT_URL,
)
from .runners import Sam2Runner, Sam3Runner
from .ui import BBoxLabeler, find_default_image, tk_photo_from_image


__all__ = [
    "BBoxLabeler",
    "BoxAnnotation",
    "DEFAULT_IMAGE_DIR",
    "DEFAULT_SAM2_CHECKPOINT",
    "DEFAULT_SAM2_CONFIG",
    "IMAGE_EXTENSIONS",
    "PromptPoint",
    "SAM2_TINY_CHECKPOINT_URL",
    "Sam2Runner",
    "Sam3Runner",
    "annotations_to_payload",
    "coerce_sam2_masks",
    "coerce_sam2_prompt_masks",
    "coerce_sam3_boxes",
    "find_default_image",
    "group_prompt_points",
    "main",
    "parse_args",
    "prompt_points_to_payload",
    "save_annotations",
    "tk_photo_from_image",
]
