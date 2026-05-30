#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from sam_bbox_labeler import (  # noqa: E402
    BBoxLabeler,
    BoxAnnotation,
    DEFAULT_IMAGE_DIR,
    DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_SAM2_CONFIG,
    IMAGE_EXTENSIONS,
    PromptPoint,
    SAM2_TINY_CHECKPOINT_URL,
    Sam2Runner,
    Sam3Runner,
    annotations_to_payload,
    coerce_sam2_masks,
    coerce_sam2_prompt_masks,
    coerce_sam3_boxes,
    find_default_image,
    group_prompt_points,
    main,
    parse_args,
    prompt_points_to_payload,
    save_annotations,
    tk_photo_from_image,
)


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


if __name__ == "__main__":
    main()
