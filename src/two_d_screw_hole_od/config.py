from __future__ import annotations

from pathlib import Path

import requests

TARGET_IMAGE_URL = (
    "https://media.printables.com/media/prints/304678/images/"
    "2657447_21c90dd4-e7f2-415d-b8ff-c7336b9b4f1a/thumbs/inside/"
    "1600x1200/jpg/2022-10-27-222638.webp"
)

YOLO_WORLD_TARGET_PROMPTS = (
    "screw hole",
    "screw hole without screw",
    "screw",
    "a circular screw hole",
    # "screw hole with screw",
    # "circular hole",
    # "round hole",
    # "counterbore screw hole",
)

YOLO_WORLD_DIAGNOSTIC_PROMPTS = (
    # "hole",
    # "screw",
    # "circle",
    # "opening",
    # "black circle",
    # "round opening",
    "screw holes",
)

GROUNDING_DINO_TARGET_PROMPTS = (
    "a circular screw hole",
    "a circular screw hole without screw",
    # "screw hole for fixing board",
    # "screw",
)

FS_DETR_SUPPORT_PROMPTS: tuple[str, ...] = ()

YOLO_WORLD_CONFIDENCE = 0.001
YOLO_WORLD_DIAGNOSTIC_CONFIDENCE_THRESHOLDS = (0.02, 0.005, 0.001)
GROUNDING_DINO_CONFIDENCE = 0.12
GROUNDING_DINO_TEXT_THRESHOLD = 0.3

PROMPTS_BY_MODEL: dict[str, dict[str, tuple[str, ...]]] = {
    "YOLO-World": {
        "target": YOLO_WORLD_TARGET_PROMPTS,
        "diagnostic": YOLO_WORLD_DIAGNOSTIC_PROMPTS,
    },
    "Grounding DINO": {"target": GROUNDING_DINO_TARGET_PROMPTS},
    "FS-DETR": {"support": FS_DETR_SUPPORT_PROMPTS},
}

# Backwards-compatible alias for existing prompt-focused tests and callers.
PROMPTS = GROUNDING_DINO_TARGET_PROMPTS


def download_target_image(path: Path, url: str = TARGET_IMAGE_URL) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path
