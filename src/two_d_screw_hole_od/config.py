from __future__ import annotations

from pathlib import Path

import requests


TARGET_IMAGE_URL = (
    "https://media.printables.com/media/prints/304678/images/"
    "2657447_21c90dd4-e7f2-415d-b8ff-c7336b9b4f1a/thumbs/inside/"
    "1600x1200/jpg/2022-10-27-222638.webp"
)

PROMPTS = (
    "screw hole",
    "circular hole",
    "round hole",
    "counterbore screw hole",
)

YOLO_WORLD_DIAGNOSTIC_PROMPTS = (
    "hole",
    "screw",
    "circle",
    "opening",
    "black circle",
    "round opening",
    "screw holes",
)


def download_target_image(path: Path, url: str = TARGET_IMAGE_URL) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path
