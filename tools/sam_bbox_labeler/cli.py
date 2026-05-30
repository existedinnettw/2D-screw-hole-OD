from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .constants import DEFAULT_IMAGE_DIR, DEFAULT_SAM2_CHECKPOINT, DEFAULT_SAM2_CONFIG
from .ui import BBoxLabeler, find_default_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small bbox labeler with optional SAM2/SAM3 proposals."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Image to label. Defaults to the first image in tests/fixtures.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path. Defaults to prompting on first save.",
    )
    parser.add_argument(
        "--prompt",
        default="screw hole",
        help="Proposal label and default manual label. SAM3 uses it as text prompt.",
    )
    parser.add_argument(
        "--model",
        choices=("sam2", "sam3"),
        default="sam2",
        help="Proposal model. SAM2 uses automatic masks; SAM3 uses text prompts.",
    )
    parser.add_argument(
        "--sam2-checkpoint",
        type=Path,
        default=DEFAULT_SAM2_CHECKPOINT,
        help="Local SAM2 checkpoint path.",
    )
    parser.add_argument(
        "--sam2-config",
        default=DEFAULT_SAM2_CONFIG,
        help="SAM2 model config name or path.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional model device argument, for example cuda or cpu.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    image_path = args.image or find_default_image(DEFAULT_IMAGE_DIR)
    app = BBoxLabeler(
        image_path=image_path,
        output_path=args.output,
        prompt=args.prompt,
        device=args.device,
        model=args.model,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
    )
    app.mainloop()
