from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from two_d_screw_hole_od.types import Detection


def draw_detections(
    image_path: Path,
    detections: list[Detection],
    output_path: Path,
    title: str,
    label_suffix: str = "",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.rectangle((8, 8, 8 + max(220, len(title) * 7), 34), fill=(255, 255, 255))
    draw.text((14, 14), title, fill=(0, 0, 0), font=font)

    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection.box_xyxy
        color = _color_for_index(index)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        label = _detection_label(index, detection, label_suffix)
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_height = text_bbox[3] - text_bbox[1]
        text_width = text_bbox[2] - text_bbox[0]
        label_y = max(0, y1 - text_height - 6)
        label_x = min(max(0, x1), max(0, image.width - text_width - 8))
        draw.rectangle(
            (label_x, label_y, label_x + text_width + 8, label_y + text_height + 6),
            fill=color,
        )
        draw.text((label_x + 4, label_y + 3), label, fill=(0, 0, 0), font=font)

    image.save(output_path)
    return output_path


def _color_for_index(index: int) -> tuple[int, int, int]:
    palette = (
        (255, 196, 0),
        (0, 190, 255),
        (255, 96, 96),
        (0, 220, 130),
        (210, 140, 255),
        (255, 145, 0),
    )
    return palette[(index - 1) % len(palette)]


def _detection_label(index: int, detection: Detection, label_suffix: str) -> str:
    label = f'{index}. score={detection.confidence:.3f} label="{detection.label}"'
    if detection.text_match is not None:
        label = f"{label} text_match={detection.text_match:.3f}"
    return f"{label}{label_suffix}"
