from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


TOOL_PATH = Path(__file__).parents[1] / "tools" / "sam3_bbox_labeler.py"


def load_tool_module() -> Any:
    spec = importlib.util.spec_from_file_location("sam3_bbox_labeler", TOOL_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_coerce_sam3_boxes_converts_output_to_annotations() -> None:
    tool = load_tool_module()

    annotations = tool.coerce_sam3_boxes(
        {"boxes": [[1, 2, 3, 4], [10.5, 20.5, 30.5, 40.5]], "scores": [0.91, 0.42]},
        "screw hole",
    )

    assert annotations == [
        tool.BoxAnnotation(
            label="screw hole",
            box_xyxy=(1.0, 2.0, 3.0, 4.0),
            confidence=0.91,
            source="sam3",
        ),
        tool.BoxAnnotation(
            label="screw hole",
            box_xyxy=(10.5, 20.5, 30.5, 40.5),
            confidence=0.42,
            source="sam3",
        ),
    ]


def test_coerce_sam2_masks_converts_xywh_to_xyxy_annotations() -> None:
    tool = load_tool_module()

    annotations = tool.coerce_sam2_masks(
        [
            {"bbox": [10, 20, 30, 40], "predicted_iou": 0.88},
            {"bbox": [1.5, 2.5, 3.5, 4.5], "stability_score": 0.77},
        ],
        "screw hole",
    )

    assert annotations == [
        tool.BoxAnnotation(
            label="screw hole",
            box_xyxy=(10.0, 20.0, 40.0, 60.0),
            confidence=0.88,
            source="sam2",
        ),
        tool.BoxAnnotation(
            label="screw hole",
            box_xyxy=(1.5, 2.5, 5.0, 7.0),
            confidence=0.77,
            source="sam2",
        ),
    ]


def test_coerce_sam2_prompt_masks_keeps_best_mask_bbox() -> None:
    tool = load_tool_module()

    annotations = tool.coerce_sam2_prompt_masks(
        [
            [[False, False, False], [False, True, False], [False, False, False]],
            [[False, True, True], [False, True, False], [False, False, False]],
        ],
        [0.2, 0.9],
        "screw hole",
    )

    assert annotations == [
        tool.BoxAnnotation(
            label="screw hole",
            box_xyxy=(1.0, 0.0, 3.0, 2.0),
            confidence=0.9,
            source="sam2-points",
        )
    ]


def test_prompt_points_to_payload_labels_positive_and_negative_points() -> None:
    tool = load_tool_module()

    payload = tool.prompt_points_to_payload(
        [
            tool.PromptPoint(x=10.0, y=20.0, label=1),
            tool.PromptPoint(x=30.5, y=40.5, label=0),
        ]
    )

    assert payload == [
        {"x": 10.0, "y": 20.0, "label": 1, "kind": "positive", "group": 1},
        {"x": 30.5, "y": 40.5, "label": 0, "kind": "negative", "group": 1},
    ]


def test_group_prompt_points_orders_points_by_object_group() -> None:
    tool = load_tool_module()

    grouped = tool.group_prompt_points(
        [
            tool.PromptPoint(x=30.0, y=40.0, label=1, group=2),
            tool.PromptPoint(x=10.0, y=20.0, label=1, group=1),
            tool.PromptPoint(x=35.0, y=45.0, label=0, group=2),
        ]
    )

    assert grouped == [
        [tool.PromptPoint(x=10.0, y=20.0, label=1, group=1)],
        [
            tool.PromptPoint(x=30.0, y=40.0, label=1, group=2),
            tool.PromptPoint(x=35.0, y=45.0, label=0, group=2),
        ],
    ]


def test_save_annotations_writes_bbox_json(tmp_path: Path) -> None:
    tool = load_tool_module()
    output_path = tmp_path / "labels.json"

    tool.save_annotations(
        image_path=Path("tests/fixtures/printables_screw_hole.webp"),
        image_size=(1421, 1200),
        annotations=[
            tool.BoxAnnotation(
                label="screw hole",
                box_xyxy=(1.0, 2.0, 3.0, 4.0),
                confidence=None,
                source="manual",
            )
        ],
        output_path=output_path,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == {
        "image": "tests/fixtures/printables_screw_hole.webp",
        "width": 1421,
        "height": 1200,
        "annotations": [
            {
                "label": "screw hole",
                "box_xyxy": [1.0, 2.0, 3.0, 4.0],
                "confidence": None,
                "source": "manual",
            }
        ],
    }
