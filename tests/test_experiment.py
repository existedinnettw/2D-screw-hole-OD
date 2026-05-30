from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from two_d_screw_hole_od.config import (
    GROUNDING_DINO_TARGET_PROMPTS,
    PROMPTS_BY_MODEL,
    YOLO_WORLD_DIAGNOSTIC_PROMPTS,
    YOLO_WORLD_TARGET_PROMPTS,
)
from two_d_screw_hole_od.experiment import (
    run_zero_shot_experiment,
    save_report,
)
from two_d_screw_hole_od.models.fs_detr import fs_detr_status
from two_d_screw_hole_od.types import (
    Detection,
    ExperimentReport,
    ModelResult,
)


FIXTURE_IMAGE = Path(__file__).parent / "fixtures" / "printables_screw_hole.webp"


def test_prompts_are_split_by_model() -> None:
    yolo_joined = " ".join(YOLO_WORLD_TARGET_PROMPTS)
    grounding_dino_joined = " ".join(GROUNDING_DINO_TARGET_PROMPTS)

    assert "screw hole" in yolo_joined
    assert "screw hole without screw" in yolo_joined
    assert "screw hole" in grounding_dino_joined
    assert "screw hole without screw" in grounding_dino_joined
    assert PROMPTS_BY_MODEL["YOLO-World"]["target"] == YOLO_WORLD_TARGET_PROMPTS
    assert PROMPTS_BY_MODEL["YOLO-World"]["diagnostic"] == YOLO_WORLD_DIAGNOSTIC_PROMPTS
    assert PROMPTS_BY_MODEL["Grounding DINO"]["target"] == GROUNDING_DINO_TARGET_PROMPTS


def test_save_report_writes_json(tmp_path: Path) -> None:
    report = ExperimentReport(
        image_url="https://example.test/image.webp",
        prompts_by_model={
            "YOLO-World": {
                "target": ("screw hole",),
                "diagnostic": ("screw holes",),
            },
            "Grounding DINO": {"target": ("screw hole",)},
            "FS-DETR": {"support": ()},
        },
        results=[
            ModelResult(
                model="fake-detector",
                status="ok",
                confidence_threshold=0.5,
                elapsed_seconds=0.1,
                inference_seconds=0.01,
                detections=[
                    Detection(
                        label="screw hole",
                        confidence=0.9,
                        box_xyxy=(1.0, 2.0, 3.0, 4.0),
                        text_match=0.41,
                    )
                ],
                text_threshold=0.25,
            )
        ],
        yolo_diagnostics=[],
    )

    output_path = save_report(report, tmp_path / "report.json")

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert_valid_report_payload(payload)
    assert payload["prompts_by_model"]["YOLO-World"]["target"] == ["screw hole"]
    assert payload["prompts_by_model"]["Grounding DINO"]["target"] == ["screw hole"]
    assert payload["results"][0]["text_threshold"] == 0.25
    assert payload["results"][0]["detections"][0]["text_match"] == 0.41
    assert payload["results"][0]["detections"][0]["box_xyxy"] == [1.0, 2.0, 3.0, 4.0]


def test_fs_detr_is_reported_as_few_shot_not_comparable() -> None:
    result = fs_detr_status()

    assert result.model == "FS-DETR"
    assert result.status == "not-run"
    assert "support images" in result.notes


@pytest.mark.model_experiment
def test_yolo_world_grounding_dino_and_fs_detr_on_target_image(
    request: pytest.FixtureRequest,
) -> None:
    assert FIXTURE_IMAGE.exists()

    report = run_zero_shot_experiment(FIXTURE_IMAGE)
    report_path = save_report(report, Path("artifacts/model_experiment_report.json"))
    assert_valid_report_payload(json.loads(report_path.read_text(encoding="utf-8")))
    request.node.experiment_report_path = report_path
    request.node.experiment_annotated_images = [
        result.annotated_image for result in report.results if result.annotated_image
    ]
    request.node.experiment_yolo_diagnostics = report.yolo_diagnostics

    result_by_model = {result.model: result for result in report.results}
    assert result_by_model["YOLO-World"].status == "ok"
    assert result_by_model["Grounding DINO"].status == "ok"
    assert result_by_model["FS-DETR"].status == "not-run"

    for model_name in ("YOLO-World", "Grounding DINO"):
        result = result_by_model[model_name]
        assert result.elapsed_seconds is not None
        assert result.elapsed_seconds > 0
        assert result.inference_seconds is not None
        assert result.inference_seconds > 0

    assert report.yolo_diagnostics


def assert_valid_report_payload(payload: dict[str, Any]) -> None:
    assert set(payload) == {
        "image_url",
        "prompts_by_model",
        "results",
        "yolo_diagnostics",
    }
    assert isinstance(payload["image_url"], str)
    assert payload["image_url"].startswith("https://")
    assert_prompts_by_model(payload["prompts_by_model"])
    assert isinstance(payload["results"], list)
    assert payload["results"]
    for result in payload["results"]:
        assert_model_result(result)
    assert isinstance(payload["yolo_diagnostics"], list)
    for diagnostic in payload["yolo_diagnostics"]:
        assert_yolo_diagnostic(diagnostic)


def assert_prompts_by_model(value: Any) -> None:
    assert isinstance(value, dict)
    assert set(value) == {"YOLO-World", "Grounding DINO", "FS-DETR"}
    for prompt_sets in value.values():
        assert isinstance(prompt_sets, dict)
        for prompts in prompt_sets.values():
            assert isinstance(prompts, list)
            assert all(isinstance(prompt, str) for prompt in prompts)


def assert_model_result(value: Any) -> None:
    assert set(value) == {
        "model",
        "status",
        "confidence_threshold",
        "elapsed_seconds",
        "inference_seconds",
        "detections",
        "text_threshold",
        "notes",
        "annotated_image",
    }
    assert isinstance(value["model"], str)
    assert isinstance(value["status"], str)
    assert value["confidence_threshold"] is None or isinstance(
        value["confidence_threshold"], int | float
    )
    assert value["elapsed_seconds"] is None or isinstance(
        value["elapsed_seconds"], int | float
    )
    assert value["inference_seconds"] is None or isinstance(
        value["inference_seconds"], int | float
    )
    assert value["text_threshold"] is None or isinstance(
        value["text_threshold"], int | float
    )
    assert isinstance(value["notes"], str)
    assert value["annotated_image"] is None or isinstance(value["annotated_image"], str)
    assert isinstance(value["detections"], list)
    for detection in value["detections"]:
        assert_detection(detection)


def assert_detection(value: Any) -> None:
    assert set(value) == {"label", "confidence", "box_xyxy", "text_match"}
    assert isinstance(value["label"], str)
    assert isinstance(value["confidence"], int | float)
    assert isinstance(value["box_xyxy"], list)
    assert len(value["box_xyxy"]) == 4
    assert all(isinstance(coordinate, int | float) for coordinate in value["box_xyxy"])
    assert value["text_match"] is None or isinstance(value["text_match"], int | float)


def assert_yolo_diagnostic(value: Any) -> None:
    assert set(value) == {
        "prompt_set",
        "confidence_threshold",
        "inference_seconds",
        "detections",
    }
    assert isinstance(value["prompt_set"], str)
    assert isinstance(value["confidence_threshold"], int | float)
    assert isinstance(value["inference_seconds"], int | float)
    assert isinstance(value["detections"], list)
    for detection in value["detections"]:
        assert_detection(detection)
