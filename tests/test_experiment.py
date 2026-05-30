from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from two_d_screw_hole_od.config import (
    GROUNDING_DINO_CONFIDENCE,
    GROUNDING_DINO_TARGET_PROMPTS,
    GROUNDING_DINO_TEXT_THRESHOLD,
    PROMPTS_BY_MODEL,
    YOLO_WORLD_CONFIDENCE,
    YOLO_WORLD_DIAGNOSTIC_CONFIDENCE_THRESHOLDS,
    YOLO_WORLD_DIAGNOSTIC_PROMPTS,
    YOLO_WORLD_TARGET_PROMPTS,
)
from two_d_screw_hole_od.experiment import (
    run_zero_shot_experiment,
    save_report,
)
from two_d_screw_hole_od.models.fs_detr import fs_detr_status
from two_d_screw_hole_od.models.grounding_dino import run_grounding_dino
from two_d_screw_hole_od.models.yolo_world import run_yolo_world_with_diagnostics
from two_d_screw_hole_od.types import (
    Detection,
    ExperimentReport,
    ModelResult,
    YoloDiagnostic,
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


def test_confidence_thresholds_are_configured_in_config() -> None:
    assert YOLO_WORLD_CONFIDENCE == 0.001
    assert GROUNDING_DINO_CONFIDENCE == 0.12
    assert GROUNDING_DINO_TEXT_THRESHOLD == 0.3
    assert YOLO_WORLD_DIAGNOSTIC_CONFIDENCE_THRESHOLDS == (0.02, 0.005, 0.001)


def test_model_confidence_arguments_must_be_passed_explicitly() -> None:
    yolo_parameters = inspect.signature(run_yolo_world_with_diagnostics).parameters
    grounding_dino_parameters = inspect.signature(run_grounding_dino).parameters

    assert yolo_parameters["confidence"].default is inspect.Parameter.empty
    assert yolo_parameters["confidence"].kind is inspect.Parameter.KEYWORD_ONLY
    assert grounding_dino_parameters["confidence"].default is inspect.Parameter.empty
    assert (
        grounding_dino_parameters["confidence"].kind is inspect.Parameter.KEYWORD_ONLY
    )
    assert (
        grounding_dino_parameters["text_threshold"].default is inspect.Parameter.empty
    )
    assert (
        grounding_dino_parameters["text_threshold"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )


def test_zero_shot_experiment_passes_configured_confidence_thresholds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, float] = {}

    def fake_run_yolo_world_with_diagnostics(
        image_path: Path,
        *,
        prompts: tuple[str, ...],
        confidence: float,
        device: str | None,
    ) -> tuple[ModelResult, list[YoloDiagnostic]]:
        captured["yolo_confidence"] = confidence
        return (
            ModelResult(
                model="YOLO-World",
                status="ok",
                confidence_threshold=confidence,
                elapsed_seconds=0.1,
                inference_seconds=0.01,
                detections=[],
            ),
            [],
        )

    def fake_run_grounding_dino(
        image_path: Path,
        *,
        prompts: tuple[str, ...],
        confidence: float,
        text_threshold: float,
        device: str | None,
    ) -> ModelResult:
        captured["grounding_dino_confidence"] = confidence
        captured["grounding_dino_text_threshold"] = text_threshold
        return ModelResult(
            model="Grounding DINO",
            status="ok",
            confidence_threshold=confidence,
            elapsed_seconds=0.1,
            inference_seconds=0.01,
            detections=[],
            text_threshold=text_threshold,
        )

    monkeypatch.setattr(
        "two_d_screw_hole_od.experiment.run_yolo_world_with_diagnostics",
        fake_run_yolo_world_with_diagnostics,
    )
    monkeypatch.setattr(
        "two_d_screw_hole_od.experiment.run_grounding_dino",
        fake_run_grounding_dino,
    )

    report = run_zero_shot_experiment(FIXTURE_IMAGE, output_dir=tmp_path)

    assert captured == {
        "yolo_confidence": YOLO_WORLD_CONFIDENCE,
        "grounding_dino_confidence": GROUNDING_DINO_CONFIDENCE,
        "grounding_dino_text_threshold": GROUNDING_DINO_TEXT_THRESHOLD,
    }
    assert [result.confidence_threshold for result in report.results[:2]] == [
        YOLO_WORLD_CONFIDENCE,
        GROUNDING_DINO_CONFIDENCE,
    ]


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
