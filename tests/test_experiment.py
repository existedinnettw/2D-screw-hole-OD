from __future__ import annotations

import json
from pathlib import Path

import pytest

from two_d_screw_hole_od.config import PROMPTS
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


def test_prompts_target_screw_hole_variants() -> None:
    joined = " ".join(PROMPTS)

    assert "screw hole" in joined
    assert "circular hole" in joined


def test_save_report_writes_json(tmp_path: Path) -> None:
    report = ExperimentReport(
        image_url="https://example.test/image.webp",
        prompts=("screw hole",),
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
                    )
                ],
            )
        ],
        yolo_diagnostics=[],
    )

    output_path = save_report(report, tmp_path / "report.json")

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["prompts"] == ["screw hole"]
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
    request.node.experiment_report_path = report_path
    request.node.experiment_prompts = report.prompts
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
