from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from pytest_html import extras

from two_d_screw_hole_od.types import to_jsonable


logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[Any]):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call":
        return

    report_path = getattr(item, "experiment_report_path", None)
    annotated_images = getattr(item, "experiment_annotated_images", None)
    yolo_diagnostics = getattr(item, "experiment_yolo_diagnostics", None)
    if not report_path:
        return

    report_path = Path(report_path)
    payload = _read_json(report_path)
    report_extras = list(getattr(report, "extras", []))
    report_extras.append(
        extras.text(_format_model_prompts(payload), name="Model prompts")
    )
    report_extras.append(
        extras.url(
            _image_report_path(report_path, report_path.parent), name="JSON report"
        )
    )

    for image_path in annotated_images or ():
        path = Path(image_path)
        if path.exists():
            report_extras.append(
                extras.image(
                    _image_report_path(path, report_path.parent), name=path.stem
                )
            )

    if yolo_diagnostics:
        report_extras.append(
            extras.text(
                json.dumps(to_jsonable(yolo_diagnostics), indent=2),
                name="YOLO-World diagnostics",
            )
        )
    report.extras = report_extras


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_model_prompts(payload: dict[str, Any]) -> str:
    prompt_lines: list[str] = []
    prompts_by_model = payload.get("prompts_by_model", {})
    if not isinstance(prompts_by_model, dict):
        return ""

    for model_name, prompt_sets in prompts_by_model.items():
        prompt_lines.append(str(model_name))
        if not isinstance(prompt_sets, dict):
            continue
        for prompt_set_name, prompts in prompt_sets.items():
            prompt_lines.append(f"  {prompt_set_name}:")
            for prompt in prompts:
                prompt_lines.append(f"    - {prompt}")
    return "\n".join(prompt_lines)


def _image_report_path(path: Path, report_dir: Path) -> str:
    try:
        return str(path.relative_to(report_dir))
    except ValueError:
        return str(path)
