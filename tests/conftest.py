from __future__ import annotations

import json
import base64
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
    prompts = getattr(item, "experiment_prompts", None)
    annotated_images = getattr(item, "experiment_annotated_images", None)
    yolo_diagnostics = getattr(item, "experiment_yolo_diagnostics", None)
    if not report_path:
        return

    report_extras = list(getattr(report, "extras", []))
    report_extras.append(extras.text("\n".join(prompts or ()), name="Text prompts"))
    report_extras.append(extras.json(_read_json(Path(report_path)), name="JSON report"))

    for image_path in annotated_images or ():
        path = Path(image_path)
        if path.exists():
            image_data = base64.b64encode(path.read_bytes()).decode("ascii")
            report_extras.append(extras.image(image_data, name=path.stem))

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
