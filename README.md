# two-d-screw-hole-OD

two-d screw hole (k-shot) object detection algorithm experiment

## Model experiment

The pytest suite includes a live detector experiment for the Printables image
linked in the HackMD note. The image is stored locally at
`tests/fixtures/printables_screw_hole.webp` so repeated test runs do not depend
on re-downloading it.

Fast harness tests:

```bash
uv run pytest
```

Zero-shot detector run:

```bash
uv run pytest -m model_experiment -s
```

Detector run with a pytest-html report:

```bash
uv run pytest -m model_experiment --html=artifacts/model_experiment.html --self-contained-html
```

The live test compares YOLO-World and Grounding DINO on the target image and
writes `artifacts/model_experiment_report.json` plus annotated bounding-box
images. The JSON includes total elapsed time and model inference time. The HTML
report attaches the prompts, JSON report, YOLO-World diagnostics, and annotated
images. FS-DETR is recorded as `not-run` because it is a few-shot detector and
needs labeled support images, which are not available from a single target
image.

## TODO

* website version
