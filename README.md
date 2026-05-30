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

## Manual bbox labeling with SAM2/SAM3

A small Tkinter labeler lives at `tools/sam3_bbox_labeler.py`. It opens the
fixture image by default, supports manual click-drag bounding boxes, and can
append model proposals. SAM2 is the default because its public checkpoints do
not require restricted Hugging Face access. SAM2 uses automatic mask generation
and converts mask boxes to bbox proposals; SAM3 uses the `--prompt` text prompt.

By default, pressing `Run SAM2` downloads the public SAM2.1 tiny checkpoint to
`checkpoints/sam2.1_hiera_tiny.pt` if it is missing. You can also download it
up front:

```bash
mkdir -p checkpoints
curl -L \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt \
  -o checkpoints/sam2.1_hiera_tiny.pt
```

```bash
uv run python tools/sam3_bbox_labeler.py \
  --image tests/fixtures/printables_screw_hole.webp \
  --output artifacts/printables_screw_hole.bboxes.json \
  --prompt "screw hole" \
  --model sam2
```

SAM3 auto-detection requires the upstream SAM3 runtime, CUDA-capable PyTorch,
and Hugging Face access to the `facebook/sam3` checkpoint. If those are not
available, use `--model sam2` or draw boxes manually.

SAM2 does not understand the `--prompt` text. For SAM2, `--prompt` is only the
label written to generated boxes. Use the canvas controls to switch between
manual box mode, positive point mode, and negative point mode. When prompt
points are present, `Run SAM2` uses them with `SAM2ImagePredictor`; without
points, it falls back to automatic mask proposals.

## TODO

* website version
* use `Flet` for GUI

## ref

* [introduction-to-the-sam-family-segmentation-with-sam-and-sam2-on-images-and-videos](https://medium.com/@siromermer/introduction-to-the-sam-family-segmentation-with-sam-and-sam2-on-images-and-videos-5223c5bf5cf1)
