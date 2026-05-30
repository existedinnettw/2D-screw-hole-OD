#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import logging
import queue
import threading
import time
import tkinter as tk
from dataclasses import asdict, dataclass
from pathlib import Path
from tkinter import filedialog
from typing import Any

from PIL import Image


DEFAULT_IMAGE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
DEFAULT_SAM2_CHECKPOINT = (
    Path(__file__).resolve().parents[1] / "checkpoints" / "sam2.1_hiera_tiny.pt"
)
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_TINY_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
)
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
TOOLBAR_HEIGHT = 46
SIDE_PANEL_WIDTH = 330
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoxAnnotation:
    label: str
    box_xyxy: tuple[float, float, float, float]
    confidence: float | None = None
    source: str = "manual"


@dataclass(frozen=True)
class PromptPoint:
    x: float
    y: float
    label: int


def prompt_points_to_payload(points: list[PromptPoint]) -> list[dict[str, Any]]:
    return [
        {
            "x": point.x,
            "y": point.y,
            "label": point.label,
            "kind": "positive" if point.label == 1 else "negative",
        }
        for point in points
    ]


def annotations_to_payload(
    *,
    image_path: Path,
    image_size: tuple[int, int],
    annotations: list[BoxAnnotation],
) -> dict[str, Any]:
    return {
        "image": str(image_path),
        "width": image_size[0],
        "height": image_size[1],
        "annotations": [asdict(annotation) for annotation in annotations],
    }


def save_annotations(
    *,
    image_path: Path,
    image_size: tuple[int, int],
    annotations: list[BoxAnnotation],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            annotations_to_payload(
                image_path=image_path,
                image_size=image_size,
                annotations=annotations,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path


def coerce_sam3_boxes(output: dict[str, Any], prompt: str) -> list[BoxAnnotation]:
    boxes = _to_python_list(output.get("boxes", []))
    scores = _to_python_list(output.get("scores", []))
    annotations: list[BoxAnnotation] = []

    for index, raw_box in enumerate(boxes):
        if len(raw_box) != 4:
            continue
        score = scores[index] if index < len(scores) else None
        annotations.append(
            BoxAnnotation(
                label=prompt,
                box_xyxy=tuple(float(value) for value in raw_box),
                confidence=float(score) if score is not None else None,
                source="sam3",
            )
        )
    return annotations


def coerce_sam2_masks(masks: list[dict[str, Any]], prompt: str) -> list[BoxAnnotation]:
    annotations: list[BoxAnnotation] = []
    for mask in masks:
        raw_bbox = mask.get("bbox")
        if raw_bbox is None or len(raw_bbox) != 4:
            continue
        x, y, width, height = (float(value) for value in raw_bbox)
        score = mask.get("predicted_iou", mask.get("stability_score"))
        annotations.append(
            BoxAnnotation(
                label=prompt,
                box_xyxy=(x, y, x + width, y + height),
                confidence=float(score) if score is not None else None,
                source="sam2",
            )
        )
    return annotations


def coerce_sam2_prompt_masks(
    masks: Any, scores: Any, prompt: str
) -> list[BoxAnnotation]:
    masks_list = _to_python_list(masks)
    scores_list = _to_python_list(scores)
    if not masks_list:
        return []

    best_index = _best_score_index(scores_list, len(masks_list))
    box = _mask_to_box_xyxy(masks_list[best_index])
    if box is None:
        return []

    score = scores_list[best_index] if best_index < len(scores_list) else None
    return [
        BoxAnnotation(
            label=prompt,
            box_xyxy=box,
            confidence=float(score) if score is not None else None,
            source="sam2-points",
        )
    ]


def _to_python_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value is None:
        return []
    return list(value)


def _best_score_index(scores: list[Any], mask_count: int) -> int:
    if not scores:
        return 0
    scored = [(float(score), index) for index, score in enumerate(scores[:mask_count])]
    return max(scored)[1]


def _mask_to_box_xyxy(mask: Any) -> tuple[float, float, float, float] | None:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "SAM2 point-prompt bbox conversion requires numpy."
        ) from error

    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        return None
    y_indices, x_indices = np.where(mask_array)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None
    return (
        float(x_indices.min()),
        float(y_indices.min()),
        float(x_indices.max() + 1),
        float(y_indices.max() + 1),
    )


class Sam3Runner:
    def __init__(self, device: str | None = None) -> None:
        self._device = device
        self._processor: Any | None = None

    def predict(
        self, image_path: Path, prompt: str, points: list[PromptPoint]
    ) -> list[BoxAnnotation]:
        if self._processor is None:
            self._processor = self._build_processor()

        image = Image.open(image_path).convert("RGB")
        state = self._processor.set_image(image)
        output = self._processor.set_text_prompt(state=state, prompt=prompt)
        return coerce_sam3_boxes(output, prompt)

    def _build_processor(self) -> Any:
        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as error:
            raise RuntimeError(
                "SAM3 is not installed. Install facebookresearch/sam3 in this "
                "environment and authenticate to Hugging Face for the facebook/sam3 "
                "checkpoint before running auto-detection."
            ) from error

        model_kwargs: dict[str, str] = {}
        if self._device:
            model_kwargs["device"] = self._device
        model = build_sam3_image_model(**model_kwargs)
        return Sam3Processor(model)


class Sam2Runner:
    def __init__(
        self,
        *,
        checkpoint: Path,
        config: str,
        device: str | None = None,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.9,
        min_mask_region_area: int = 25,
        checkpoint_url: str = SAM2_TINY_CHECKPOINT_URL,
    ) -> None:
        self._checkpoint = checkpoint
        self._config = config
        self._device = device
        self._points_per_side = points_per_side
        self._pred_iou_thresh = pred_iou_thresh
        self._stability_score_thresh = stability_score_thresh
        self._min_mask_region_area = min_mask_region_area
        self._checkpoint_url = checkpoint_url
        self._mask_generator: Any | None = None
        self._predictor: Any | None = None

    def predict(
        self, image_path: Path, prompt: str, points: list[PromptPoint]
    ) -> list[BoxAnnotation]:
        if points:
            return self._predict_with_points(image_path, prompt, points)
        if self._mask_generator is None:
            self._mask_generator = self._build_mask_generator()

        image = Image.open(image_path).convert("RGB")
        masks = self._mask_generator.generate(_image_to_numpy(image))
        return coerce_sam2_masks(masks, prompt)

    def _predict_with_points(
        self, image_path: Path, prompt: str, points: list[PromptPoint]
    ) -> list[BoxAnnotation]:
        if self._predictor is None:
            self._predictor = self._build_predictor()

        image = Image.open(image_path).convert("RGB")
        self._predictor.set_image(_image_to_numpy(image))
        point_coords, point_labels = _points_to_numpy(points)
        masks, scores, _logits = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        return coerce_sam2_prompt_masks(masks, scores, prompt)

    def _build_mask_generator(self) -> Any:
        if not self._checkpoint.exists():
            self._download_checkpoint()

        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2
        except ImportError as error:
            raise RuntimeError(
                "SAM2 is not installed. Install facebookresearch/sam2 in this "
                "environment before running SAM2 auto-detection."
            ) from error

        model_kwargs: dict[str, str] = {}
        if self._device:
            model_kwargs["device"] = self._device
        model = build_sam2(
            self._config,
            str(self._checkpoint),
            apply_postprocessing=False,
            **model_kwargs,
        )
        return SAM2AutomaticMaskGenerator(
            model,
            points_per_side=self._points_per_side,
            pred_iou_thresh=self._pred_iou_thresh,
            stability_score_thresh=self._stability_score_thresh,
            min_mask_region_area=self._min_mask_region_area,
        )

    def _build_predictor(self) -> Any:
        if not self._checkpoint.exists():
            self._download_checkpoint()

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as error:
            raise RuntimeError(
                "SAM2 is not installed. Install facebookresearch/sam2 in this "
                "environment before running SAM2 point prompts."
            ) from error

        model_kwargs: dict[str, str] = {}
        if self._device:
            model_kwargs["device"] = self._device
        model = build_sam2(
            self._config,
            str(self._checkpoint),
            apply_postprocessing=False,
            **model_kwargs,
        )
        return SAM2ImagePredictor(model)

    def _download_checkpoint(self) -> None:
        try:
            import requests
        except ImportError as error:
            raise RuntimeError(
                f"SAM2 checkpoint not found: {self._checkpoint}. Install requests or "
                "download the checkpoint manually."
            ) from error

        self._checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._checkpoint.with_suffix(f"{self._checkpoint.suffix}.part")
        with requests.get(self._checkpoint_url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with temporary_path.open("wb") as checkpoint_file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        checkpoint_file.write(chunk)
        temporary_path.replace(self._checkpoint)


class BBoxLabeler(tk.Tk):
    def __init__(
        self,
        *,
        image_path: Path,
        output_path: Path | None,
        prompt: str,
        device: str | None,
        model: str,
        sam2_checkpoint: Path,
        sam2_config: str,
    ) -> None:
        super().__init__()
        self.title("SAM bbox labeler")
        self.minsize(980, 680)

        self._image_path = image_path
        self._output_path = output_path
        self._prompt = prompt
        self._label = prompt
        self._model = model
        if model == "sam2":
            LOGGER.warning(
                "SAM2 does not support text prompts; --prompt is used only as "
                "the output label. Add positive/negative points or run automatic "
                "mask proposals."
            )
        self._status = tk.StringVar(value="Open an image or draw a box.")
        self._runner = self._build_runner(
            model=model,
            device=device,
            sam2_checkpoint=sam2_checkpoint,
            sam2_config=sam2_config,
        )
        self._proposal_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

        self._image: Image.Image | None = None
        self._photo: tk.PhotoImage | None = None
        self._image_size = (0, 0)
        self._scale = 1.0
        self._display_offset = (0, 0)
        self._annotations: list[BoxAnnotation] = []
        self._selected_index: int | None = None
        self._point_mode: int | None = None
        self._prompt_points: list[PromptPoint] = []
        self._drag_start: tuple[int, int] | None = None
        self._preview_id: int | None = None
        self._buttons: dict[str, tuple[int, int, int, int]] = {}

        self._build_ui()
        self._load_image(image_path)
        self.after(100, self._poll_proposal_queue)

    @staticmethod
    def _build_runner(
        *,
        model: str,
        device: str | None,
        sam2_checkpoint: Path,
        sam2_config: str,
    ) -> Any:
        if model == "sam2":
            return Sam2Runner(
                checkpoint=sam2_checkpoint,
                config=sam2_config,
                device=device,
            )
        return Sam3Runner(device=device)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._canvas = tk.Canvas(self, background="#1f1f1f")
        self._canvas.grid(row=0, column=0, sticky="nsew")
        self._canvas.bind("<ButtonPress-1>", self._start_box)
        self._canvas.bind("<B1-Motion>", self._drag_box)
        self._canvas.bind("<ButtonRelease-1>", self._finish_box)
        self._canvas.bind("<Configure>", lambda _event: self._redraw())
        self.bind("s", lambda _event: self._save())
        self.bind("r", lambda _event: self._run_model())
        self.bind("d", lambda _event: self._delete_selected())
        self.bind("c", lambda _event: self._clear_boxes())
        self.bind("b", lambda _event: self._set_box_mode())
        self.bind("p", lambda _event: self._set_point_mode(1))
        self.bind("n", lambda _event: self._set_point_mode(0))
        self.bind("u", lambda _event: self._undo_point())

    def _open_image(self) -> None:
        initial_dir = self._image_path.parent if self._image_path else DEFAULT_IMAGE_DIR
        chosen = filedialog.askopenfilename(
            initialdir=initial_dir,
            filetypes=(("Images", "*.bmp *.jpeg *.jpg *.png *.tif *.tiff *.webp"),),
        )
        if chosen:
            self._load_image(Path(chosen))

    def _load_image(self, image_path: Path) -> None:
        self._image_path = image_path
        self._image = Image.open(image_path).convert("RGB")
        self._image_size = self._image.size
        self._annotations = []
        self._selected_index = None
        self._set_status(
            f"Loaded {image_path.name} ({self._image_size[0]}x{self._image_size[1]})."
        )
        self._redraw()

    def _redraw(self) -> None:
        if self._image is None:
            return

        canvas_width = max(1, self._canvas.winfo_width())
        canvas_height = max(1, self._canvas.winfo_height())
        image_area_width = max(1, canvas_width - SIDE_PANEL_WIDTH)
        image_area_height = max(1, canvas_height - TOOLBAR_HEIGHT)
        image_width, image_height = self._image_size
        self._scale = min(
            image_area_width / image_width, image_area_height / image_height
        )
        display_size = (
            max(1, int(image_width * self._scale)),
            max(1, int(image_height * self._scale)),
        )
        self._display_offset = (
            (image_area_width - display_size[0]) // 2,
            TOOLBAR_HEIGHT + (image_area_height - display_size[1]) // 2,
        )
        display_image = self._image.resize(display_size)
        self._photo = tk_photo_from_image(display_image)

        self._canvas.delete("all")
        self._draw_shell(canvas_width, canvas_height)
        self._canvas.create_image(
            self._display_offset[0],
            self._display_offset[1],
            anchor=tk.NW,
            image=self._photo,
        )
        for index, annotation in enumerate(self._annotations):
            self._draw_annotation(index, annotation)
        self._draw_prompt_points()
        self._refresh_list()

    def _draw_shell(self, canvas_width: int, canvas_height: int) -> None:
        side_x = max(0, canvas_width - SIDE_PANEL_WIDTH)
        self._canvas.create_rectangle(
            0, 0, canvas_width, TOOLBAR_HEIGHT, fill="#303030", outline=""
        )
        self._canvas.create_rectangle(
            side_x,
            TOOLBAR_HEIGHT,
            canvas_width,
            canvas_height,
            fill="#262626",
            outline="",
        )
        self._buttons = {}
        x = 10
        for name, label in (
            ("open", "Open"),
            ("save", "Save"),
            ("run", f"Run {self._model.upper()}"),
            ("box_mode", "Box"),
            ("positive", "+ Point"),
            ("negative", "- Point"),
            ("undo_point", "Undo Pt"),
            ("delete", "Delete"),
            ("clear", "Clear"),
        ):
            width = 74
            if name in {"run", "undo_point"}:
                width = 104
            if name in {"positive", "negative"}:
                width = 94
            self._buttons[name] = (x, 8, x + width, 36)
            fill = self._button_fill(name)
            self._canvas.create_rectangle(
                x, 8, x + width, 36, fill=fill, outline="#777777"
            )
            self._canvas.create_text(x + width / 2, 22, text=label, fill="#ffffff")
            x += width + 8
        self._canvas.create_text(
            x + 10,
            22,
            anchor=tk.W,
            text=f'Model: {self._model.upper()}  Label: "{self._prompt}"',
            fill="#ffffff",
        )
        self._canvas.create_text(
            10,
            canvas_height - 34,
            anchor=tk.W,
            text=self._point_status(),
            fill="#ffffff",
        )
        self._canvas.create_text(
            side_x + 14,
            TOOLBAR_HEIGHT + 20,
            anchor=tk.W,
            text="Boxes",
            fill="#ffffff",
        )
        self._canvas.create_text(
            10,
            canvas_height - 14,
            anchor=tk.W,
            text=self._status.get(),
            fill="#ffffff",
        )

    def _button_fill(self, name: str) -> str:
        if name == "box_mode" and self._point_mode is None:
            return "#5a4b12"
        if name == "positive" and self._point_mode == 1:
            return "#165c32"
        if name == "negative" and self._point_mode == 0:
            return "#6a2020"
        return "#444444"

    def _point_status(self) -> str:
        positives = sum(point.label == 1 for point in self._prompt_points)
        negatives = sum(point.label == 0 for point in self._prompt_points)
        mode = "box"
        if self._point_mode == 1:
            mode = "positive point"
        if self._point_mode == 0:
            mode = "negative point"
        return (
            f"Mode: {mode}. Points: +{positives} -{negatives}. "
            "Keys: b box, p positive, n negative, u undo, r run, s save."
        )

    def _draw_prompt_points(self) -> None:
        for point in self._prompt_points:
            x, y = self._image_to_canvas_point((point.x, point.y))
            color = "#27d267" if point.label == 1 else "#ff6666"
            self._canvas.create_oval(
                x - 6,
                y - 6,
                x + 6,
                y + 6,
                fill=color,
                outline="#ffffff",
                width=2,
            )
            marker = "+" if point.label == 1 else "-"
            self._canvas.create_text(x, y, text=marker, fill="#111111")

    def _draw_annotation(self, index: int, annotation: BoxAnnotation) -> None:
        x1, y1, x2, y2 = self._image_to_canvas_box(annotation.box_xyxy)
        color = "#ffd000" if index == self._selected_index else "#00d084"
        if annotation.source == "sam3":
            color = "#00b7ff" if index != self._selected_index else "#ffd000"
        if annotation.source == "sam2-points":
            color = "#ff66cc" if index != self._selected_index else "#ffd000"
        self._canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=3)
        label = annotation.label
        if annotation.confidence is not None:
            label = f"{label} {annotation.confidence:.3f}"
        text_id = self._canvas.create_text(
            x1 + 5,
            max(self._display_offset[1] + 12, y1 - 10),
            anchor=tk.W,
            text=label,
            fill="#111111",
        )
        bbox = self._canvas.bbox(text_id)
        if bbox:
            self._canvas.create_rectangle(
                bbox[0] - 4,
                bbox[1] - 2,
                bbox[2] + 4,
                bbox[3] + 2,
                fill=color,
                outline=color,
            )
            self._canvas.tag_raise(text_id)

    def _refresh_list(self) -> None:
        canvas_width = max(1, self._canvas.winfo_width())
        side_x = max(0, canvas_width - SIDE_PANEL_WIDTH)
        for index, annotation in enumerate(self._annotations, start=1):
            x1, y1, x2, y2 = annotation.box_xyxy
            score = ""
            if annotation.confidence is not None:
                score = f" score={annotation.confidence:.3f}"
            y = TOOLBAR_HEIGHT + 44 + (index - 1) * 34
            fill = "#3a3a3a" if index - 1 != self._selected_index else "#5a4b12"
            self._canvas.create_rectangle(
                side_x + 10,
                y - 13,
                canvas_width - 10,
                y + 15,
                fill=fill,
                outline="#555555",
            )
            self._canvas.create_text(
                side_x + 18,
                y,
                anchor=tk.W,
                text=(
                    f"{index}. {annotation.label} [{annotation.source}]{score} "
                    f"({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
                ),
                fill="#ffffff",
            )

    def _start_box(self, event: tk.Event[tk.Canvas]) -> None:
        if self._handle_toolbar_click(event.x, event.y):
            self._drag_start = None
            return
        if self._handle_side_click(event.x, event.y):
            self._drag_start = None
            return
        point = self._canvas_to_image_point(event.x, event.y)
        if point is None:
            self._drag_start = None
            return
        if self._point_mode is not None:
            self._prompt_points.append(
                PromptPoint(
                    x=float(point[0]), y=float(point[1]), label=self._point_mode
                )
            )
            label = "positive" if self._point_mode == 1 else "negative"
            self._set_status(f"Added {label} point at ({point[0]}, {point[1]}).")
            self._redraw()
            return
        self._drag_start = point
        if self._preview_id is not None:
            self._canvas.delete(self._preview_id)
            self._preview_id = None

    def _drag_box(self, event: tk.Event[tk.Canvas]) -> None:
        if self._drag_start is None:
            return
        current = self._canvas_to_image_point(event.x, event.y)
        if current is None:
            return
        if self._preview_id is not None:
            self._canvas.delete(self._preview_id)
        box = self._normalized_box(self._drag_start, current)
        self._preview_id = self._canvas.create_rectangle(
            *self._image_to_canvas_box(box),
            outline="#ff6666",
            dash=(6, 4),
            width=2,
        )

    def _finish_box(self, event: tk.Event[tk.Canvas]) -> None:
        if self._drag_start is None:
            return
        current = self._canvas_to_image_point(event.x, event.y)
        if self._preview_id is not None:
            self._canvas.delete(self._preview_id)
            self._preview_id = None
        start = self._drag_start
        self._drag_start = None
        if current is None:
            return
        box = self._normalized_box(start, current)
        if box[2] - box[0] < 3 or box[3] - box[1] < 3:
            return
        label = self._label.strip() or "object"
        self._annotations.append(BoxAnnotation(label=label, box_xyxy=box))
        self._selected_index = len(self._annotations) - 1
        self._set_status(f"Added manual box for {label}.")
        self._redraw()

    def _delete_selected(self) -> None:
        if self._selected_index is None:
            return
        del self._annotations[self._selected_index]
        self._selected_index = None
        self._set_status("Deleted selected box.")
        self._redraw()

    def _clear_boxes(self) -> None:
        self._annotations = []
        self._selected_index = None
        self._set_status("Cleared all boxes.")
        self._redraw()

    def _set_box_mode(self) -> None:
        self._point_mode = None
        self._set_status("Box mode. Drag on the image to add a manual bbox.")
        self._redraw()

    def _set_point_mode(self, label: int) -> None:
        self._point_mode = label
        point_label = "positive" if label == 1 else "negative"
        self._set_status(
            f"{point_label.title()} point mode. Click image to add prompt points."
        )
        self._redraw()

    def _undo_point(self) -> None:
        if self._prompt_points:
            self._prompt_points.pop()
            self._set_status("Removed last prompt point.")
            self._redraw()

    def _run_model(self) -> None:
        prompt = self._prompt.strip()
        if not prompt:
            self._set_status("Missing prompt label.")
            return
        image_path = self._image_path
        prompt_points = list(self._prompt_points)
        self._set_status(f"Running {self._model.upper()} for '{prompt}'...")
        threading.Thread(
            target=self._run_model_worker,
            args=(image_path, prompt, prompt_points),
            daemon=True,
        ).start()

    def _run_model_worker(
        self, image_path: Path, prompt: str, prompt_points: list[PromptPoint]
    ) -> None:
        started = time.perf_counter()
        try:
            self._log_model_execution(image_path, prompt, prompt_points)
            annotations = self._runner.predict(image_path, prompt, prompt_points)
        except Exception as error:
            self._proposal_queue.put(("error", str(error)))
            return
        self._proposal_queue.put(
            ("result", (prompt, annotations, time.perf_counter() - started))
        )

    def _log_model_execution(
        self, image_path: Path, prompt: str, prompt_points: list[PromptPoint]
    ) -> None:
        positives = sum(point.label == 1 for point in prompt_points)
        negatives = sum(point.label == 0 for point in prompt_points)
        LOGGER.info(
            "Executing %s on %s with label=%r point_prompts=%s "
            "positive_points=%d negative_points=%d",
            self._model.upper(),
            image_path,
            prompt,
            json.dumps(prompt_points_to_payload(prompt_points), separators=(",", ":")),
            positives,
            negatives,
        )

    def _poll_proposal_queue(self) -> None:
        try:
            kind, payload = self._proposal_queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_proposal_queue)
            return

        if kind == "error":
            self._set_status(f"{self._model.upper()} failed.")
            print(f"{self._model.upper()} error: {payload}")
        else:
            prompt, annotations, elapsed = payload
            self._annotations.extend(annotations)
            self._selected_index = len(self._annotations) - 1 if annotations else None
            self._set_status(
                f"{self._model.upper()} added {len(annotations)} boxes for "
                f"'{prompt}' in {elapsed:.1f}s."
            )
            self._redraw()
        self.after(100, self._poll_proposal_queue)

    def _save(self) -> None:
        output_path = self._output_path
        if output_path is None:
            default_name = f"{self._image_path.stem}.bboxes.json"
            chosen = filedialog.asksaveasfilename(
                initialdir=self._image_path.parent,
                initialfile=default_name,
                defaultextension=".json",
                filetypes=(("JSON", "*.json"),),
            )
            if not chosen:
                return
            output_path = Path(chosen)
            self._output_path = output_path
        save_annotations(
            image_path=self._image_path,
            image_size=self._image_size,
            annotations=self._annotations,
            output_path=output_path,
        )
        self._set_status(f"Saved {len(self._annotations)} boxes to {output_path}.")
        self._redraw()

    def _set_status(self, value: str) -> None:
        self._status.set(value)

    def _canvas_to_image_point(self, x: int, y: int) -> tuple[int, int] | None:
        offset_x, offset_y = self._display_offset
        image_width, image_height = self._image_size
        image_x = int((x - offset_x) / self._scale)
        image_y = int((y - offset_y) / self._scale)
        if (
            image_x < 0
            or image_y < 0
            or image_x >= image_width
            or image_y >= image_height
        ):
            return None
        return image_x, image_y

    def _handle_toolbar_click(self, x: int, y: int) -> bool:
        if y > TOOLBAR_HEIGHT:
            return False
        for name, (x1, y1, x2, y2) in self._buttons.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                {
                    "open": self._open_image,
                    "save": self._save,
                    "run": self._run_model,
                    "box_mode": self._set_box_mode,
                    "positive": lambda: self._set_point_mode(1),
                    "negative": lambda: self._set_point_mode(0),
                    "undo_point": self._undo_point,
                    "delete": self._delete_selected,
                    "clear": self._clear_boxes,
                }[name]()
                return True
        return True

    def _handle_side_click(self, x: int, y: int) -> bool:
        canvas_width = max(1, self._canvas.winfo_width())
        side_x = max(0, canvas_width - SIDE_PANEL_WIDTH)
        if x < side_x or y < TOOLBAR_HEIGHT:
            return False
        index = int((y - (TOOLBAR_HEIGHT + 31)) // 34)
        if 0 <= index < len(self._annotations):
            self._selected_index = index
            self._redraw()
        return True

    def _image_to_canvas_box(
        self, box: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        offset_x, offset_y = self._display_offset
        x1, y1, x2, y2 = box
        return (
            offset_x + x1 * self._scale,
            offset_y + y1 * self._scale,
            offset_x + x2 * self._scale,
            offset_y + y2 * self._scale,
        )

    def _image_to_canvas_point(self, point: tuple[float, float]) -> tuple[float, float]:
        offset_x, offset_y = self._display_offset
        x, y = point
        return offset_x + x * self._scale, offset_y + y * self._scale

    @staticmethod
    def _normalized_box(
        start: tuple[int, int], current: tuple[int, int]
    ) -> tuple[float, float, float, float]:
        x1, y1 = start
        x2, y2 = current
        return (
            float(min(x1, x2)),
            float(min(y1, y2)),
            float(max(x1, x2)),
            float(max(y1, y2)),
        )


def find_default_image(image_dir: Path) -> Path:
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            return path
    raise FileNotFoundError(f"No images found in {image_dir}")


def tk_photo_from_image(image: Image.Image) -> tk.PhotoImage:
    buffer = io.BytesIO()
    image.save(buffer, format="PPM")
    return tk.PhotoImage(data=buffer.getvalue(), format="PPM")


def _image_to_numpy(image: Image.Image) -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("SAM2 auto-detection requires numpy.") from error
    return np.array(image)


def _points_to_numpy(points: list[PromptPoint]) -> tuple[Any, Any]:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("SAM2 point prompts require numpy.") from error
    point_coords = np.array([(point.x, point.y) for point in points], dtype=np.float32)
    point_labels = np.array([point.label for point in points], dtype=np.int32)
    return point_coords, point_labels


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


if __name__ == "__main__":
    main()
