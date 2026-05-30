from __future__ import annotations

import io
import json
import logging
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import Any

from PIL import Image

from .annotations import (
    BoxAnnotation,
    PromptPoint,
    group_prompt_points,
    prompt_points_to_payload,
    save_annotations,
)
from .constants import DEFAULT_IMAGE_DIR, IMAGE_EXTENSIONS, SIDE_PANEL_WIDTH
from .constants import TOOLBAR_HEIGHT
from .runners import Sam2Runner, Sam3Runner


LOGGER = logging.getLogger(__name__)


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
        self._active_point_group = 1
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
        self.bind("o", lambda _event: self._next_point_group())
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
        self._prompt_points = []
        self._active_point_group = 1
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
            ("next_group", "Next"),
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
        group_count = len(group_prompt_points(self._prompt_points))
        mode = "box"
        if self._point_mode == 1:
            mode = "positive point"
        if self._point_mode == 0:
            mode = "negative point"
        return (
            f"Mode: {mode}. Object: {self._active_point_group}. "
            f"Groups: {group_count}. Points: +{positives} -{negatives}. "
            "Keys: b box, p positive, n negative, o next object, u undo, r run, s save."
        )

    def _draw_prompt_points(self) -> None:
        for point in self._prompt_points:
            x, y = self._image_to_canvas_point((point.x, point.y))
            color = self._point_color(point)
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
            self._canvas.create_text(
                x + 10,
                y - 10,
                text=str(point.group),
                fill="#ffffff",
            )

    @staticmethod
    def _point_color(point: PromptPoint) -> str:
        if point.label == 0:
            return "#ff6666"
        colors = ("#27d267", "#00b7ff", "#ff66cc", "#ffd000", "#c58cff", "#69e2d0")
        return colors[(point.group - 1) % len(colors)]

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
                    x=float(point[0]),
                    y=float(point[1]),
                    label=self._point_mode,
                    group=self._active_point_group,
                )
            )
            label = "positive" if self._point_mode == 1 else "negative"
            self._set_status(
                f"Added object {self._active_point_group} {label} point "
                f"at ({point[0]}, {point[1]})."
            )
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
            f"{point_label.title()} point mode for object "
            f"{self._active_point_group}. Click image to add prompt points."
        )
        self._redraw()

    def _next_point_group(self) -> None:
        self._active_point_group = self._next_unused_point_group()
        self._set_status(
            f"Object {self._active_point_group}. Add positive/negative points."
        )
        self._redraw()

    def _next_unused_point_group(self) -> int:
        if not self._prompt_points:
            return 1
        return max(point.group for point in self._prompt_points) + 1

    def _undo_point(self) -> None:
        if self._prompt_points:
            removed = self._prompt_points.pop()
            if not any(point.group == removed.group for point in self._prompt_points):
                self._active_point_group = max(1, self._next_unused_point_group() - 1)
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
            "point_groups=%d positive_points=%d negative_points=%d",
            self._model.upper(),
            image_path,
            prompt,
            json.dumps(prompt_points_to_payload(prompt_points), separators=(",", ":")),
            len(group_prompt_points(prompt_points)),
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
                    "next_group": self._next_point_group,
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
