from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .annotations import (
    BoxAnnotation,
    PromptPoint,
    coerce_sam2_masks,
    coerce_sam2_prompt_masks,
    coerce_sam3_boxes,
    group_prompt_points,
)
from .constants import SAM2_TINY_CHECKPOINT_URL


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
        annotations: list[BoxAnnotation] = []
        for point_group in group_prompt_points(points):
            point_coords, point_labels = _points_to_numpy(point_group)
            masks, scores, _logits = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
            annotations.extend(coerce_sam2_prompt_masks(masks, scores, prompt))
        return annotations

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
