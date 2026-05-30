from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PIL import Image

from two_d_screw_hole_od.config import GROUNDING_DINO_TARGET_PROMPTS
from two_d_screw_hole_od.types import Detection, ModelResult


def run_grounding_dino(
    image_path: Path,
    prompts: tuple[str, ...] = GROUNDING_DINO_TARGET_PROMPTS,
    confidence: float = 0.25,
    text_threshold: float = 0.25,
    device: str | None = None,
    model_id: str = "IDEA-Research/grounding-dino-tiny",
) -> ModelResult:
    started_at = time.perf_counter()

    try:
        import torch
        from transformers.image_transforms import center_to_corners_format
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        image = Image.open(image_path).convert("RGB")
        text = ". ".join(prompts) + "."

        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
            selected_device
        )
        inputs = processor(images=image, text=text, return_tensors="pt").to(
            selected_device
        )
        prompt_token_spans = _prompt_token_spans(
            input_ids=inputs.input_ids[0],
            prompts=prompts,
            tokenizer=processor.tokenizer,
        )

        inference_started_at = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
        inference_seconds = time.perf_counter() - inference_started_at

        target_sizes = [image.size[::-1]]
        processed = _post_process_grounding_dino_result(
            batch_boxes=center_to_corners_format(outputs.pred_boxes),
            batch_probs=torch.sigmoid(outputs.logits),
            target_sizes=target_sizes,
            threshold=confidence,
            prompt_token_spans=prompt_token_spans,
        )
        detections = _detections_from_grounding_dino_result(processed)
        return ModelResult(
            model="Grounding DINO",
            status="ok",
            confidence_threshold=confidence,
            elapsed_seconds=time.perf_counter() - started_at,
            inference_seconds=inference_seconds,
            detections=detections,
            text_threshold=text_threshold,
        )
    except Exception as exc:  # pragma: no cover - exercised by opt-in live runs.
        return ModelResult(
            model="Grounding DINO",
            status="error",
            confidence_threshold=confidence,
            elapsed_seconds=time.perf_counter() - started_at,
            inference_seconds=None,
            detections=[],
            text_threshold=text_threshold,
            notes=repr(exc),
        )


def _post_process_grounding_dino_result(
    batch_boxes: Any,
    batch_probs: Any,
    target_sizes: list[tuple[int, int]],
    threshold: float,
    prompt_token_spans: list[tuple[str, tuple[int, ...]]],
) -> dict[str, Any]:
    batch_scores = batch_probs.max(dim=-1)[0]

    if target_sizes is not None:
        import torch

        img_h = torch.Tensor([size[0] for size in target_sizes])
        img_w = torch.Tensor([size[1] for size in target_sizes])
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(
            batch_boxes.device
        )
        batch_boxes = batch_boxes * scale_fct[:, None, :]

    scores = batch_scores[0]
    boxes = batch_boxes[0]
    probs = batch_probs[0]
    keep = scores > threshold
    kept_scores = scores[keep]
    kept_boxes = boxes[keep]
    kept_probs = probs[keep]
    labels, text_matches = _best_prompt_matches(kept_probs, prompt_token_spans)

    return {
        "scores": kept_scores,
        "boxes": kept_boxes,
        "text_labels": labels,
        "text_matches": text_matches,
    }


def _prompt_token_spans(
    input_ids: Any,
    prompts: tuple[str, ...],
    tokenizer: Any,
) -> list[tuple[str, tuple[int, ...]]]:
    ids = [int(input_id) for input_id in input_ids.tolist()]
    spans: list[tuple[str, tuple[int, ...]]] = []
    cursor = 0

    for prompt in prompts:
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        span = _find_subsequence(ids, prompt_ids, start=cursor)
        if span is None:
            span = _find_subsequence(ids, prompt_ids, start=0)
        if span is None:
            continue
        spans.append((prompt, tuple(span)))
        cursor = span[-1] + 1
    return spans


def _find_subsequence(
    values: list[int],
    subsequence: list[int],
    start: int,
) -> list[int] | None:
    if not subsequence:
        return None

    stop = len(values) - len(subsequence) + 1
    for index in range(start, stop):
        if values[index : index + len(subsequence)] == subsequence:
            return list(range(index, index + len(subsequence)))
    return None


def _best_prompt_matches(
    probs: Any,
    prompt_token_spans: list[tuple[str, tuple[int, ...]]],
) -> tuple[list[str], list[float | None]]:
    labels: list[str] = []
    text_matches: list[float | None] = []
    for prob in probs:
        best_label = ""
        best_match: float | None = None
        for prompt, token_indexes in prompt_token_spans:
            token_probs = prob[list(token_indexes)]
            prompt_match = float(token_probs.mean().item())
            if best_match is None or prompt_match > best_match:
                best_label = prompt
                best_match = prompt_match
        labels.append(best_label)
        text_matches.append(best_match)
    return labels, text_matches


def _detections_from_grounding_dino_result(result: dict[str, Any]) -> list[Detection]:
    labels = result.get("text_labels", result.get("labels", []))
    scores = result.get("scores", [])
    boxes = result.get("boxes", [])
    text_matches = result.get("text_matches", [])

    detections: list[Detection] = []
    for label, score, box, text_match in zip(
        labels, scores, boxes, text_matches, strict=False
    ):
        detections.append(
            Detection(
                label=str(label),
                confidence=float(score.item() if hasattr(score, "item") else score),
                box_xyxy=_box_xyxy(box),
                text_match=text_match,
            )
        )
    return detections


def _box_xyxy(box: Any) -> tuple[float, float, float, float]:
    values = box.tolist() if hasattr(box, "tolist") else box
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))
