from __future__ import annotations

from pathlib import Path


DEFAULT_IMAGE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
DEFAULT_SAM2_CHECKPOINT = (
    Path(__file__).resolve().parents[2] / "checkpoints" / "sam2.1_hiera_tiny.pt"
)
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_TINY_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
)
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
TOOLBAR_HEIGHT = 46
SIDE_PANEL_WIDTH = 330
