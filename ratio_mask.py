"""Geometric padding masks; independent of image pixel colors."""

import math

import torch


RATIO_PRESETS = ["16:9", "3:2", "4:3", "1:1", "9:16", "2:3", "21:9", "Custom"]
ALIGNMENTS = [
    "Top", "Center", "Bottom", "Left", "Right",
    "Top Left", "Top Right", "Bottom Left", "Bottom Right",
]


def content_bounds(width, height, ratio_width, ratio_height, alignment="Top"):
    """Fit the original aspect ratio inside the input canvas, without cropping.

    Returns (x, y, content_width, content_height). Round to the nearest pixel
    (half up); never align to model multiples, which would shift mask edges.
    """
    if width < 1 or height < 1:
        raise ValueError("Image dimensions must be positive.")
    if not all(math.isfinite(v) and v > 0 for v in (ratio_width, ratio_height)):
        raise ValueError("Original aspect ratio dimensions must be positive and finite.")
    if alignment not in ALIGNMENTS:
        raise ValueError(f"Unknown content alignment: {alignment}")

    if width * ratio_height <= height * ratio_width:
        content_width = width
        content_height = min(height, max(1, math.floor(width * ratio_height / ratio_width + 0.5)))
    else:
        content_height = height
        content_width = min(width, max(1, math.floor(height * ratio_width / ratio_height + 0.5)))

    gap_x, gap_y = width - content_width, height - content_height
    x = 0 if "Left" in alignment else gap_x if "Right" in alignment else gap_x // 2
    y = 0 if "Top" in alignment else gap_y if "Bottom" in alignment else gap_y // 2
    return x, y, content_width, content_height


def create_ratio_padding_mask(batch, height, width, ratio_width, ratio_height,
                              alignment="Top", device=None):
    """Return float32 [B,H,W]: padding=1, original content=0."""
    x, y, content_width, content_height = content_bounds(
        width, height, ratio_width, ratio_height, alignment
    )
    masks = torch.ones((batch, height, width), dtype=torch.float32, device=device)
    masks[:, y:y + content_height, x:x + content_width] = 0.0
    return masks
