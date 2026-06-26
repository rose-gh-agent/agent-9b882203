#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


INPUT_PATH = Path("/home/user/attachments/rs.jpeg")
OUTPUT_PATH = Path("/home/user/output/rs_yellow.jpeg")


def main() -> None:
    img = cv2.imread(str(INPUT_PATH), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load input image: {INPUT_PATH}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Keep black / near-black / neutral regions unchanged.
    active = (s > 30.0) & (v > 25.0)

    # Target the teal/cyan metallic band while avoiding other colors.
    teal_mask = active & (h >= 42.0) & (h <= 100.0)

    # Create a soft mask so the recolor blends cleanly at the edges.
    soft_mask = teal_mask.astype(np.float32)
    soft_mask = cv2.GaussianBlur(soft_mask, (0, 0), sigmaX=1.2, sigmaY=1.2)
    soft_mask = np.clip(soft_mask, 0.0, 1.0)

    target_hue = 30.0  # OpenCV hue for warm yellow/gold
    new_h = h.copy()
    new_h[teal_mask] = target_hue
    feathered = (soft_mask > 0.0) & (~teal_mask)
    new_h[feathered] = ((1.0 - soft_mask[feathered]) * h[feathered]) + (
        soft_mask[feathered] * target_hue
    )

    # Slight saturation lift helps read as gold while preserving value/contrast.
    sat_boost = 1.0 + (0.28 * soft_mask)
    new_s = np.clip(s * sat_boost, 0.0, 255.0)

    hsv[:, :, 0] = np.mod(new_h, 180.0)
    hsv[:, :, 1] = new_s
    hsv[:, :, 2] = v

    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(OUTPUT_PATH), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError(f"Failed to write output image: {OUTPUT_PATH}")

    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
