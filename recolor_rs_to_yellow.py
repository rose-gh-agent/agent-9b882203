#!/usr/bin/env python3
from pathlib import Path

import numpy as np
from PIL import Image


INPUT_PATH = Path("/home/user/attachments/rs.jpeg")
OUTPUT_PATH = Path("/home/user/output/rs_yellow.jpeg")


def main() -> None:
    img = Image.open(INPUT_PATH).convert("RGB")
    data = np.array(img, dtype=np.float32) / 255.0

    r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    s = np.zeros_like(maxc)
    nonzero_max = maxc != 0
    s[nonzero_max] = (maxc[nonzero_max] - minc[nonzero_max]) / maxc[nonzero_max]

    delta = maxc - minc
    h = np.zeros_like(maxc)

    mask = delta != 0
    rm = mask & (maxc == r)
    h[rm] = ((g[rm] - b[rm]) / delta[rm]) % 6
    gm = mask & (maxc == g)
    h[gm] = (b[gm] - r[gm]) / delta[gm] + 2
    bm = mask & (maxc == b)
    h[bm] = (r[bm] - g[bm]) / delta[bm] + 4
    h = h / 6.0

    non_bg = (s > 0.05) & (v > 0.05)
    h[non_bg] = 0.167

    h6 = h * 6.0
    i = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    result = np.zeros_like(data)
    for idx, (ri, gi, bi) in enumerate(
        [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    ):
        mask_i = i == idx
        result[:, :, 0][mask_i] = ri[mask_i]
        result[:, :, 1][mask_i] = gi[mask_i]
        result[:, :, 2][mask_i] = bi[mask_i]

    result[~non_bg] = data[~non_bg]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_img = Image.fromarray((result * 255).astype(np.uint8), "RGB")
    result_img.save(OUTPUT_PATH, quality=95)
    print(f"Done! Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
