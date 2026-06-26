from __future__ import annotations

import colorsys
from pathlib import Path

from PIL import Image


SOURCE = Path("/home/user/attachments/rs.jpeg")
DESTINATION = Path("/home/user/output/rs_blue.png")

# Roughly 229 degrees on the HSV color wheel.
TARGET_HUE = 229 / 360


def recolor_pixel(red: int, green: int, blue: int) -> tuple[int, int, int]:
    r, g, b = red / 255.0, green / 255.0, blue / 255.0
    hue, saturation, value = colorsys.rgb_to_hsv(r, g, b)

    # Preserve the black background and deep shadows.
    if value < 0.08:
        return 0, 0, 0

    # Keep metallic highlights readable while pushing colored regions blue.
    new_saturation = min(1.0, max(saturation, 0.35))
    nr, ng, nb = colorsys.hsv_to_rgb(TARGET_HUE, new_saturation, value)
    return round(nr * 255), round(ng * 255), round(nb * 255)


def main() -> None:
    image = Image.open(SOURCE).convert("RGB")
    result = Image.new("RGB", image.size)
    source_pixels = image.load()
    result_pixels = result.load()

    for y in range(image.height):
        for x in range(image.width):
            result_pixels[x, y] = recolor_pixel(*source_pixels[x, y])

    result.save(DESTINATION, format="PNG")


if __name__ == "__main__":
    main()
