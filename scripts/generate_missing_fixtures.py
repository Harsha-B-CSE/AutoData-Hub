"""Generate the synthetic golden night-scene fixture, not a photograph of Mysore Palace."""

from pathlib import Path

import cv2
import numpy as np


def generate_fixtures() -> None:
    fixture_dir = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
    target = fixture_dir / "Mysore_palace_illuminated.jpg"
    if target.is_file():
        print(f"Leaving existing fixture unchanged: {target}")
        return

    y, x = np.ogrid[:480, :640]
    radius = np.hypot(x - 320, y - 240)
    glow = np.maximum(1.0 - radius / 240, 0.0) ** 2
    glow = cv2.GaussianBlur(glow, (9, 9), 2.0)
    glow /= glow[240, 320]
    image = np.rint(glow[:, :, None] * (0, 215, 255)).astype(np.uint8)

    # Keep the glow smooth, but give the night scene focused edges with a lit arcade.
    trim = (8, 148, 204)
    cv2.rectangle(image, (132, 313), (508, 356), (2, 9, 16), -1)
    for cx in range(144, 497, 22):
        spring_y = 327 if abs(cx - 320) <= 44 else 337
        cv2.ellipse(image, (cx, spring_y), (8, 10), 0, 180, 360, trim, 1, cv2.LINE_AA)
        for jamb in (cx - 8, cx + 8):
            cv2.line(image, (jamb, spring_y), (jamb, 353), trim, 1)
        cv2.rectangle(image, (cx - 2, spring_y), (cx + 2, 351), (5, 69, 115), -1)

    cv2.line(image, (128, 356), (512, 356), trim, 1)
    for cx in range(128, 513, 6):
        cv2.circle(image, (cx, 356), 1, (18, 175, 230), -1)

    fixture_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError(f"Could not write synthetic fixture: {target}")
    print(f"Generated synthetic fixture: {target}")


if __name__ == "__main__":
    generate_fixtures()
