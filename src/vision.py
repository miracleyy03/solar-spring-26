from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def find_sun_blob(gray: np.ndarray,
                  percentile: float = 99.0,
                  blur_ksize: int = 5,
                  min_thresh: int = 200,
                  ) -> Optional[Tuple[int, int, int]]:
    if gray is None or gray.size == 0:
        return None

    blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    thresh_value = float(np.percentile(blur, percentile))
    if thresh_value < min_thresh:
        return None
    _, mask = cv2.threshold(blur, thresh_value, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    sun_contour = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(sun_contour)
    return int(x), int(y), int(radius)


def isCloudy(image_bgr: np.ndarray) -> bool:
    hls = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HLS)
    lightness = cv2.extractChannel(hls, 1)
    mean_intensity = float(cv2.mean(lightness)[0])
    std_dev = float(np.std(lightness))
    max_lightness = float(np.max(lightness))

    if max_lightness >= 230:
        return False
    if max_lightness >= 200 and std_dev >= 30:
        return True
    if mean_intensity < 100:
        return True
    if mean_intensity < 175:
        return std_dev < 15
    return False


def annotate(frame_bgr: np.ndarray,
             blob: Optional[Tuple[int, int, int]]) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    cv2.drawMarker(out, (w // 2, h // 2), (0, 255, 0),
                   markerType=cv2.MARKER_CROSS,
                   markerSize=20, thickness=2)
    if blob is not None:
        x, y, r = blob
        cv2.circle(out, (x, y), max(r, 4), (0, 0, 255), 2)
        cv2.circle(out, (x, y), 4, (255, 0, 0), -1)
    return out
