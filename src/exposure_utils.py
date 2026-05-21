import cv2
import numpy as np

def analyze_brightness(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    mean_v = float(np.mean(v))
    max_v = float(np.max(v))
    contrast_v = float(np.std(v))

    return {
        "mean_v": mean_v,
        "max_v": max_v,
        "contrast_v": contrast_v,
    }


def adaptive_exposure_drc(frame, target_mean=140.0,
                          clip_limit=2.0, tile_grid_size=(8, 8)):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)


    metrics = analyze_brightness(frame)
    mean_v = metrics["mean_v"]


    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=tile_grid_size)
    v_eq = clahe.apply(v)

    hsv_eq = cv2.merge([h, s, v_eq])
    adjusted_frame = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)


    exposure_scale = target_mean / (mean_v + 1e-6)
    exposure_scale = float(np.clip(exposure_scale, 0.5, 2.0))

    return adjusted_frame, metrics, exposure_scale
