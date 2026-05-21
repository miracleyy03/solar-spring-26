from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import actuators
from src import sun_ephemeris as eph_mod
from src import controller as ctrl_mod
from src.controller import EphemerisAwareController
from src.exposure_utils import adaptive_exposure_drc
from src.sun_ephemeris import SiteConfig, SunPosition, get_sun_position as real_get_sun_position
from src.vision import find_sun_blob, isCloudy

IMAGES_DIR = ROOT / "images"

DEMO_SITE = SiteConfig(
    latitude_deg=42.444,
    longitude_deg=-76.502,
    elevation_m=250.0,
    mount_home_azimuth_deg=180.0,
    mount_home_elevation_deg=45.0,
    mount_azimuth_range_deg=60.0,
    mount_elevation_range_deg=45.0,
)

FIXED_WHEN_UTC = datetime(2026, 6, 21, 17, 0, tzinfo=timezone.utc)

CURATED_IMAGES: List[Tuple[str, str]] = [
    ("sample_sun3.jpg",  "clear sky, sun off-center"),
    ("sample_sun5.jpg",  "clear sky, different sun location"),
    ("cloudy1.jpg",      "overcast sky"),
    ("no_sun1.jpg",      "dark / no sun"),
    ("sample_sun1.jpeg", "clear sky, sun near edge"),
]

DEMO_FRAME_W, DEMO_FRAME_H = 640, 480

_actuator_log: List[Tuple[int, str, float]] = []


def install_recording_drive() -> None:
    def recording_drive(idx: int, hi: int, ret: int, dur: float) -> None:
        if dur <= 0:
            return
        if hi and ret:
            print(f"    [actuators] WARNING: actuator {idx} extend+retract both active -- skipping")
            return
        if hi:
            dur = min(dur, max(0.0, actuators.FULL_LENGTH - actuators.position[idx]))
        if ret:
            dur = min(dur, max(0.0, actuators.position[idx]))
        if dur <= 0:
            return
        action = "EXTEND " if hi else "RETRACT"
        _actuator_log.append((idx, action, dur))
        if hi:
            actuators.position[idx] = min(actuators.FULL_LENGTH, actuators.position[idx] + dur)
        if ret:
            actuators.position[idx] = max(0.0, actuators.position[idx] - dur)

    actuators._drive = recording_drive


def install_fixed_ephemeris() -> None:
    def fixed_get_sun_position(site: SiteConfig, when=None) -> SunPosition:
        return real_get_sun_position(site, FIXED_WHEN_UTC)

    eph_mod.get_sun_position = fixed_get_sun_position
    ctrl_mod.get_sun_position = fixed_get_sun_position


def clear_log() -> None:
    _actuator_log.clear()


def load_image(name: str):
    path = IMAGES_DIR / name
    if not path.exists():
        return None, path
    img = cv2.imread(str(path))
    if img is None:
        return None, path
    img = cv2.resize(img, (DEMO_FRAME_W, DEMO_FRAME_H))
    return img, path


def format_actuator_block() -> List[str]:
    rolled: dict = {1: [], 2: [], 3: [], 4: []}
    for idx, action, dur in _actuator_log:
        rolled[idx].append((action, dur))
    lines = []
    for idx in (1, 2, 3, 4):
        entries = rolled[idx]
        if not entries:
            lines.append(f"    Actuator {idx}:  (no motion)")
            continue
        for action, dur in entries:
            lines.append(f"    Actuator {idx}:  {action}  {dur:.3f} s")
    return lines


def process_image(ctrl: EphemerisAwareController, name: str, description: str) -> None:
    print(f"\n=== {name} -- {description} ===")
    frame, path = load_image(name)
    if frame is None:
        print(f"  (skipped: could not load {path})")
        return
    print(f"  loaded {path.name}  shape={frame.shape}")

    working_frame, metrics, exp_scale = adaptive_exposure_drc(frame)
    print(f"  exposure: mean_v={metrics['mean_v']:.1f}  max_v={metrics['max_v']:.1f}  "
          f"contrast={metrics['contrast_v']:.1f}  -> scale={exp_scale:.3f}")

    sun = real_get_sun_position(DEMO_SITE, FIXED_WHEN_UTC)
    print(f"  ephemeris ({FIXED_WHEN_UTC.isoformat()}): "
          f"az={sun.azimuth_deg:.2f} deg, el={sun.elevation_deg:.2f} deg")

    clear_log()
    cloudy = isCloudy(working_frame)
    if cloudy:
        print(f"  cloud check: CLOUDY  ->  open-loop fallback")
        state, info = ctrl.step_open_loop()
    else:
        print(f"  cloud check: clear  ->  closed-loop vision")
        gray = cv2.cvtColor(working_frame, cv2.COLOR_BGR2GRAY)
        blob = find_sun_blob(gray)
        if blob is None:
            print(f"  sun blob: NOT FOUND  ->  open-loop fallback")
            state, info = ctrl.step_open_loop()
        else:
            x, y, r = blob
            print(f"  sun blob: x={x}  y={y}  radius={r} px")
            state, info = ctrl.step_closed_loop(gray)

    print(f"  controller state: {state}")
    if "nx" in info and "ny" in info:
        print(f"  normalized command: nx={info['nx']:+.3f}  ny={info['ny']:+.3f}")
    if "ang_err_deg" in info:
        print(f"  angular error vs ephemeris: {info['ang_err_deg']:.2f} deg")
    if "reason" in info:
        print(f"  reason: {info['reason']}")

    print(f"  actuator commands issued:")
    for line in format_actuator_block():
        print(line)

    print(f"  mount pointing now: az={ctrl.mount_az_deg:.2f} deg, "
          f"el={ctrl.mount_el_deg:.2f} deg")


def print_final_positions() -> None:
    print(f"\n=== Final actuator positions ===")
    print(f"  (units: seconds-of-extension; mechanical end-stop = {actuators.FULL_LENGTH:.1f} s)")
    for idx in (1, 2, 3, 4):
        pos = actuators.position[idx]
        bar_len = 20
        filled = int(round(bar_len * pos / actuators.FULL_LENGTH))
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"  Actuator {idx}:  [{bar}]  {pos:.3f} s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the solar tracker pipeline on sample images and "
                    "show how each of the four linear actuators would move. "
                    "Mocks the camera (uses images/), the Raspberry Pi GPIO "
                    "(stub), and the actuators (recording wrapper).")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--image", metavar="NAME",
                   help="Process a single image from images/ (e.g. sample_sun2.jpg)")
    g.add_argument("--all", action="store_true",
                   help="Process every image found in images/")
    return p.parse_args()


def select_images(args: argparse.Namespace) -> List[Tuple[str, str]]:
    if args.image:
        return [(args.image, "user-specified image")]
    if args.all:
        items = []
        for path in sorted(IMAGES_DIR.iterdir()):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                items.append((path.name, "image from images/"))
        return items
    return CURATED_IMAGES


def main() -> int:
    args = parse_args()

    print("=" * 70)
    print(" solar_tracker DEMO -- no camera, no Raspberry Pi, no actuators")
    print("=" * 70)
    print(f" site:        lat={DEMO_SITE.latitude_deg}  lon={DEMO_SITE.longitude_deg}  "
          f"(home az={DEMO_SITE.mount_home_azimuth_deg} el={DEMO_SITE.mount_home_elevation_deg})")
    print(f" mock time:   {FIXED_WHEN_UTC.isoformat()}  (Cornell solar noon, deterministic)")
    print(f" plausibility tolerance: 180 deg (relaxed so test images pass)")
    print(f" images dir:  {IMAGES_DIR}")
    print(f"")
    print(" Stand-ins for missing hardware:")
    print("   - Camera         : cv2.imread() of each image in images/")
    print("   - Raspberry Pi   : src.actuators._StubGPIO (auto-active on non-Pi)")
    print("   - Actuators      : recording wrapper around src.actuators._drive")

    actuators.setup_gpio()
    install_recording_drive()
    install_fixed_ephemeris()

    ctrl = EphemerisAwareController(
        DEMO_SITE,
        sanity_tolerance_deg=180.0,
    )

    images = select_images(args)
    if not images:
        print("\nNo images to process.")
        actuators.cleanup_gpio()
        return 1

    start = time.time()
    try:
        for name, description in images:
            process_image(ctrl, name, description)
    finally:
        print_final_positions()
        actuators.cleanup_gpio()

    print(f"\nDemo finished in {time.time() - start:.2f} s. Processed {len(images)} image(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
