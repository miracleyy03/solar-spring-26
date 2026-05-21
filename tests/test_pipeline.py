import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import cv2


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vision import find_sun_blob, isCloudy
from src.exposure_utils import adaptive_exposure_drc, analyze_brightness
from src.sun_ephemeris import (
    SiteConfig, get_sun_position, get_sun_events_today,
    pixel_offset_to_sky_angles, vision_detection_is_plausible,
    open_loop_command, angular_separation_deg,
)
from src.controller import EphemerisAwareController
from src import actuators


def make_sunny_frame(w=640, h=480, sun_x=420, sun_y=180, radius=30):
    frame = np.full((h, w, 3), (120, 150, 180), dtype=np.uint8)
    cv2.circle(frame, (sun_x, sun_y), radius + 15, (210, 210, 210), -1)
    cv2.circle(frame, (sun_x, sun_y), radius + 8, (235, 235, 235), -1)
    cv2.circle(frame, (sun_x, sun_y), radius, (255, 255, 255), -1)
    cv2.GaussianBlur(frame, (5, 5), 0, dst=frame)
    return frame


def test_vision():
    print("=== vision ===")
    frame = make_sunny_frame()
    assert isCloudy(frame) is False, "synthetic sunny frame should not be cloudy"
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blob = find_sun_blob(gray)
    assert blob is not None, "expected to find the synthetic sun"
    x, y, r = blob
    assert abs(x - 420) < 15 and abs(y - 180) < 15, f"sun off-target: ({x},{y})"
    print(f"  found sun at ({x},{y}) radius={r}  OK")


def test_exposure():
    print("=== exposure ===")
    frame = make_sunny_frame()
    adj, metrics, scale = adaptive_exposure_drc(frame)
    assert adj.shape == frame.shape
    assert 0.5 <= scale <= 2.0
    assert metrics["mean_v"] > 0
    print(f"  metrics={metrics}, scale={scale:.3f}  OK")


def test_ephemeris():
    print("=== ephemeris ===")
    site = SiteConfig(latitude_deg=42.444, longitude_deg=-76.502)

    when = datetime(2026, 6, 21, 17, 0, tzinfo=timezone.utc)
    sun = get_sun_position(site, when)
    assert sun.is_above_horizon, f"sun should be up: el={sun.elevation_deg}"
    assert 100 < sun.azimuth_deg < 260, f"midday az implausible: {sun.azimuth_deg}"
    events = get_sun_events_today(site, when)
    assert events["sunrise"] < events["solar_noon"] < events["sunset"]
    print(f"  noon az={sun.azimuth_deg:.1f}, el={sun.elevation_deg:.1f}  OK")
    print(f"  sunrise={events['sunrise'].time()} sunset={events['sunset'].time()}  OK")


def test_plausibility():
    print("=== plausibility ===")
    site = SiteConfig(latitude_deg=42.444, longitude_deg=-76.502)
    when = datetime(2026, 6, 21, 17, 0, tzinfo=timezone.utc)
    sun = get_sun_position(site, when)

    ok, err, _ = vision_detection_is_plausible(
        site, sun.azimuth_deg + 3.0, sun.elevation_deg - 2.0,
        tolerance_deg=15.0, when=when)
    assert ok and err < 15.0

    ok2, err2, _ = vision_detection_is_plausible(
        site, (sun.azimuth_deg + 90.0) % 360.0, sun.elevation_deg,
        tolerance_deg=15.0, when=when)
    assert not ok2 and err2 > 15.0
    print(f"  near-sun: ok={ok} err={err:.2f};  90 off: ok={ok2} err={err2:.2f}  OK")


def test_pixel_to_sky():
    print("=== pixel <-> sky ===")
    az, el = pixel_offset_to_sky_angles(
        320, 240, 640, 480,
        mount_azimuth_deg=180.0, mount_elevation_deg=45.0,
        hfov_deg=60.0, vfov_deg=40.0)

    assert abs(az - 180.0) < 0.01 and abs(el - 45.0) < 0.01

    az_r, _ = pixel_offset_to_sky_angles(
        639, 240, 640, 480,
        mount_azimuth_deg=180.0, mount_elevation_deg=45.0,
        hfov_deg=60.0, vfov_deg=40.0)
    assert abs(az_r - 210.0) < 0.5, f"right edge az: {az_r}"
    print(f"  center=({az:.1f},{el:.1f})  right_edge_az={az_r:.1f}  OK")


def test_controller_wiring():
    print("=== controller wiring (stub GPIO) ===")
    site = SiteConfig(latitude_deg=42.444, longitude_deg=-76.502)
    actuators.setup_gpio()
    ctrl = EphemerisAwareController(site, sanity_tolerance_deg=180.0)


    frame = make_sunny_frame(sun_x=550, sun_y=240)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    az_before = ctrl.mount_az_deg
    state, info = ctrl.step_closed_loop(gray)
    assert state == "tracking", f"unexpected state: {state}  info={info}"
    assert info["nx"] > 0, f"blob is to the right; nx should be +ve, got {info['nx']}"
    assert ctrl.mount_az_deg != az_before, "mount-az model should have updated"
    print(f"  state={state}  nx={info['nx']}  mount_az: {az_before:.2f} -> {ctrl.mount_az_deg:.2f}  OK")
    actuators.cleanup_gpio()


def test_open_loop_at_night():
    print("=== open loop at night ===")
    site = SiteConfig(latitude_deg=42.444, longitude_deg=-76.502)
    midnight = datetime(2026, 6, 21, 5, 0, tzinfo=timezone.utc)
    cmd = open_loop_command(site, when=midnight)
    assert not cmd.should_track
    assert "horizon" in cmd.reason or "too low" in cmd.reason
    print(f"  reason='{cmd.reason}'  OK")


def main():
    test_vision()
    test_exposure()
    test_ephemeris()
    test_plausibility()
    test_pixel_to_sky()
    test_controller_wiring()
    test_open_loop_at_night()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
