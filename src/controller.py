from __future__ import annotations

import argparse
import csv
import math
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2

from . import actuators
from .actuators import (
    act1_ext_ret, act2_ext_ret, act3_ext_ret, act4_ext_ret,
    pulse_parallel, FULL_LENGTH,
)
from .exposure_utils import adaptive_exposure_drc
from .vision import find_sun_blob, isCloudy, annotate
from .sun_ephemeris import (
    SiteConfig,
    get_sun_position,
    open_loop_command,
    pixel_offset_to_sky_angles,
    vision_detection_is_plausible,
)


class EphemerisAwareController:
    def __init__(self,
                 site: SiteConfig,
                 k_seconds_per_unit: float = 0.8,
                 max_pulse: float = 0.25,
                 deadband: float = 0.05,
                 settle_delay: float = 0.05,
                 sanity_tolerance_deg: float = 15.0,
                 camera_hfov_deg: float = 60.0,
                 camera_vfov_deg: float = 40.0):
        self.site = site
        self.k = float(k_seconds_per_unit)
        self.max_pulse = float(max_pulse)
        self.deadband = float(deadband)
        self.settle_delay = float(settle_delay)
        self.sanity_tolerance_deg = float(sanity_tolerance_deg)
        self.hfov = float(camera_hfov_deg)
        self.vfov = float(camera_vfov_deg)


        self.mount_az_deg = site.mount_home_azimuth_deg
        self.mount_el_deg = site.mount_home_elevation_deg


    def _pulse_azimuth(self, nx: float) -> None:
        dur = min(self.max_pulse, abs(nx) * self.k)
        if dur <= 0:
            return
        if nx > 0:

            pulse_parallel([
                (act2_ext_ret, 1, 0, dur),
                (act4_ext_ret, 1, 0, dur),
                (act1_ext_ret, 0, 1, dur),
                (act3_ext_ret, 0, 1, dur),
            ])
        else:
            pulse_parallel([
                (act1_ext_ret, 1, 0, dur),
                (act3_ext_ret, 1, 0, dur),
                (act2_ext_ret, 0, 1, dur),
                (act4_ext_ret, 0, 1, dur),
            ])

        change = math.copysign(1.0, nx) * 2.0 * self.site.mount_azimuth_range_deg * dur / FULL_LENGTH
        self.mount_az_deg = (self.mount_az_deg + change) % 360.0

    def _pulse_elevation(self, ny: float) -> None:
        dur = min(self.max_pulse, abs(ny) * self.k)
        if dur <= 0:
            return
        if ny > 0:

            pulse_parallel([
                (act1_ext_ret, 1, 0, dur),
                (act2_ext_ret, 1, 0, dur),
                (act3_ext_ret, 0, 1, dur),
                (act4_ext_ret, 0, 1, dur),
            ])
        else:
            pulse_parallel([
                (act3_ext_ret, 1, 0, dur),
                (act4_ext_ret, 1, 0, dur),
                (act1_ext_ret, 0, 1, dur),
                (act2_ext_ret, 0, 1, dur),
            ])
        change = math.copysign(1.0, ny) * 2.0 * self.site.mount_elevation_range_deg * dur / FULL_LENGTH
        self.mount_el_deg = max(0.0, min(90.0, self.mount_el_deg + change))

    def command(self, nx: float, ny: float) -> None:
        if abs(nx) < self.deadband: nx = 0.0
        if abs(ny) < self.deadband: ny = 0.0
        if nx == 0.0 and ny == 0.0:
            return
        self._pulse_azimuth(nx)
        time.sleep(self.settle_delay)
        self._pulse_elevation(ny)
        time.sleep(self.settle_delay)


    def step_closed_loop(self, gray) -> tuple:
        h, w = gray.shape[:2]
        blob = find_sun_blob(gray)
        if blob is None:
            return "no_blob", {}

        sun_px_x, sun_px_y, radius = blob

        vision_az, vision_el = pixel_offset_to_sky_angles(
            sun_px_x, sun_px_y, w, h,
            self.mount_az_deg, self.mount_el_deg,
            self.hfov, self.vfov)

        ok, err_deg, expected = vision_detection_is_plausible(
            self.site, vision_az, vision_el,
            tolerance_deg=self.sanity_tolerance_deg)

        info = {
            "vision_az": round(vision_az, 2),
            "vision_el": round(vision_el, 2),
            "expected_az": round(expected.azimuth_deg, 2),
            "expected_el": round(expected.elevation_deg, 2),
            "ang_err_deg": round(err_deg, 2),
            "mount_az": round(self.mount_az_deg, 2),
            "mount_el": round(self.mount_el_deg, 2),
            "blob_x": sun_px_x, "blob_y": sun_px_y, "blob_r": radius,
        }

        if not ok:
            return "vision_implausible", info

        err_x_px = sun_px_x - w / 2.0
        err_y_px = (h / 2.0) - sun_px_y
        nx = max(-1.0, min(1.0, err_x_px / (w / 2.0)))
        ny = max(-1.0, min(1.0, err_y_px / (h / 2.0)))
        info["nx"], info["ny"] = round(nx, 3), round(ny, 3)

        self.command(nx, ny)
        return "tracking", info

    def step_open_loop(self) -> tuple:
        cmd = open_loop_command(
            self.site,
            current_mount_azimuth_deg=self.mount_az_deg,
            current_mount_elevation_deg=self.mount_el_deg)
        info = {
            "expected_az": round(cmd.expected_sun.azimuth_deg, 2),
            "expected_el": round(cmd.expected_sun.elevation_deg, 2),
            "mount_az": round(self.mount_az_deg, 2),
            "mount_el": round(self.mount_el_deg, 2),
            "nx": round(cmd.normalized_x, 3),
            "ny": round(cmd.normalized_y, 3),
            "reason": cmd.reason,
        }
        if not cmd.should_track:
            return "sun_not_usable", info
        self.command(cmd.normalized_x, cmd.normalized_y)
        return "open_loop", info


class CsvLogger:
    FIELDS = [
        "ts_utc", "state", "sun_az", "sun_el", "vision_az", "vision_el",
        "expected_az", "expected_el", "ang_err_deg", "mount_az", "mount_el",
        "blob_x", "blob_y", "blob_r", "nx", "ny", "reason",
    ]

    def __init__(self, path: Optional[str]):
        self.path = path
        self._f = None
        self._w = None
        if path:
            new = not Path(path).exists()
            self._f = open(path, "a", newline="")
            self._w = csv.DictWriter(self._f, fieldnames=self.FIELDS,
                                     extrasaction="ignore")
            if new:
                self._w.writeheader()
                self._f.flush()

    def log(self, state: str, info: dict, sun_az=None, sun_el=None):
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        row = {"ts_utc": ts, "state": state,
               "sun_az": sun_az, "sun_el": sun_el}
        row.update({k: v for k, v in info.items() if k in self.FIELDS})
        msg = f"[{ts}] {state:20s} " + " ".join(
            f"{k}={v}" for k, v in info.items())
        print(msg)
        if self._w is not None:
            self._w.writerow(row)
            self._f.flush()

    def close(self):
        if self._f:
            self._f.close()


def run_tracker(site: SiteConfig,
                camera_index: int = 0,
                display: bool = False,
                max_fps: float = 5.0,
                csv_path: Optional[str] = "tracker_log.csv",
                use_adaptive_exposure: bool = True,
                idle_sleep_s: float = 30.0,
                hfov_deg: float = 60.0,
                vfov_deg: float = 40.0,
                home_on_start: bool = False):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: could not open camera index {camera_index}")
        sys.exit(1)

    actuators.setup_gpio()
    if home_on_start:
        print("Homing all actuators to retracted position...")
        actuators.home_all()
        print("Homing complete.")
    ctrl = EphemerisAwareController(site,
                                     camera_hfov_deg=hfov_deg,
                                     camera_vfov_deg=vfov_deg)
    logger = CsvLogger(csv_path)

    stop = {"flag": False}
    def _shutdown(*_):
        print("\nShutting down...")
        stop["flag"] = True
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    min_dt = 1.0 / max(0.1, max_fps)
    print(f"Tracker running. Site: lat={site.latitude_deg}, "
          f"lon={site.longitude_deg}. Ctrl+C to exit.")

    try:
        while not stop["flag"]:
            loop_start = time.time()

            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.1)
                continue

            sun = get_sun_position(site)


            if not sun.is_usable:
                logger.log("sun_not_usable",
                           {"reason": f"el={sun.elevation_deg:.1f}"},
                           sun_az=round(sun.azimuth_deg, 2),
                           sun_el=round(sun.elevation_deg, 2))

                for _ in range(int(idle_sleep_s * 10)):
                    if stop["flag"]:
                        break
                    time.sleep(0.1)
                continue


            if use_adaptive_exposure:
                working_frame, _, exp_scale = adaptive_exposure_drc(frame)
                try:
                    cur = cap.get(cv2.CAP_PROP_EXPOSURE)
                    if cur and cur > 0:
                        cap.set(cv2.CAP_PROP_EXPOSURE, cur * exp_scale)
                except Exception:
                    pass
            else:
                working_frame = frame


            if isCloudy(working_frame):
                state, info = ctrl.step_open_loop()
                logger.log(f"cloudy/{state}", info,
                           sun_az=round(sun.azimuth_deg, 2),
                           sun_el=round(sun.elevation_deg, 2))
            else:
                gray = cv2.cvtColor(working_frame, cv2.COLOR_BGR2GRAY)
                state, info = ctrl.step_closed_loop(gray)
                if state in ("no_blob", "vision_implausible"):
                    state_ol, info_ol = ctrl.step_open_loop()
                    merged = {**info, **info_ol}
                    logger.log(f"{state}->{state_ol}", merged,
                               sun_az=round(sun.azimuth_deg, 2),
                               sun_el=round(sun.elevation_deg, 2))
                else:
                    logger.log(state, info,
                               sun_az=round(sun.azimuth_deg, 2),
                               sun_el=round(sun.elevation_deg, 2))

            if display:
                blob_for_display = None
                if "blob_x" in info:
                    blob_for_display = (info["blob_x"], info["blob_y"], info["blob_r"])
                disp = annotate(working_frame, blob_for_display)
                cv2.putText(disp,
                            f"az={sun.azimuth_deg:.1f} el={sun.elevation_deg:.1f} | {state}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 2)
                cv2.imshow("Sun Tracker", disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            elapsed = time.time() - loop_start
            if elapsed < min_dt:
                time.sleep(min_dt - elapsed)
    finally:
        print("Releasing camera and GPIO...")
        try: cap.release()
        except Exception: pass
        try: cv2.destroyAllWindows()
        except Exception: pass
        actuators.cleanup_gpio()
        logger.close()


def main():
    p = argparse.ArgumentParser(description="Ephemeris-aware solar tracker")
    p.add_argument("--lat", type=float, required=True, help="Site latitude (deg, + N)")
    p.add_argument("--lon", type=float, required=True, help="Site longitude (deg, + E)")
    p.add_argument("--elev", type=float, default=0.0, help="Site elevation (m)")
    p.add_argument("--home-az", type=float, default=180.0,
                   help="Mount azimuth at mid-stroke (deg, 0=N CW)")
    p.add_argument("--home-el", type=float, default=45.0,
                   help="Mount elevation at mid-stroke (deg)")
    p.add_argument("--az-range", type=float, default=60.0,
                   help="Mount azimuth half-range from home (deg)")
    p.add_argument("--el-range", type=float, default=45.0,
                   help="Mount elevation half-range from home (deg)")
    p.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    p.add_argument("--hfov", type=float, default=60.0,
                   help="Camera horizontal FOV (deg)")
    p.add_argument("--vfov", type=float, default=40.0,
                   help="Camera vertical FOV (deg)")
    p.add_argument("--fps", type=float, default=5.0, help="Max loop rate")
    p.add_argument("--display", action="store_true", help="Show preview window")
    p.add_argument("--log", default="tracker_log.csv",
                   help="CSV log path (use '' to disable)")
    p.add_argument("--no-drc", action="store_true",
                   help="Disable adaptive exposure / DRC")
    p.add_argument("--home", action="store_true",
                   help="Retract all actuators to home before starting")
    args = p.parse_args()

    site = SiteConfig(
        latitude_deg=args.lat,
        longitude_deg=args.lon,
        elevation_m=args.elev,
        mount_home_azimuth_deg=args.home_az,
        mount_home_elevation_deg=args.home_el,
        mount_azimuth_range_deg=args.az_range,
        mount_elevation_range_deg=args.el_range,
    )
    run_tracker(site,
                camera_index=args.camera,
                display=args.display,
                max_fps=args.fps,
                csv_path=args.log or None,
                use_adaptive_exposure=not args.no_drc,
                hfov_deg=args.hfov,
                vfov_deg=args.vfov,
                home_on_start=args.home)


if __name__ == "__main__":
    main()
