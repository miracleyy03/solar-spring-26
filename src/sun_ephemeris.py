from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

try:
    from pysolar.solar import get_altitude, get_azimuth
    _HAS_PYSOLAR = True
except ImportError:
    _HAS_PYSOLAR = False


@dataclass(frozen=True)
class SiteConfig:
    latitude_deg: float
    longitude_deg: float
    elevation_m: float = 0.0
    mount_home_azimuth_deg: float = 180.0
    mount_home_elevation_deg: float = 45.0
    mount_azimuth_range_deg: float = 60.0
    mount_elevation_range_deg: float = 45.0


@dataclass
class SunPosition:
    azimuth_deg: float
    elevation_deg: float
    timestamp_utc: datetime

    @property
    def is_above_horizon(self) -> bool:
        return self.elevation_deg > 0.0

    @property
    def is_usable(self) -> bool:
        return self.elevation_deg >= 5.0


def get_sun_position(site: SiteConfig,
                     when: Optional[datetime] = None) -> SunPosition:
    if when is None:
        when = datetime.now(timezone.utc)
    if when.tzinfo is None:
        raise ValueError("`when` must be timezone-aware")

    if _HAS_PYSOLAR:
        elev = get_altitude(site.latitude_deg, site.longitude_deg, when,
                            elevation=site.elevation_m)
        az = get_azimuth(site.latitude_deg, site.longitude_deg, when,
                         elevation=site.elevation_m) % 360.0
    else:
        az, elev = _fallback_sun_position(site, when)

    return SunPosition(azimuth_deg=az, elevation_deg=elev, timestamp_utc=when)


def get_sun_events_today(site: SiteConfig,
                         when: Optional[datetime] = None,
                         step_minutes: int = 2) -> dict:
    if when is None:
        when = datetime.now(timezone.utc)
    if when.tzinfo is None:
        raise ValueError("`when` must be timezone-aware")

    when_utc = when.astimezone(timezone.utc)
    scan_start = when_utc - timedelta(hours=18)
    scan_end   = when_utc + timedelta(hours=18)
    step = timedelta(minutes=step_minutes)

    samples = []
    t = scan_start
    while t <= scan_end:
        samples.append((t, get_sun_position(site, t).elevation_deg))
        t += step


    crossings = []
    current_sunrise = None
    current_peak_e = -90.0
    current_peak_t = None
    for i in range(1, len(samples)):
        t0, e0 = samples[i - 1]
        t1, e1 = samples[i]

        if e0 <= 0.0 < e1:
            current_sunrise = t0 + (-e0 / (e1 - e0)) * step
            current_peak_e = -90.0
            current_peak_t = None

        if current_sunrise is not None and e1 > current_peak_e:
            current_peak_e = e1
            current_peak_t = t1

        if e0 > 0.0 >= e1 and current_sunrise is not None:
            sunset_t = t0 + (e0 / (e0 - e1)) * step
            crossings.append({
                "sunrise": current_sunrise,
                "solar_noon": current_peak_t,
                "sunset": sunset_t,
                "peak_elevation_deg": current_peak_e,
            })
            current_sunrise = None

    if not crossings:
        return {"sunrise": None, "solar_noon": None, "sunset": None,
                "peak_elevation_deg": None}


    for c in crossings:
        if c["sunrise"] <= when_utc <= c["sunset"]:
            return c
    for c in crossings:
        if c["sunrise"] >= when_utc:
            return c
    return crossings[-1]


def angular_separation_deg(az1: float, el1: float,
                           az2: float, el2: float) -> float:
    a1, e1 = math.radians(az1), math.radians(el1)
    a2, e2 = math.radians(az2), math.radians(el2)
    cos_sep = (math.sin(e1) * math.sin(e2)
               + math.cos(e1) * math.cos(e2) * math.cos(a1 - a2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_sep))))


def vision_detection_is_plausible(
    site: SiteConfig,
    vision_azimuth_deg: float,
    vision_elevation_deg: float,
    tolerance_deg: float = 15.0,
    when: Optional[datetime] = None,
) -> Tuple[bool, float, SunPosition]:
    expected = get_sun_position(site, when)
    err = angular_separation_deg(vision_azimuth_deg, vision_elevation_deg,
                                 expected.azimuth_deg, expected.elevation_deg)
    return err <= tolerance_deg, err, expected


def pixel_offset_to_sky_angles(sun_pixel_x: int, sun_pixel_y: int,
                               frame_width: int, frame_height: int,
                               mount_azimuth_deg: float,
                               mount_elevation_deg: float,
                               hfov_deg: float = 60.0,
                               vfov_deg: float = 40.0) -> Tuple[float, float]:
    cx, cy = frame_width / 2.0, frame_height / 2.0
    nx = (sun_pixel_x - cx) / cx
    ny = (cy - sun_pixel_y) / cy
    d_az = nx * (hfov_deg / 2.0)
    d_el = ny * (vfov_deg / 2.0)
    return (mount_azimuth_deg + d_az) % 360.0, mount_elevation_deg + d_el


def sky_angles_to_normalized_command(
    site: SiteConfig,
    target_azimuth_deg: float,
    target_elevation_deg: float,
    current_mount_azimuth_deg: Optional[float] = None,
    current_mount_elevation_deg: Optional[float] = None,
) -> Tuple[float, float]:
    if current_mount_azimuth_deg is None:
        current_mount_azimuth_deg = site.mount_home_azimuth_deg
    if current_mount_elevation_deg is None:
        current_mount_elevation_deg = site.mount_home_elevation_deg

    d_az = ((target_azimuth_deg - current_mount_azimuth_deg + 540.0) % 360.0) - 180.0
    d_el = target_elevation_deg - current_mount_elevation_deg
    nx = max(-1.0, min(1.0, d_az / site.mount_azimuth_range_deg))
    ny = max(-1.0, min(1.0, d_el / site.mount_elevation_range_deg))
    return nx, ny


@dataclass
class OpenLoopCommand:
    should_track: bool
    normalized_x: float
    normalized_y: float
    reason: str
    expected_sun: SunPosition


def open_loop_command(
    site: SiteConfig,
    current_mount_azimuth_deg: Optional[float] = None,
    current_mount_elevation_deg: Optional[float] = None,
    when: Optional[datetime] = None,
) -> OpenLoopCommand:
    sun = get_sun_position(site, when)
    if not sun.is_above_horizon:
        return OpenLoopCommand(False, 0.0, 0.0, "sun below horizon", sun)
    if not sun.is_usable:
        return OpenLoopCommand(False, 0.0, 0.0,
                               f"sun too low ({sun.elevation_deg:.1f} deg)", sun)

    nx, ny = sky_angles_to_normalized_command(
        site, sun.azimuth_deg, sun.elevation_deg,
        current_mount_azimuth_deg, current_mount_elevation_deg)
    return OpenLoopCommand(True, nx, ny, "open-loop ephemeris tracking", sun)


def _fallback_sun_position(site: SiteConfig, when: datetime) -> Tuple[float, float]:
    utc = when.astimezone(timezone.utc)

    a = (14 - utc.month) // 12
    y = utc.year + 4800 - a
    m = utc.month + 12 * a - 3
    jd = (utc.day + (153 * m + 2) // 5 + 365 * y + y // 4
          - y // 100 + y // 400 - 32045)
    day_frac = (utc.hour + utc.minute / 60.0 + utc.second / 3600.0) / 24.0
    jd = jd + day_frac - 0.5
    n = jd - 2451545.0

    L = (280.460 + 0.9856474 * n) % 360.0
    g = math.radians((357.528 + 0.9856003 * n) % 360.0)
    lam = math.radians(L + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g))
    eps = math.radians(23.439 - 0.0000004 * n)

    ra = math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))
    dec = math.asin(math.sin(eps) * math.sin(lam))

    gmst = (18.697374558 + 24.06570982441908 * n) % 24.0
    lst = math.radians((gmst * 15.0 + site.longitude_deg) % 360.0)
    ha = lst - ra

    lat = math.radians(site.latitude_deg)
    sin_alt = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(ha)
    alt = math.asin(max(-1.0, min(1.0, sin_alt)))
    cos_az = ((math.sin(dec) - math.sin(alt) * math.sin(lat))
              / (math.cos(alt) * math.cos(lat)))
    az = math.acos(max(-1.0, min(1.0, cos_az)))
    if math.sin(ha) > 0.0:
        az = 2 * math.pi - az
    return math.degrees(az) % 360.0, math.degrees(alt)


if __name__ == "__main__":
    site = SiteConfig(latitude_deg=42.4440, longitude_deg=-76.5019,
                      elevation_m=250.0)
    now = datetime.now(timezone.utc)
    sun = get_sun_position(site, now)
    print(f"pysolar available: {_HAS_PYSOLAR}")
    print(f"time (UTC): {now.isoformat()}")
    print(f"sun: az={sun.azimuth_deg:.2f} deg, el={sun.elevation_deg:.2f} deg")
    print(f"above horizon: {sun.is_above_horizon}  usable: {sun.is_usable}")
    events = get_sun_events_today(site, now)
    print(f"events: sunrise={events['sunrise']}, "
          f"noon={events['solar_noon']}, sunset={events['sunset']}")
