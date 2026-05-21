# Solar Tracker (ephemeris-assisted CV)

Computer-vision solar tracking on a Raspberry Pi, with a sun-position
predictor that catches bad vision detections and keeps the rig pointed in
the right direction when the sun is hidden behind clouds.

## What it does, per frame

```
capture frame
  -> ephemeris night gate     (sun below 5 deg? sleep)
  -> adaptive exposure + DRC  (recover detail under direct sunlight)
  -> isCloudy?                (yes -> open-loop, no -> closed-loop)
  -> closed-loop:
       find brightest blob
       convert pixel -> sky angles
       plausibility check vs ephemeris   (>15 deg off -> reject)
       drive actuators proportionally toward image center
  -> open-loop fallback:
       point mount at ephemeris-predicted sun position
  -> append CSV log line
```

## Layout

```
solar_tracker/
├── src/
│   ├── __init__.py
│   ├── actuators.py         # GPIO control (stub on non-Pi machines)
│   ├── vision.py            # find_sun_blob, isCloudy
│   ├── exposure_utils.py    # adaptive_exposure_drc (CLAHE on V channel)
│   ├── sun_ephemeris.py     # SiteConfig, ephemeris, plausibility, fallback
│   └── controller.py        # the main tracker
├── tests/
│   └── test_pipeline.py     # offline smoke tests (no hardware needed)
├── images/                  # sample sky photos used by demo.py as a mock camera
├── demo.py                  # run the full pipeline on sample images, no hardware
├── main.py                  # live tracker entry point (needs camera + Pi + actuators)
├── requirements.txt         # all platforms
├── requirements-pi.txt      # Pi only (RPi.GPIO)
└── README.md
```

## Install

### On a laptop (development / unit tests)

```bash
git clone <this-repo> solar_tracker && cd solar_tracker
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`actuators.py` autodetects that `RPi.GPIO` is unavailable and uses a stub,
so the whole package imports and the smoke tests run on any OS.

### On the Raspberry Pi

```bash
git clone <this-repo> solar_tracker && cd solar_tracker
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-pi.txt
```

`pysolar` is listed in `requirements.txt` for higher accuracy. If `pip`
can't install it on your platform, the code falls back to a built-in
NOAA implementation (~0.5 deg, fine for plausibility gating).

## Run the smoke tests

These confirm every module wires together without needing a camera or Pi:

```bash
python tests/test_pipeline.py
```

Expected output ends with `All smoke tests passed.`

## Run the demo (no hardware)

`demo.py` replays the photos in `images/` through the full per-frame pipeline
and prints, for each image, **which of the four actuators would move and for
how long** — so you can see the controller working end-to-end on a laptop
with no camera, no Raspberry Pi, and no actuators wired up.

```bash
python demo.py                       # 5 curated images, ~1 second
python demo.py --image cloudy1.jpg   # just one image
python demo.py --all                 # every image in images/
```

What it mocks for you:

| Missing hardware | Stand-in used by demo.py |
|------------------|--------------------------|
| Camera           | `cv2.imread()` of each file in `images/` |
| Raspberry Pi GPIO| `src.actuators._StubGPIO` (auto-active on non-Pi) |
| Linear actuators | Recording wrapper around `src.actuators._drive` — captures direction + duration without sleeping or touching pins |

For determinism, the demo pins the ephemeris to `2026-06-21 17:00 UTC`
(Cornell solar noon) and relaxes the plausibility tolerance to 180° so the
test photos always trigger the closed-loop branch instead of being rejected
for not aligning with the real sky.

Sample output for one image:

```
=== sample_sun3.jpg -- clear sky, sun off-center ===
  loaded sample_sun3.jpg  shape=(480, 640, 3)
  exposure: mean_v=162.0  max_v=255.0  contrast=59.3  -> scale=0.864
  ephemeris (2026-06-21T17:00:00+00:00): az=174.48 deg, el=70.93 deg
  cloud check: clear  ->  closed-loop vision
  sun blob: x=344  y=304  radius=50 px
  controller state: tracking
  normalized command: nx=+0.075  ny=-0.267
  actuator commands issued:
    Actuator 1:  (no motion)
    Actuator 2:  EXTEND   0.060 s
    Actuator 2:  RETRACT  0.060 s
    Actuator 3:  EXTEND   0.213 s
    Actuator 4:  EXTEND   0.060 s
    Actuator 4:  EXTEND   0.213 s
  mount pointing now: az=181.44 deg, el=41.16 deg
```

## Run the tracker

```bash
python -m src.controller --lat 42.444 --lon -76.502
```

All flags:

```
--lat / --lon        site latitude / longitude (degrees, + N / + E)   REQUIRED
--elev               site elevation (m, default 0)
--home-az / --home-el      mount pointing at mid-stroke
--az-range / --el-range    half-range the mount can sweep from home
--camera             OpenCV camera index (default 0)
--hfov / --vfov      camera FOV in degrees (default 60 / 40)
--fps                max loop rate (default 5)
--display            show a preview window with the detected blob
--log                CSV log path (default tracker_log.csv; '' to disable)
--no-drc             disable adaptive exposure / DRC
```

Stop with Ctrl+C — the controller releases the camera and runs
`GPIO.cleanup()` on shutdown.

## One-time site calibration

Before the rig works correctly you need to measure four things and pass
them as CLI flags. None of them require any special equipment.

1. **Latitude / longitude.** From any maps app at the install site.
2. **`--home-az` and `--home-el`** — where the mount is *physically*
   pointing when all four actuators are at mid-stroke. Measure with a
   compass and a phone clinometer. Convention: azimuth is 0 deg at
   north and increases clockwise (E=90, S=180, W=270).
3. **`--az-range` and `--el-range`** — half the total angle the mount
   sweeps from full retract to full extend on each axis. Drive each
   axis end-to-end and measure with the same compass / clinometer.
4. **`--hfov` and `--vfov`** — the camera's actual field of view. Either
   look it up in your webcam's spec sheet, or measure once: point at a
   wall a known distance D away, mark the leftmost and rightmost points
   visible in the frame, measure that width W, then `hfov = 2 * atan(W/2 / D)`.

Once you have those, the same command line works every time.

## Log format

Each loop appends one row to `tracker_log.csv`:

```
ts_utc, state, sun_az, sun_el, vision_az, vision_el,
expected_az, expected_el, ang_err_deg,
mount_az, mount_el, blob_x, blob_y, blob_r, nx, ny, reason
```

`state` is one of:
- `tracking`            — closed-loop, vision drove the mount
- `cloudy/open_loop`    — cloud pre-classifier triggered open-loop slew
- `no_blob->open_loop`  — vision didn't find anything; ephemeris took over
- `vision_implausible->open_loop` — blob too far from where the sun should be
- `sun_not_usable`      — sun below 5 deg; the loop is idling

This is the data you use to tune `k_seconds_per_unit`, `deadband`, and
`max_pulse` later. Drop the CSV into pandas / Excel and look at
`ang_err_deg` over time — if it stays large, gain is too low; if it
oscillates, gain is too high.

## What was changed vs the original code

| Original                  | Status                                       |
|---------------------------|----------------------------------------------|
| `main.py` `isCloudy`      | extracted into `vision.py`                   |
| `main.py` `getSunCoords`  | rewritten as `find_sun_blob` (no plt.show)   |
| `exposure_utils.py`       | kept; long comment block trimmed             |
| `threadingtest.py`        | refactored into `actuators.py` (no side effects on import; stub on non-Pi; end-stop clamping; idempotent setup; `cleanup_gpio`) |
| `controller.py`           | rewritten as `EphemerisAwareController` + main loop, with night gate / plausibility / open-loop fallback / CSV logging / CLI args / proper SIGINT and SIGTERM handling |
| `requirements.txt`        | merge-conflict markers removed; pysolar added; RPi.GPIO moved to separate requirements-pi.txt |
| `webcamcv.py`             | not carried over (duplicate code, unused)    |
| (new) `sun_ephemeris.py`  | astronomical predictor                       |
| (new) `tests/`            | offline smoke tests                          |

## Known limitations

- Mount pointing is tracked by integrating commands — no encoders, so
  the model drifts. Re-home the rig periodically (or implement a
  startup `actuators.home_all()` call once you trust the mechanics).
- `pixel_offset_to_sky_angles` assumes a rectilinear camera. Fisheye or
  wide-angle lenses need a calibrated model.
- The proportional controller has no integral term. Steady-state error
  inside the deadband will not be corrected.

## License

Inherits the license of the original SolarReceiving project.
