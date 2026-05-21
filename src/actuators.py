from __future__ import annotations

import threading
import time
from typing import Dict, Optional

try:
    import RPi.GPIO as GPIO
    _ON_PI = True
except (ImportError, RuntimeError):

    _ON_PI = False

    class _StubGPIO:
        BOARD = "BOARD"
        OUT = "OUT"
        LOW = 0
        HIGH = 1
        _state: Dict[int, int] = {}

        def setmode(self, _mode): pass
        def setup(self, pin, _mode): self._state[pin] = 0
        def output(self, pin, val): self._state[pin] = int(bool(val))
        def cleanup(self): self._state.clear()

    GPIO = _StubGPIO()


EXT_PINS = [18, 15, 11, 31]
RET_PINS = [16, 13,  7, 29]

FULL_LENGTH = 5.0


position: Dict[int, float] = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}

_setup_done = False


def setup_gpio() -> None:
    global _setup_done
    if _setup_done:
        return
    GPIO.setmode(GPIO.BOARD)
    for pin in EXT_PINS + RET_PINS:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)
    _setup_done = True


def cleanup_gpio() -> None:
    global _setup_done
    try:
        GPIO.cleanup()
    except Exception:
        pass
    _setup_done = False


def _drive(idx: int, hi: int, ret: int, dur: float) -> None:
    if hi and ret:
        print(f"[actuators] WARNING: actuator {idx} extend+retract both active — skipping to prevent shoot-through")
        return
    if not _setup_done:
        setup_gpio()
    if dur <= 0:
        return

    if hi:
        dur = min(dur, max(0.0, FULL_LENGTH - position[idx]))
    if ret:
        dur = min(dur, max(0.0, position[idx]))
    if dur <= 0:
        return

    e_pin = EXT_PINS[idx - 1]
    r_pin = RET_PINS[idx - 1]
    GPIO.output(e_pin, int(bool(hi)))
    GPIO.output(r_pin, int(bool(ret)))
    time.sleep(dur)
    GPIO.output(e_pin, 0)
    GPIO.output(r_pin, 0)

    if hi:
        position[idx] = min(FULL_LENGTH, position[idx] + dur)
    if ret:
        position[idx] = max(0.0, position[idx] - dur)


def act1_ext_ret(hi: int, ret: int, dur: float = 0.0) -> None: _drive(1, hi, ret, dur)
def act2_ext_ret(hi: int, ret: int, dur: float = 0.0) -> None: _drive(2, hi, ret, dur)
def act3_ext_ret(hi: int, ret: int, dur: float = 0.0) -> None: _drive(3, hi, ret, dur)
def act4_ext_ret(hi: int, ret: int, dur: float = 0.0) -> None: _drive(4, hi, ret, dur)


def pulse_parallel(commands) -> None:
    threads = []
    for fn, hi, ret, dur in commands:
        t = threading.Thread(target=fn, args=(hi, ret, dur))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


def home_all(timeout: Optional[float] = None) -> None:
    if timeout is None:
        timeout = FULL_LENGTH + 0.5
    pulse_parallel([
        (act1_ext_ret, 0, 1, timeout),
        (act2_ext_ret, 0, 1, timeout),
        (act3_ext_ret, 0, 1, timeout),
        (act4_ext_ret, 0, 1, timeout),
    ])
    for k in position:
        position[k] = 0.0


if __name__ == "__main__":

    setup_gpio()
    print(f"Running on Pi: {_ON_PI}")
    print(f"Pins set up. Positions: {position}")
    cleanup_gpio()
