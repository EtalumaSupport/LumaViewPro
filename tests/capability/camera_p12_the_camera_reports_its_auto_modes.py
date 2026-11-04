"""Camera P12: what the camera reports of its own auto loops, and how auto-gain settles.

The auto-loop write guard reads the camera's mode (`read_auto_gain`,
`read_auto_exposure`) and trusts the held value except for a one-shot, on
three facts no Pylon text states: a one-shot's `Once` returns to `Off` on its
own, and how soon; a `Continuous` loop never turns itself off; and what a
`Gain` write does while the loop runs (Pylon may refuse it as a locked node,
or take it and let the loop move it back). #741's settle on convergence needs
a fourth: how a body's per-frame chunk gain and exposure move after an arm,
from a dark start and from a bright one, so the window of steady frames and
its ceiling are measured rather than guessed.

    python tests/capability/camera_p12_the_camera_reports_its_auto_modes.py             # simulator
    python tests/capability/camera_p12_the_camera_reports_its_auto_modes.py --hardware  # a Basler scope

On hardware: a transmitted-light sample in the field, nothing else moving,
and the BF layer's illumination set in LumaViewPro to a mid-range field:
the probe lights BF at that stored current, since a field the loop can only
answer at the camera's shortest exposure never settles. It runs a one-shot, holds continuous auto-gain for a
minute, writes the gain under it, then arms twice more from a dark and a
bright start, recording every frame's chunks. It turns the loop and the LED
off at the end. Nothing is saved and the stage does not move beyond the
bring-up home.
"""

import sys
import threading
import time
import traceback

from harness import HARDWARE, check, figure, hardware_session, make_session, report

AUTO_GAIN = {
    'target_brightness': 0.5,
    'min_gain_db': 0.0,
    'max_gain_db': 20.0,
    'max_exposure_ms': 100.0,
    'min_exposure_ms': 0.1,
}
ONCE_WATCH_S = 15.0
CONTINUOUS_HOLD_S = 60.0
SETTLE_RECORD_S = 10.0
# (label, gain dB, exposure ms) the loop starts from when armed.
STARTS = (('dark_start', 0.0, 1.0), ('bright_start', 12.0, 50.0))
# The steady windows reported: frames in a row with gain and exposure unchanged.
WINDOWS = (2, 3, 5)
GAIN_STEADY_DB = 0.05
EXPOSURE_STEADY_FRACTION = 0.005


def _modes(camera):
    return str(camera.read_auto_gain()), str(camera.read_auto_exposure())


def _one_shot(camera):
    camera.set_auto_gain(False, AUTO_GAIN)
    camera.auto_gain_once(
        state=True,
        target_brightness=AUTO_GAIN['target_brightness'],
        min_gain_db=AUTO_GAIN['min_gain_db'],
        max_gain_db=AUTO_GAIN['max_gain_db'],
        ae_max_exposure_ms=AUTO_GAIN['max_exposure_ms'],
    )
    t0 = time.monotonic()
    seen = []
    off_at = {}
    while time.monotonic() - t0 < ONCE_WATCH_S:
        gain, exposure = _modes(camera)
        t = round(time.monotonic() - t0, 3)
        if not seen or seen[-1][1:] != (gain, exposure):
            seen.append((t, gain, exposure))
        for name, mode in (('gain', gain), ('exposure', exposure)):
            if mode == 'Off' and name not in off_at:
                off_at[name] = t
        if len(off_at) == 2:
            break
        time.sleep(0.05)
    figure('one_shot', {'mode_changes': seen, 'off_at_s': off_at})
    check(
        'a one-shot reads Off on its own, both loops',
        len(off_at) == 2,
        f'still {seen[-1][1:]} after {ONCE_WATCH_S} s',
    )


def _continuous_hold(camera):
    camera.set_auto_gain(True, AUTO_GAIN)
    t0 = time.monotonic()
    readings = []
    while time.monotonic() - t0 < CONTINUOUS_HOLD_S:
        readings.append((round(time.monotonic() - t0, 1), *_modes(camera)))
        time.sleep(1.0)
    left = [r for r in readings if r[1:] != ('Continuous', 'Continuous')]
    figure('continuous_hold', {'reads': len(readings), 'not_continuous': left})
    check('a continuous loop never turns itself off', left == [], str(left[:5]))


def _gain_write_under_the_loop(camera):
    before = camera.read_gain_db()
    asked = before + 3.0 if before + 3.0 <= AUTO_GAIN['max_gain_db'] else before - 3.0
    try:
        answered = camera.set_gain_db(asked)
        outcome = f'answered {answered}'
    except Exception as ex:
        outcome = f'raised {type(ex).__name__}: {getattr(ex, "reason", "")} {ex}'
    read_now = camera.read_gain_db()
    time.sleep(1.0)
    read_later = camera.read_gain_db()
    modes = _modes(camera)
    figure(
        'gain_write_under_continuous',
        {
            'before_db': before,
            'asked_db': asked,
            'outcome': outcome,
            'read_at_once_db': read_now,
            'read_after_1s_db': read_later,
            'modes_after': modes,
        },
    )
    check('the loop is still on after a gain write under it', modes == ('Continuous', 'Continuous'))


def _steady_after(series, window):
    """The index of the first frame that opens `window` frames in a row unchanged."""
    for i in range(len(series) - window + 1):
        run = series[i : i + window]
        gains = [g for _s, g, _e in run]
        exposures = [e for _s, _g, e in run]
        if max(gains) - min(gains) <= GAIN_STEADY_DB and (
            max(exposures) - min(exposures) <= EXPOSURE_STEADY_FRACTION * max(exposures)
        ):
            return i
    return None


def _settle(scope, label, gain_db, exposure_ms):
    camera = scope.camera
    camera.set_auto_gain(False, AUTO_GAIN)
    camera.set_gain_db(gain_db)
    camera.set_exposure_ms(exposure_ms)
    time.sleep(0.5)
    frames = []
    lock = threading.Lock()

    def on_frame(frame):
        chunks = frame.chunks or {}
        if 'Gain' in chunks and 'ExposureTime' in chunks:
            with lock:
                frames.append((frame.seq, float(chunks['Gain']), float(chunks['ExposureTime'])))

    scope.imaging.add_frame_listener(on_frame, name='p12_settle')
    try:
        time.sleep(0.5)
        with lock:
            armed_after_seq = frames[-1][0] if frames else -1
        camera.set_auto_gain(True, AUTO_GAIN)
        time.sleep(SETTLE_RECORD_S)
    finally:
        scope.imaging.remove_frame_listener(on_frame)
    with lock:
        under = [f for f in frames if f[0] > armed_after_seq]
    steady = {f'window_{w}': _steady_after(under, w) for w in WINDOWS}
    figure(
        label,
        {
            'start': {'gain_db': gain_db, 'exposure_ms': exposure_ms},
            'frames_under_the_arm': len(under),
            'first_steady_frame': steady,
            'series': [(s, round(g, 3), round(e, 1)) for s, g, e in under],
        },
    )
    check(f'{label}: frames carry chunk gain and exposure', len(under) > 0)
    first = steady[f'window_{WINDOWS[-1]}']
    check(
        f'{label}: the loop holds steady within the record',
        first is not None,
        f'no {WINDOWS[-1]} frames in a row steady in {SETTLE_RECORD_S} s'
        if first is None
        else f'{WINDOWS[-1]} frames in a row steady from frame {first} under the arm',
    )


def _probe(session):
    scope = session.scope
    camera = scope.camera
    has_auto_gain = check(
        'the camera has hardware auto-gain',
        scope.capabilities.camera_supports_auto_gain,
        'this probe is for a body with auto-gain (a Basler)',
    )
    figure('camera', {'model': scope.capabilities.camera_model, 'at_bring_up': _modes(camera)})
    # Every step below arms a loop this camera does not have.
    if not has_auto_gain:
        return
    bf_ma = session.get_layer_configs(['BF'])['BF']['illumination_ma']
    figure('BF mA, the layer stored', bf_ma)
    scope.illumination.led_on('BF', bf_ma)
    try:
        _one_shot(camera)
        _continuous_hold(camera)
        _gain_write_under_the_loop(camera)
        for label, gain_db, exposure_ms in STARTS:
            _settle(scope, label, gain_db, exposure_ms)
    finally:
        camera.set_auto_gain(False, AUTO_GAIN)
        scope.illumination.leds_off()
    check('both loops read Off at the end', _modes(camera) == ('Off', 'Off'))


def main():
    if HARDWARE:
        with hardware_session() as (session, _runner):
            _probe(session)
        return
    session, _live = make_session('probe_auto_modes')
    try:
        _probe(session)
    finally:
        session.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        check('probe completed without an unexpected raise', False)
    sys.exit(report())
