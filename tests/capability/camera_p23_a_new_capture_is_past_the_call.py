"""Camera P23: a new capture is a frame stored after the call, on the camera's own wait.

`grab_new_capture(timeout_s)` promises a frame that arrived after the call.
The base `Camera` waits on the handler's stored count; Pylon overrides it
with a private one-slot queue it drains and then blocks on. Whether the
base's wait answers the same on a Basler under `LatestImageOnly` decides
whether that override can go; this probe gives the figures for either
driver, so a run with the override and one without compare like for like.

Two passes, live view on, nothing lit:

- 50 direct calls to the driver's `grab_new_capture`: each returned frame's
  `stored` must be past the handler's stored count read at the call, and
  none may return None; the wait of each is recorded.
- 50 `capture_still` calls, the exposure alternated between two values
  before each, so each capture owes frames for the change and drains
  through `grab_new_capture`: none may raise, and each frame's camera
  timestamp must be past the one before; the time of each is recorded.

    python tests/capability/camera_p23_a_new_capture_is_past_the_call.py             # simulator
    python tests/capability/camera_p23_a_new_capture_is_past_the_call.py --hardware  # the connected scope

The exposure is put back as it was at the end. The stage does not move
beyond the bring-up home.
"""

import statistics
import sys
import time
import traceback

from harness import HARDWARE, check, figure, hardware_session, make_session, report
from _lumascope.modules.scope_parts import Board

CALLS = 50
GRAB_TIMEOUT_S = 2.0
# How long live view runs before the first call, so frames are flowing.
GRAB_WARMUP_S = 2.0
# The two exposures the capture pass alternates between, in ms.
EXPOSURES_MS = (5.0, 6.0)


def _spread(values_ms):
    return {
        'n': len(values_ms),
        'median_ms': round(statistics.median(values_ms), 2),
        'max_ms': round(max(values_ms), 2),
        'min_ms': round(min(values_ms), 2),
    }


def _direct_pass(driver):
    not_past = []
    nones = 0
    waits_ms = []
    for i in range(CALLS):
        before = driver.delivered_counts[0]
        t0 = time.monotonic()
        frame = driver.grab_new_capture(timeout_s=GRAB_TIMEOUT_S)
        waits_ms.append((time.monotonic() - t0) * 1e3)
        if frame is None:
            nones += 1
            continue
        if frame.stored <= before:
            not_past.append({'call': i, 'stored_at_call': before, 'frame_stored': frame.stored})
    figure('direct: wait', _spread(waits_ms))
    figure('direct: frames not past the call', not_past)
    check(f'direct: every one of {CALLS} calls returned a frame', nones == 0, f'{nones} None')
    check('direct: every frame was stored after its call', not not_past, str(not_past[:5]))


def _capture_pass(scope):
    camera = scope.camera
    imaging = scope.imaging
    held_ms = camera.exposure_ms
    raised = []
    not_later = []
    took_ms = []
    last_ticks = None
    try:
        for i in range(CALLS):
            camera.set_exposure_ms(EXPOSURES_MS[i % 2])
            t0 = time.monotonic()
            try:
                still = imaging.capture_still(timeout_s=5.0)
            except Exception as ex:
                raised.append({'call': i, 'raised': repr(ex)})
                continue
            took_ms.append((time.monotonic() - t0) * 1e3)
            ticks = still.record.camera_timestamp_ticks
            if last_ticks is not None and ticks is not None and ticks <= last_ticks:
                not_later.append({'call': i, 'ticks': ticks, 'before': last_ticks})
            last_ticks = ticks
    finally:
        if held_ms is not None:
            camera.set_exposure_ms(held_ms)
    figure('capture: exposure before the pass, ms', held_ms)
    if took_ms:
        figure('capture: time', _spread(took_ms))
    figure('capture: raised', raised)
    check(f'capture: every one of {CALLS} stills was taken', not raised, str(raised[:5]))
    check(
        'capture: each still is a later frame than the one before',
        not not_later,
        str(not_later[:5]),
    )


def _probe(session):
    scope = session.scope
    driver = scope._camera_driver
    figure('camera', {'model': scope.capabilities.camera_model, 'driver': type(driver).__name__})
    figure(
        'grab_new_capture',
        'its own override'
        if 'grab_new_capture' in type(driver).__dict__
        else 'the base Camera wait',
    )
    imaging = scope.imaging
    imaging.enable_streaming()
    try:
        time.sleep(GRAB_WARMUP_S)
        _direct_pass(driver)
        _capture_pass(scope)
    finally:
        imaging.disable_streaming()


def main():
    if HARDWARE:
        with hardware_session(needs=(Board.CAMERA,)) as (session, _runner):
            _probe(session)
        return
    session, _live = make_session('probe_new_capture')
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
