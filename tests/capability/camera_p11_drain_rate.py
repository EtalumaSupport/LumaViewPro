"""Camera P11: the IDS frame rate with LumaViewPro's unpack out of the loop.

At 2x2 binning the U3-34L delivers 46 to 54 fps under LumaViewPro and 66 in
IDS Cockpit, with the camera set to 66 in both and no frame counted lost
(camera P10, 2026-10-20): the frames are never started on the host. This
splits the cause. Each size is read twice: the normal pipeline through the
diagnostic probe (the stream's own delivered count over 10 s), then a drain
in which the unpack worker is paused and a loop only waits for each finished
buffer and hands it straight back to the camera (the driver's
`_drain_finished_buffers`, the unpack benchmark's own loop), timed over
DRAIN_FRAMES. A drain at 66 puts the gap in LumaViewPro's pipeline; a drain
at the normal rate puts it in the stream's configuration.

    python tests/capability/camera_p11_drain_rate.py --hardware  # the connected IDS scope

The drain reaches the driver directly, as the unpack benchmark does: it is a
bench instrument, not an API member. While it runs the live view stores no
frame, so the stream check may report one stall; that report is expected.
The simulated camera has no drain, so the simulator run is a void. Nothing
is saved.
"""

import sys
import time
import traceback

from harness import HARDWARE, check, figure, hardware_session, make_session, report, void

EXPOSURE_MS = 5
WINDOW_S = 10
SETTLE_S = 3
DRAIN_FRAMES = 300


def _normal(scope, label):
    time.sleep(SETTLE_S)
    probe = scope.diagnostics.run_pylon_diagnostic_probe(WINDOW_S)
    derived = probe.get('derived', {})
    deltas = probe.get('deltas', {})
    row = {
        'observed_fps': derived.get('observed_fps'),
        'is_camera_rate': derived.get('observed_fps_is_camera_rate'),
        'started': deltas.get('StreamStartedFrameCount'),
        'delivered': deltas.get('StreamDeliveredFrameCount'),
        'fps_max': probe.get('config', {}).get('acquisition_frame_rate_max'),
        'frame': scope.camera.read_frame_size().to_dict(),
        'binning': scope.camera.read_binning(),
    }
    figure(f'{label}_normal', row)
    return row


def _drain(scope, label):
    driver = scope._camera_driver
    if not hasattr(driver, '_drain_finished_buffers'):
        void(f'{label}: the drain runs', False, 'this camera driver has no drain (the simulator)')
        return None
    frames = []
    started = time.monotonic()
    driver._drain_finished_buffers(
        DRAIN_FRAMES, lambda img: frames.append(time.monotonic()), label='drain-rate'
    )
    span = (frames[-1] - frames[0]) if len(frames) > 1 else 0.0
    row = {
        'frames': len(frames),
        'of': DRAIN_FRAMES,
        'fps': (len(frames) - 1) / span if span > 0 else None,
        'wall_s': round(time.monotonic() - started, 2),
    }
    figure(f'{label}_drain', row)
    check(f'{label}: the drain received frames', len(frames) > 1, str(row))
    return row


def _measure(session):
    scope = session.scope
    cam = scope.camera
    figure('camera_model', scope.capabilities.camera_model)
    figure('firmware', scope.diagnostics.run_pylon_diagnostic_probe(0).get('camera'))
    if not scope.imaging.is_streaming():
        scope.imaging.enable_streaming()
    check('the camera is streaming', scope.imaging.is_streaming())
    figure('exposure_ms', cam.set_exposure_ms(EXPOSURE_MS))
    cam.set_image_mode('8bit')

    cam.set_binning(2)
    _normal(scope, 'B1_binning_2')
    _drain(scope, 'B1_binning_2')
    _normal(scope, 'B1_binning_2_after')

    cam.set_binning(1)
    cam.set_frame_size(1900, 1900)
    _normal(scope, 'B2_1900')
    _drain(scope, 'B2_1900')


def main():
    if HARDWARE:
        with hardware_session() as (session, _runner):
            _measure(session)
        return
    session, _live = make_session('probe_drain_rate')
    try:
        _measure(session)
    finally:
        session.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        check('probe completed without an unexpected raise', False)
    sys.exit(report())
