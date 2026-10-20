"""Camera P10: the frames per second the camera delivers, with no display.

The camera's own count of frames handed to the host (the stream's delivered
counter, read by the diagnostic probe before any unpack drops a frame) over a
10 s window, at each frame size, depth and binning that bounds the rate, with
what bounded it: the camera's maximum rate, the bandwidth limit and the
component the camera says limits it, the sensor mode. A window is the camera's
rate only when no frame was lost or short of a buffer in it
(`observed_fps_is_camera_rate`). Beside it, the rate LumaViewPro received
(`get_delivered_rate`): the gap between the two is the host's loss.

Rows, each at a 5 ms exposure so exposure never bounds the rate:
1900 x 1900 at 8-bit mode (the lowest-bandwidth format) and at 12-bit; 956
rows; the largest frame the scope allows; 2x2 binning; and a stop and restart
of the stream. Run it on the IDS scope, then on the Basler on the same
computer; the Basler leaves the IDS-only fields empty.

    python tests/capability/camera_p10_delivered_rate.py             # simulator
    python tests/capability/camera_p10_delivered_rate.py --hardware  # the connected scope

The simulated camera has no stream counters, so on the simulator the camera's
own count is a void and only the received rate is read. Nothing is saved; the
camera returns to the installation's settings at the next bring-up.
"""

import sys
import time
import traceback

from harness import HARDWARE, check, figure, hardware_session, make_session, report, void

EXPOSURE_MS = 5
WINDOW_S = 10
SETTLE_S = 3


def _row(scope, label):
    time.sleep(SETTLE_S)
    probe = scope.diagnostics.run_pylon_diagnostic_probe(WINDOW_S)
    config = probe.get('config', {})
    derived = probe.get('derived', {})
    deltas = probe.get('deltas', {})
    row = {
        'observed_fps': derived.get('observed_fps'),
        'is_camera_rate': derived.get('observed_fps_is_camera_rate'),
        'lvp_fps': scope.imaging.get_delivered_rate().frames_per_s,
        'fps_max': config.get('acquisition_frame_rate_max', config.get('resulting_frame_rate')),
        'limiter': config.get('dltl_component'),
        'dltl_bps': config.get('dltl_value_bps'),
        'dltl_max_bps': config.get('dltl_max_bps'),
        'sensor_mode': config.get('sensor_operation_mode'),
        'pixel_format': config.get('pixel_format'),
        'frame': scope.camera.read_frame_size().to_dict(),
        'binning': scope.camera.read_binning(),
        'lost': deltas.get('StreamLostFrameCount'),
        'underrun': deltas.get('StreamUnderrunCount'),
        'incomplete': deltas.get('StreamIncompleteFrameCount'),
        'dropped': deltas.get('StreamDroppedFrameCount'),
        'probe_file': probe.get('output_path'),
    }
    figure(label, row)
    counted = isinstance(row['observed_fps'], (int, float)) and row['observed_fps'] > 0
    if not HARDWARE:
        void(
            f'{label}: the camera counts the frames it delivers',
            counted,
            'the simulated camera has no stream counters',
        )
    else:
        # On hardware a probe that did not reach the camera is a failure:
        # no camera came up, or its driver has no stream counters.
        check(
            f'{label}: the camera counts the frames it delivers',
            counted,
            f'camera {scope.capabilities.camera_model!r}, probe errors {probe.get("errors")}',
        )
    check(f'{label}: LumaViewPro receives frames', row['lvp_fps'] > 0)
    return row


def _largest_frame(scope, cam, label):
    from modules.exceptions import CameraSettingOutOfRangeError

    limits = scope.capabilities.camera_limits
    largest = limits.frame_max if limits is not None else None
    figure(f'{label}_frame_max', largest.to_dict() if largest else None)
    if largest is None:
        check(f'{label}: the scope publishes its largest frame', False)
        return
    try:
        cam.set_frame_size(1900, largest.height)
    except CameraSettingOutOfRangeError as e:
        check(f'{label}: the largest published height is accepted', False, str(e))
        return
    _row(scope, label)


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
    cam.set_binning(1)
    cam.set_frame_size(1900, 1900)
    _row(scope, 'A1_1900_8bit_mode')

    cam.set_image_mode('12bit_scientific')
    _row(scope, 'A2_1900_12bit')
    cam.set_image_mode('8bit')

    cam.set_frame_size(1900, 956)
    _row(scope, 'A3_956_rows')
    _largest_frame(scope, cam, 'A3_largest_rows')
    cam.set_frame_size(1900, 1900)

    cam.set_binning(2)
    _row(scope, 'A4_binning_2')
    cam.set_binning(1)
    cam.set_frame_size(1900, 1900)

    scope.imaging.disable_streaming()
    scope.imaging.enable_streaming()
    _row(scope, 'A8_after_restart')


def main():
    if HARDWARE:
        with hardware_session() as (session, _runner):
            _measure(session)
        return
    session, _live = make_session('probe_delivered_rate')
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
