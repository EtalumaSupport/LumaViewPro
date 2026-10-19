"""Camera P06: how well this host maps the camera's clock -- the latch bracket, the drift, and where a frame waits.

Every capture maps the camera's timestamp clock onto the host's steady clock
(`FrameRecord.clock_mapping`): the host latches the camera's clock between
two reads of its own, and half that bracket is how far the mapping can be
off. Step 7 of the stale-frame plan needs three figures per camera and
host, read here through the capture a REST or SDK caller makes:

  - the bracket, over CAPTURES captures;
  - the drift between the two clocks, from two captures DRIFT_S apart;
  - each delivered frame's arrival on the host after its exposure ended
    (its Timestamp chunk, mapped, plus its ExposureTime chunk), over
    ARRIVAL_S of live frames held through `imaging.add_frame_listener`.

    python tests/capability/camera_p06_the_clock_mapping_figures.py             # simulator
    python tests/capability/camera_p06_the_clock_mapping_figures.py --hardware  # the connected scope

The figures are printed for a person to judge; the checks are that every
capture carried a mapping and that enough frames were timed. About
DRIFT_S + ARRIVAL_S seconds. Reads only; nothing moves beyond the bring-up home.
"""

import sys
import threading
import time
import traceback

from harness import HARDWARE, check, figure, hardware_session, make_session, report

CAPTURES = 50
DRIFT_S = 60.0
ARRIVAL_S = 10.0


def _spread(values, unit_scale, digits):
    v = sorted(values)
    return {
        'n': len(v),
        'min': round(v[0] * unit_scale, digits),
        'median': round(v[len(v) // 2] * unit_scale, digits),
        'p95': round(v[int(len(v) * 0.95)] * unit_scale, digits),
        'max': round(v[-1] * unit_scale, digits),
    }


def _mapping(imaging):
    """One capture's clock mapping, or None when the capture carried none."""
    imaging.capture_and_wait(accept_dark=True, timeout_s=5.0)
    record = (imaging.last_capture_info or {}).get('frame_record')
    return None if record is None else record.clock_mapping


def _arrivals(imaging, mapping):
    """Each held frame's arrival after its exposure ended, in ms."""
    delays = {}
    done = threading.Event()

    def hold(frame):
        chunks = frame.chunks or {}
        ticks, exposure_us = chunks.get('Timestamp'), chunks.get('ExposureTime')
        if ticks is None or exposure_us is None or frame.seq in delays:
            return
        began_ns = mapping.host_ns + (ticks - mapping.camera_ticks) * 1e9 / mapping.tick_hz
        delays[frame.seq] = (frame.arrived_ns - began_ns - exposure_us * 1e3) / 1e6

    imaging.add_frame_listener(hold, name='clock_figures_probe')
    try:
        done.wait(ARRIVAL_S)
    finally:
        imaging.remove_frame_listener(hold)
    return list(delays.values())


def _measure(session):
    imaging = session.scope.imaging
    camera_api = session.scope.camera
    figure('camera', session.scope.capabilities.camera_model)
    figure('exposure_ms', camera_api.read_exposure_ms())

    start = time.monotonic()
    mappings = [_mapping(imaging) for _ in range(CAPTURES)]
    carried = [m for m in mappings if m is not None]
    check(
        f'every one of {CAPTURES} captures carried a clock mapping',
        len(carried) == CAPTURES,
        f'{len(carried)} of {CAPTURES}',
    )
    if not carried:
        return
    figure('bracket_ms', _spread([2 * m.uncertainty_ns for m in carried], 1e-6, 3))
    figure('tick_hz', carried[0].tick_hz)

    delays = _arrivals(imaging, carried[-1])
    check('at least 10 delivered frames were timed', len(delays) >= 10, len(delays))
    if delays:
        figure('arrival_after_exposure_end_ms', _spread(delays, 1.0, 3))

    time.sleep(max(0.0, DRIFT_S - (time.monotonic() - start)))
    last = _mapping(imaging)
    check('the drift capture carried a clock mapping', last is not None)
    if last is None:
        return
    first = carried[0]
    host_s = (last.host_ns - first.host_ns) / 1e9
    camera_s = (last.camera_ticks - first.camera_ticks) / first.tick_hz
    figure('drift_over_s', round(host_s, 1))
    figure('drift_us', round((camera_s - host_s) * 1e6, 1))
    figure('drift_ppm', round((camera_s - host_s) / host_s * 1e6, 2))
    figure('drift_uncertainty_us', round((first.uncertainty_ns + last.uncertainty_ns) / 1e3, 1))


def main():
    if HARDWARE:
        with hardware_session() as (session, _runner):
            _measure(session)
    else:
        session, _live = make_session('probe_clock_mapping_figures')
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
