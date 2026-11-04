"""Camera P13: how late a frame listener hears a frame after the driver had it.

A frame's ``arrived_ns`` is stamped by the driver when the grab returns; a
listener added through ``imaging.add_frame_listener`` (a plugin, REST's live
view, the manual recorder) runs later on the same thread, behind the
runtime's own handling and any listener before it. The probe reads
``scope.host_steady_ns()`` as its listener is called and figures the gap per
delivered frame, under three loads:

  - the camera streaming at a short exposure (SHORT_MS) and a long one
    (LONG_MS), gain 0, nothing written;
  - the short exposure while the gain is written back and forth every
    WRITE_EVERY_S, the load a transition measurement puts on the stream.

    python tests/capability/camera_p13_how_late_a_listener_hears_a_frame.py             # simulator
    python tests/capability/camera_p13_how_late_a_listener_hears_a_frame.py --hardware  # the connected scope

The figures are printed for a person to judge; the checks are that enough
frames were heard under each load and that no listener heard a frame before
its arrival, which only a stamp off the scope's one clock could do. The
exposure and gain are put back; nothing moves beyond the bring-up home.
"""

import sys
import threading
import traceback

from harness import HARDWARE, check, figure, hardware_session, make_session, report

SHORT_MS = 18.0
LONG_MS = 134.0
BURST_GAIN_DB = 6.0
SAMPLE_S = 10.0
WRITE_EVERY_S = 0.025
MIN_FRAMES = 20


def _spread_ms(values_ns):
    v = sorted(values_ns)
    return {
        'n': len(v),
        'min': round(v[0] / 1e6, 3),
        'median': round(v[len(v) // 2] / 1e6, 3),
        'p95': round(v[int(len(v) * 0.95)] / 1e6, 3),
        'max': round(v[-1] / 1e6, 3),
    }


def _heard(scope, while_listening):
    """Each frame's listener stamp less its arrival, in ns, while ``while_listening()`` runs."""
    gaps = {}

    def hear(frame):
        heard_ns = scope.host_steady_ns()
        gaps.setdefault(frame.seq, heard_ns - frame.arrived_ns)

    scope.imaging.add_frame_listener(hear, name='listener_lateness_probe')
    try:
        while_listening()
    finally:
        scope.imaging.remove_frame_listener(hear)
    return list(gaps.values())


def _load(label, gaps):
    check(f'{label}: at least {MIN_FRAMES} frames heard', len(gaps) >= MIN_FRAMES, len(gaps))
    if not gaps:
        return
    early = [g for g in gaps if g < 0]
    check(f'{label}: no frame heard before it arrived', not early, f'{len(early)} of {len(gaps)}')
    figure(f'{label}_heard_after_arrival_ms', _spread_ms(gaps))


def _measure(session):
    scope = session.scope
    camera_api = scope.camera
    figure('camera', scope.capabilities.camera_model)
    exposure_before = camera_api.read_exposure_ms()
    gain_before = camera_api.read_gain_db()
    idle = threading.Event()
    try:
        camera_api.set_gain_db(0.0)
        for exposure_ms in (SHORT_MS, LONG_MS):
            camera_api.set_exposure_ms(exposure_ms)
            _load(f'idle_{exposure_ms:g}ms', _heard(scope, lambda: idle.wait(SAMPLE_S)))

        camera_api.set_exposure_ms(SHORT_MS)

        def burst():
            writes = int(SAMPLE_S / WRITE_EVERY_S)
            for i in range(writes):
                camera_api.set_gain_db(BURST_GAIN_DB if i % 2 == 0 else 0.0)
                idle.wait(WRITE_EVERY_S)
            figure('burst_gain_writes', writes)

        _load(f'gain_burst_{SHORT_MS:g}ms', _heard(scope, burst))
    finally:
        camera_api.set_exposure_ms(exposure_before)
        camera_api.set_gain_db(gain_before)


def main():
    if HARDWARE:
        with hardware_session() as (session, _runner):
            _measure(session)
    else:
        session, _live = make_session('probe_listener_lateness')
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
