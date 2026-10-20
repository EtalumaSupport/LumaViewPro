"""Camera P08: whether `imaging.is_streaming()` waits while the camera stops its grab.

`ScopeSession.status` reads `is_streaming()` for its `camera_streaming`
field, and the run-state event builds that record on the thread whose
transition it announces, which must not block (the claim's contract). The
read asks the camera SDK (`is_grabbing`), and a grab stop holds the SDK's
lock for as long as the camera takes to stop, measured near 11 s on a Pylon
body. The W3d plan's bench row asks whether the read waits that out.

The probe stops the feed with `imaging.disable_streaming()` on a thread of
its own, times every `is_streaming()` asked from the main thread while the
stop runs, and restarts the feed with `imaging.enable_streaming()`; it does
this CYCLES times, after timing the read on a running feed. A read that
waits on the stop takes about as long as the stop; one that does not stays
near its running-feed time.

    python tests/capability/camera_p08_is_streaming_during_a_grab_stop.py             # simulator
    python tests/capability/camera_p08_is_streaming_during_a_grab_stop.py --hardware  # the connected scope

The figures are printed for a person to judge; the checks are that each stop
and start completed and that the feed is running at the end. Nothing is
saved and nothing moves beyond the bring-up home.
"""

import sys
import threading
import time
import traceback

from harness import HARDWARE, check, figure, hardware_session, make_session, report

CYCLES = 5
BASELINE_READS = 200
POLL_GAP_S = 0.005


def _timed_read(imaging):
    start = time.perf_counter()
    imaging.is_streaming()
    return time.perf_counter() - start


def _ms(timings):
    ms = sorted(t * 1e3 for t in timings)
    return {'n': len(ms), 'median': round(ms[len(ms) // 2], 3), 'max': round(ms[-1], 3)}


def _measure(session):
    imaging = session.scope.imaging
    check('the feed is running before the first stop', imaging.is_streaming())

    figure('read_on_a_running_feed_ms', _ms([_timed_read(imaging) for _ in range(BASELINE_READS)]))

    for cycle in range(1, CYCLES + 1):
        stopped = threading.Event()
        outcome = {}

        def stop(stopped=stopped, outcome=outcome):
            start = time.perf_counter()
            try:
                imaging.disable_streaming()
            except Exception as e:
                outcome['raised'] = f'{type(e).__name__}: {e}'
            outcome['stop_s'] = time.perf_counter() - start
            stopped.set()

        stopper = threading.Thread(target=stop, name='p08_grab_stop', daemon=True)
        stopper.start()
        reads = []
        while not stopped.is_set():
            reads.append(_timed_read(imaging))
            time.sleep(POLL_GAP_S)
        stopper.join()

        check(f'cycle {cycle}: the stop completed', 'raised' not in outcome, outcome.get('raised'))
        figure(
            f'cycle_{cycle}',
            {
                'stop_ms': round(outcome['stop_s'] * 1e3, 1),
                'read_during_stop_ms': _ms(reads) if reads else None,
            },
        )

        start = time.perf_counter()
        imaging.enable_streaming()
        figure(f'cycle_{cycle}_start_ms', round((time.perf_counter() - start) * 1e3, 1))

    check('the feed is running at the end', imaging.is_streaming())


def main():
    if HARDWARE:
        with hardware_session() as (session, _runner):
            _measure(session)
    else:
        session, _live = make_session('probe_is_streaming_during_a_grab_stop')
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
