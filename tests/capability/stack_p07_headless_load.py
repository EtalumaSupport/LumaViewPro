"""Stack S6 (headless half): delivered frames per second and this process's CPU
and memory while the camera streams with no display -- the number the GUI's
run is compared against.

    python tests/capability/stack_p07_headless_load.py [--seconds N]            # simulator
    python tests/capability/stack_p07_headless_load.py --hardware [--seconds N]
"""

import os
import sys
import threading
import time
import traceback

import psutil

import harness
from harness import HARDWARE, check, figure, hardware_session, report

SECONDS = 60.0
if '--seconds' in sys.argv:
    SECONDS = float(sys.argv[sys.argv.index('--seconds') + 1])


def body(session):
    im = session.scope.imaging
    im.set_exposure_ms(10.0)
    n = {'frames': 0}
    lock = threading.Lock()

    def count(image, timestamp, chunks):
        with lock:
            n['frames'] += 1

    count.__qualname__ = 'probe_count'
    im.add_frame_listener(count)
    time.sleep(1.0)
    with lock:
        n['frames'] = 0
    proc = psutil.Process()
    max_rss = 0
    t0 = os.times()
    w0 = time.perf_counter()
    while time.perf_counter() - w0 < SECONDS:
        time.sleep(1.0)
        max_rss = max(max_rss, proc.memory_info().rss)
    t1 = os.times()
    w1 = time.perf_counter()
    with lock:
        frames = n['frames']
    im.remove_frame_listener(count)
    cpu = (t1.user - t0.user) + (t1.system - t0.system)
    figure('S6.headless.seconds', round(w1 - w0, 1))
    figure('S6.headless.delivered_fps', round(frames / (w1 - w0), 1))
    figure('S6.headless.process_cpu_percent', round(100.0 * cpu / (w1 - w0), 1))
    figure('S6.headless.max_rss_mb', round(max_rss / 1e6))
    figure('S6.headless.load_avg_1m', round(psutil.getloadavg()[0], 1))
    figure('S6.headless.frame', im.frame_size_cached)
    check('S6 headless ran', True)


def main():
    if HARDWARE:
        with hardware_session() as (session, _runner):
            body(session)
        sys.exit(report())
    session = harness.new_session(microscope='LS850T')
    try:
        body(session)
    except BaseException:
        traceback.print_exc()
        check('probe completed without an unexpected raise', False)
    finally:
        session.shutdown()
    harness.assert_no_ui()
    sys.exit(report())


if __name__ == '__main__':
    main()
