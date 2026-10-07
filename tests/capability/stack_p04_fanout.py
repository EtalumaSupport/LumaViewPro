"""Stack S5: delivered frames per second against the number and cost of frame
listeners, which run inline on the camera's delivery thread.

    python tests/capability/stack_p04_fanout.py            # simulator
    python tests/capability/stack_p04_fanout.py --hardware
"""

import sys
import threading
import time
import traceback

import harness
from harness import HARDWARE, check, figure, hardware_session, report

WINDOW_S = 5.0


def _busy(ms):
    def handler(image, timestamp, chunks):
        end = time.perf_counter() + ms / 1000.0
        while time.perf_counter() < end:
            pass

    handler.__qualname__ = f'probe_busy_{ms}ms'
    return handler


def body(session):
    im = session.scope.imaging
    im.set_exposure_ms(10.0)
    figure('S5.camera_frame_rate_reported', im.get_resulting_frame_rate())
    counter = {'n': 0}
    lock = threading.Lock()

    def count(image, timestamp, chunks):
        with lock:
            counter['n'] += 1

    count.__qualname__ = 'probe_count'
    im.add_frame_listener(count)
    try:
        for extra, cost_ms in ((0, 0), (1, 5), (3, 5), (3, 20)):
            handlers = [_busy(cost_ms) for _ in range(extra)]
            for h in handlers:
                im.add_frame_listener(h)
            time.sleep(0.5)
            with lock:
                counter['n'] = 0
            time.sleep(WINDOW_S)
            with lock:
                n = counter['n']
            figure(f'S5.fps_with_{extra}_listeners_at_{cost_ms}ms', round(n / WINDOW_S, 1))
            for h in handlers:
                im.remove_frame_listener(h)
    finally:
        im.remove_frame_listener(count)
    check('S5 ran', True)


def main():
    if HARDWARE:
        with hardware_session() as session:
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
