"""Stack S8/S9: what a headless session is made of -- its threads by name, its
bring-up and home time, its shutdown time -- on the firmware tier.

    python tests/capability/stack_p05_session_shape.py
"""

import sys
import threading
import time

import harness
from harness import check, figure, report


def main():
    before = {t.name for t in threading.enumerate()}
    t0 = time.perf_counter()
    session, _live = harness.make_session(
        'stack_p05', home=True, microscope='LS850T', simulator_tier='firmware'
    )
    figure('S9.bring_up_and_home_s', round(time.perf_counter() - t0, 2))
    figure('S9.tier', type(session.scope._motion_driver).__name__)
    threads = sorted(t.name for t in threading.enumerate() if t.name not in before)
    figure('S8.threads_headless', len(threads))
    figure('S8.thread_names', threads)
    t1 = time.perf_counter()
    session.shutdown()
    figure('S9.shutdown_s', round(time.perf_counter() - t1, 2))
    time.sleep(0.5)
    left = sorted(t.name for t in threading.enumerate() if t.name not in before)
    figure('S8.threads_left_after_shutdown', left)
    check('S8/S9 ran', True)
    harness.assert_no_ui()
    sys.exit(report())


if __name__ == '__main__':
    main()
