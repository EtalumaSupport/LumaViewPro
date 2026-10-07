"""Stack S3 (the still): wall time from a still's request to its file on disk,
through the one manual-capture path, at 8-bit and at full depth.

    python tests/capability/stack_p02_still_to_file.py            # simulator
    python tests/capability/stack_p02_still_to_file.py --hardware
"""

import statistics
import sys
import time
import traceback

import harness
from harness import HARDWARE, check, figure, hardware_session, report

N = 5


def body(session):
    il = session.scope.illumination
    il.led_on('BF', 10)
    try:
        for mode in ('8bit', '12bit_scientific'):
            session.set_image_mode(mode)
            samples = []
            sizes = []
            for _ in range(N):
                t = time.perf_counter()
                fut = session.manual_capture.capture(layer='BF', false_color_on=False)
                paths = fut.result(timeout=60)
                samples.append(time.perf_counter() - t)
                sizes.append(paths[0].stat().st_size)
            figure(
                f'S3.still_request_to_file_{mode}',
                {
                    'median_s': round(statistics.median(samples), 3),
                    'max_s': round(max(samples), 3),
                    'file_kb': round(sizes[-1] / 1024),
                    'frame': session.scope.imaging.frame_size_cached,
                },
            )
    finally:
        il.led_off('BF')
    check('S3 still ran', True)


def main():
    if HARDWARE:
        with hardware_session() as session:
            body(session)
        sys.exit(report())
    from tests.scope_fakes import TEST_TURRET_OBJECTIVES

    session, _live = harness.make_session(
        'stack_p02', home=True, microscope='LS850T', turret_objectives=dict(TEST_TURRET_OBJECTIVES)
    )
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
