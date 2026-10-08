"""Stack S11: what a remote live view costs on this host -- the time to shrink
and JPEG-encode each delivered frame inline on the camera's delivery thread,
the bytes per encoded frame, and the delivered frame rate with that listener
attached against the rate without it.

    python tests/capability/stack_p08_live_encode.py            # simulator
    python tests/capability/stack_p08_live_encode.py --hardware
"""

import statistics
import sys
import threading
import time
import traceback

import cv2
import numpy as np

import harness
from harness import HARDWARE, check, figure, hardware_session, report

WINDOW_S = 5.0
JPEG_QUALITY = 80
SCALES = (1.0, 0.5, 0.25)


def _encoder(scale, stats, lock):
    def handler(image, timestamp, chunks):
        t0 = time.perf_counter()
        frame = image
        if frame.dtype != np.uint8:
            frame = (frame >> (frame.dtype.itemsize * 8 - 8)).astype(np.uint8)
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        ms = (time.perf_counter() - t0) * 1000.0
        with lock:
            stats['ms'].append(ms)
            stats['bytes'].append(len(buf) if ok else 0)

    handler.__qualname__ = f'probe_encode_{scale}'
    return handler


def body(session):
    im = session.scope.imaging
    im.set_exposure_ms(10.0)
    figure('S11.camera_frame_rate_reported', im.get_resulting_frame_rate())
    figure('S11.frame', im.frame_size_cached)
    counter = {'n': 0}
    lock = threading.Lock()

    def count(image, timestamp, chunks):
        with lock:
            counter['n'] += 1

    count.__qualname__ = 'probe_count'
    im.add_frame_listener(count)
    try:
        time.sleep(0.5)
        with lock:
            counter['n'] = 0
        time.sleep(WINDOW_S)
        with lock:
            baseline = counter['n']
        figure('S11.fps_no_encoder', round(baseline / WINDOW_S, 1))
        for scale in SCALES:
            stats = {'ms': [], 'bytes': []}
            handler = _encoder(scale, stats, lock)
            im.add_frame_listener(handler)
            time.sleep(0.5)
            with lock:
                counter['n'] = 0
                stats['ms'].clear()
                stats['bytes'].clear()
            time.sleep(WINDOW_S)
            with lock:
                delivered = counter['n']
                ms = list(stats['ms'])
                nbytes = list(stats['bytes'])
            im.remove_frame_listener(handler)
            tag = f'scale_{scale}'
            figure(f'S11.{tag}.fps_delivered', round(delivered / WINDOW_S, 1))
            figure(f'S11.{tag}.frames_encoded', len(ms))
            if ms:
                figure(f'S11.{tag}.encode_ms_median', round(statistics.median(ms), 1))
                figure(f'S11.{tag}.encode_ms_p95', round(sorted(ms)[int(0.95 * (len(ms) - 1))], 1))
                figure(f'S11.{tag}.jpeg_kb_median', round(statistics.median(nbytes) / 1024, 1))
            check(
                f'S11 {tag}: the encoder stayed attached for the window',
                len(ms) >= 0.5 * delivered,
                f'{len(ms)} encoded of {delivered} delivered (dropped by the handler budget if far fewer)',
            )
    finally:
        im.remove_frame_listener(count)
    check('S11 ran', True)


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
