"""Camera P05: what a copy of a delivered frame costs on this host, at this camera's frame size and rate.

A delivered frame is read-only, so whoever draws on it draws on its own copy:
the preview and a capture draw the scale bar (`image_utils.add_scale_bar`),
the video writer draws the timestamp (`image_utils.add_timestamp`). The probe
holds the camera's delivered frames through `imaging.add_frame_listener`, as
a plugin or REST's live view does, and times on its own thread the bare
copy and each drawing on frames as the camera made them. The cost is this
host's: a fast development machine says nothing about the field host.

    python tests/capability/camera_p05_what_a_frame_copy_costs.py             # simulator
    python tests/capability/camera_p05_what_a_frame_copy_costs.py --hardware  # the connected scope

The figures are printed for a person to judge; the checks are that every
delivered frame refused a write and that neither drawing changed the frame
it was given. Reads only; nothing moves beyond the bring-up home.
"""

import platform
import sys
import threading
import time
import traceback

import numpy as np
from harness import HARDWARE, check, figure, hardware_session, make_session, report

FRAMES = 60
SAMPLE_S = 30.0


def _host():
    import os

    figure('host', f'{platform.system()} {platform.release()} {platform.machine()}')
    figure('host_cpu', f'{platform.processor() or "unknown"}, {os.cpu_count()} logical cores')


def _held_frames(imaging):
    """Up to FRAMES delivered frames, each as the camera delivered it, and the
    rate they arrived at."""
    frames = []
    full = threading.Event()

    def hold(frame):
        if len(frames) < FRAMES:
            frames.append(frame)
        else:
            full.set()

    imaging.add_frame_listener(hold, name='copy_cost_probe')
    try:
        full.wait(SAMPLE_S)
    finally:
        imaging.remove_frame_listener(hold)
    held = frames[:FRAMES]
    span_s = (held[-1].arrived_ns - held[0].arrived_ns) / 1e9 if len(held) > 1 else 0.0
    return held, (len(held) - 1) / span_s if span_s else 0.0


def _ms(timings):
    ms = sorted(t * 1e3 for t in timings)
    return {
        'median': round(ms[len(ms) // 2], 3),
        'p95': round(ms[int(len(ms) * 0.95)], 3),
        'max': round(ms[-1], 3),
    }


def _measure(session):
    from modules import image_utils

    scope = session.scope
    imaging = scope.imaging
    camera_api = scope.camera
    frames, rate_hz = _held_frames(imaging)
    check(
        f'the camera delivered {FRAMES} frames within {SAMPLE_S:.0f} s',
        len(frames) == FRAMES,
        len(frames),
    )
    if not frames:
        return
    first = frames[0].image
    figure(
        'frame',
        f'{first.shape[1]} x {first.shape[0]} {first.dtype}, {frames[0].significant_bits} bits',
    )
    figure('delivered_hz', round(rate_hz, 1))

    refused = 0
    for frame in frames:
        try:
            frame.image[0, 0] = 0
        except ValueError:
            refused += 1
    check(
        'every delivered frame refuses a write',
        refused == len(frames),
        f'{refused} of {len(frames)}',
    )

    objective = scope.runtime_state.get_current_objective()
    binning = camera_api.get_binning_size()

    def bar(image):
        return image_utils.add_scale_bar(
            image=image,
            objective=objective,
            binning_size=binning,
            significant_bits=frames[0].significant_bits,
            capabilities=scope.capabilities,
        )

    drawings = {
        'copy': lambda image: image.copy(),
        'timestamp': lambda image: image_utils.add_timestamp(image, '2026-10-16 12:00:00.000'),
        'scale_bar': bar,
    }
    for name, draw in drawings.items():
        timings = []
        unchanged = True
        copied = True
        for frame in frames:
            before = frame.image.copy()
            start = time.perf_counter()
            drawn = draw(frame.image)
            timings.append(time.perf_counter() - start)
            unchanged = unchanged and np.array_equal(before, frame.image)
            copied = copied and drawn is not frame.image
        ms = _ms(timings)
        figure(f'{name}_ms', ms)
        if rate_hz:
            figure(f'{name}_ms_per_s_at_{rate_hz:.0f}_hz', round(ms['median'] * rate_hz, 1))
        # A scale bar with no known pixel size returns its input: nothing was timed.
        check(f"the {name} made a copy, so its time is a copy's", copied)
        check(f'the {name} left the frame it was given unchanged', unchanged)


def main():
    _host()
    if HARDWARE:
        with hardware_session() as (session, _runner):
            _measure(session)
    else:
        session, _live = make_session('probe_frame_copy_cost')
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
