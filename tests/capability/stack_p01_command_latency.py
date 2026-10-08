"""Stack S1/S2: a command's round trip at rest and while another axis travels;
an LED command's round trip alone and while motor commands flow on the same lane.

The firmware tier runs the production drivers against the real firmware behind an
emulated serial port, so the lane hops, the driver lock and the monitor's poll
cadence are the product's; the serial transit is the simulator's.

    python tests/capability/stack_p01_command_latency.py            # simulator
    python tests/capability/stack_p01_command_latency.py --hardware
"""

import statistics
import sys
import threading
import time
import traceback

import harness
from harness import HARDWARE, check, figure, hardware_session, report
from lib import profile_trace


def _serial_rows():
    """Rows of the serial trace so far, by board: one row per exchange."""
    import collections
    import csv
    import pathlib

    out = collections.Counter()
    base = profile_trace._output_dir
    if base is None:
        return out
    f = pathlib.Path(base) / 'serial_trace.csv'
    if not f.is_file():
        return out
    with open(f) as fh:
        for row in csv.DictReader(fh):
            out[row['board']] += 1
    return out


def _exchanges(before, after, n):
    return {
        b: round((after[b] - before.get(b, 0)) / n, 1)
        for b in after
        if after[b] != before.get(b, 0)
    }


N = 20


def _ms(samples):
    s = sorted(samples)
    return {
        'median_ms': round(statistics.median(s) * 1000, 1),
        'p95_ms': round(s[int(0.95 * (len(s) - 1))] * 1000, 1),
        'max_ms': round(s[-1] * 1000, 1),
    }


def _timed(fn, n=N):
    out = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t)
    return out


def body(session):
    m = session.scope.motion
    il = session.scope.illumination
    z0 = m.get_target_position('Z')
    zl = m.get_axis_limits('Z')
    step = 100.0
    lo, hi = z0, min(zl['max'], z0 + step)
    targets = [hi, lo]

    # S1a: Z moves at rest, alternating +100 um / back.
    i = 0

    def z_move():
        nonlocal i
        m.move_absolute('Z', targets[i % 2])
        i += 1

    b = _serial_rows()
    at_rest = _timed(z_move)
    figure('S1.z_move_100um_at_rest', _ms(at_rest))
    figure('S1.serial_exchanges_per_z_move', _exchanges(b, _serial_rows(), N))

    # How long a long X travel takes here (whether "during a move" is realisable).
    has_x = session.scope.capabilities.has_xy_stage
    during = None
    if has_x:
        xl = m.get_axis_limits('X')
        x0 = m.get_target_position('X')
        far = xl['max'] if (xl['max'] - x0) > (x0 - xl['min']) else xl['min']
        t = time.perf_counter()
        m.move_absolute('X', far)
        figure('S1.x_full_travel_s', round(time.perf_counter() - t, 2))
        # S1b: Z moves while X travels back (the monitor polls X at 50 Hz meanwhile).
        handle = m.start_move_absolute('X', x0)
        during = _timed(z_move, n=min(N, 10))
        handle.wait()
        figure('S1.z_move_100um_during_x_travel', _ms(during))

    # S2a: LED alone.
    def led():
        il.led_on('BF', 10)
        il.led_off('BF')

    b = _serial_rows()
    alone = _timed(led)
    figure('S2.led_on_off_alone', _ms(alone))
    figure('S2.serial_exchanges_per_led_on_off', _exchanges(b, _serial_rows(), N))

    # S2b: LED while a thread keeps Z moves flowing on the same IO lane.
    stop = threading.Event()

    def hammer():
        j = 0
        while not stop.is_set():
            m.move_absolute('Z', targets[j % 2])
            j += 1

    th = threading.Thread(target=hammer, name='probe-z-hammer', daemon=True)
    th.start()
    time.sleep(0.2)
    busy = _timed(led)
    stop.set()
    th.join(timeout=10)
    figure('S2.led_on_off_with_z_moves_on_lane', _ms(busy))
    import os

    figure('S1.load_avg_1m', round(os.getloadavg()[0], 1))
    check('S1/S2 ran', True)


def main():
    if HARDWARE:
        profile_trace.enable(output_dir=str(harness.live_dir('stack_p01') / 'profile'))
        with hardware_session() as (session, _runner):
            body(session)
        sys.exit(report())
    session, _live = harness.make_session(
        'stack_p01', home=True, microscope='LS850T', simulator_tier='firmware'
    )
    profile_trace.enable(output_dir=str(_live / 'profile'))
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
