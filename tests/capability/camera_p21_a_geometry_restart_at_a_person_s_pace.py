"""Camera P21: whether a series of stream restarts stalls the camera's stop, the #847 reproduction.

A readout change (binning, frame size, pixel format) restarts the grab. On
a daA3840 with camera firmware 1.1.0 a series of restarts ends with every
stop taking 11 s in the camera's own AcquisitionStop, a state only a power
cycle clears; on the bench (2026-11-13) every 1.1.0 body did so under
either stop mode, after 12 to 142 restarts, and no 2.x body did. The legs:

- A: P20's changes (binning 1 at the largest frame; the smallest frame;
  binning 2; the binning and frame the scope came up with; the other image
  mode; the mode it came up with) at a person's pace, reading the rate
  delivered after each; its check is that frames came after every change.
- B: the same changes with no wait, or ``--gap-s`` after each.
- F: the frame alone, smallest and as brought up, one restart per change,
  ``--gap-s`` after each, 60 restarts.
- S: as F, between the camera characterization's frames.
- R: a logged run's camera writes made again at their spacing (``--replay``).

Each of A and B runs the series three times. What each stop took is the
driver's own record, camera.log's ``pylon StopGrabbing [...] returned after
N ms`` line with the state it stopped from; this probe records each change
and when it was made, so the log's stop lines are read against them.

    python tests/capability/camera_p21_a_geometry_restart_at_a_person_s_pace.py             # simulator
    python tests/capability/camera_p21_a_geometry_restart_at_a_person_s_pace.py --hardware  # the connected scope
    python tests/capability/camera_p21_a_geometry_restart_at_a_person_s_pace.py --hardware --legs BF --gap-s 2.5

Run it from a power cycle: a stalled 1.1.0 body stays stalled until one.
Nothing moves beyond the bring-up home; no light is lit.
"""

import pathlib
import re
import sys
import time
import traceback

from harness import HARDWARE, check, figure, hardware_session, make_session, report

from _lumascope.modules.image_mode import ImageMode

PASSES = 3

# ``--legs`` names the legs to run, in order (``A``, ``B``, ``F``, ``S``,
# ``R``); A and B by default.
_args = sys.argv[1:]
LEGS = _args[_args.index('--legs') + 1] if '--legs' in _args else 'AB'
# ``--gap-s X``: legs B, F and S wait X seconds after each change, with no
# rate reading; whether frames came between is camera.log's buffer count.
GAP_S = float(_args[_args.index('--gap-s') + 1]) if '--gap-s' in _args else 0.0
# ``--objective <catalogue key>``: one the installation names, selected at
# bring-up so a scope whose objective is unconfirmed is not stopped; the
# result does not depend on the optics.
OBJECTIVE = _args[_args.index('--objective') + 1] if '--objective' in _args else None
# ``--legs R --replay <file>``: a run's camera writes, one per line as the
# driver logged them (``HH:MM:SS.mmm pylon Width.SetValue(...) ...``), made
# again through the API at the same spacing.
REPLAY = _args[_args.index('--replay') + 1] if '--replay' in _args else None

# A person's pace: a setting changed, the image looked at, the next one
# changed. Long enough that the API's delivered rate (published each 1 s
# over the last window) has a window wholly after the change.
_PERSON_S = 2.5


def _changes(scope):
    """The series, as (label, change) pairs, P20's order; the scope's own geometry and mode are read once."""
    cam = scope.camera
    limits = scope.capabilities.camera_limits
    held_binning = cam.read_binning()
    held_frame = cam.read_frame_size()
    held_mode = cam.image_mode
    largest = limits.frame_max
    smallest = limits.frame_min
    other_mode = ImageMode.SCIENTIFIC if held_mode == ImageMode.EIGHT_BIT else ImageMode.EIGHT_BIT

    def back():
        cam.set_binning(held_binning)
        cam.set_frame_size(held_frame.width, held_frame.height)

    def largest_frame():
        cam.set_binning(1)
        cam.set_frame_size(largest.width, largest.height)

    return [
        ('binning_1_largest_frame', largest_frame),
        (
            'smallest_frame',
            lambda: cam.set_frame_size(
                smallest.width + smallest.width % 2, smallest.height + smallest.height % 2
            ),
        ),
        ('binning_2', lambda: cam.set_binning(2)),
        ('as_brought_up', back),
        (f'mode_{other_mode.value}', lambda: cam.set_image_mode(other_mode)),
        (f'mode_{held_mode.value}', lambda: cam.set_image_mode(held_mode)),
    ]


def _leg(scope, leg, paced):
    rows = []
    changes = _changes(scope)
    for n in range(1, PASSES + 1):
        for label, change in changes:
            made = time.strftime('%H:%M:%S')
            t0 = time.perf_counter()
            change()
            row = {
                'pass': n,
                'change': label,
                'made_at': made,
                'change_ms': round((time.perf_counter() - t0) * 1000.0, 1),
            }
            if paced:
                time.sleep(_PERSON_S)
                row['delivered_fps'] = scope.imaging.wait_for_delivered_rate(3.0).frame_rate_fps
            elif GAP_S:
                time.sleep(GAP_S)
            figure(f'{leg}_p{n}_{label}', row)
            rows.append(row)
    return rows


def _measure(session):
    scope = session.scope
    figure('camera_model', scope.capabilities.camera_model)
    sizes = scope.capabilities.camera_binning_sizes
    check('the camera offers 2x binning', bool(sizes) and 2 in sizes, f'binning sizes {sizes}')
    check(
        'the scope publishes its largest and smallest frames',
        scope.capabilities.camera_limits.frame_max is not None
        and scope.capabilities.camera_limits.frame_min is not None,
    )

    figure('legs', LEGS)
    figure('leg_B_gap_s', GAP_S)
    if 'A' in LEGS:
        paced = _leg(scope, 'A_paced', paced=True)
        starved = [f'pass {r["pass"]} {r["change"]}' for r in paced if not r['delivered_fps'] > 0]
        check(
            "A: the camera delivered frames after every change at a person's pace",
            not starved,
            f'none delivered after: {starved}',
        )
    if 'B' in LEGS:
        _leg(scope, 'B_rapid', paced=False)
    if 'F' in LEGS:
        _frame_only(scope)
    if 'S' in LEGS:
        _frame_only(scope, sizes=_CHARACTERIZATION_SIZES, leg='S')
    if 'R' in LEGS:
        _replay(scope, REPLAY)


_WRITE = re.compile(
    r'^(\d\d):(\d\d):(\d\d\.\d+) pylon (?:'
    r'Width\.SetValue\((?P<w>\d+)\) Height\.SetValue\((?P<h>\d+)\)'
    r"|PixelFormat\.SetValue\('(?P<pf>Mono\d+)'\)"
    r'|ExposureTime\.SetValue\((?P<us>\d+)us\)'
    r'|Gain\.SetValue\((?P<gain>[-\d.]+)\)'
    r'|BlackLevel\.SetValue\((?P<black>[-\d.]+)\)'
    r'|BinningVertical/Horizontal\.SetValue\((?P<bin>\d+)\)'
    r')'
)


def _replay(scope, path):
    """Leg R: each logged camera write made again through the API, at the spacing it was logged with."""
    cam = scope.camera
    writes = []
    for line in pathlib.Path(path).read_text(encoding='utf-8').splitlines():
        m = _WRITE.match(line)
        if m:
            at = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
            writes.append((at, m))
    figure('replay_writes', len(writes))
    refused = []
    start = time.monotonic()
    first = writes[0][0]
    for n, (at, m) in enumerate(writes, 1):
        wait = (at - first) - (time.monotonic() - start)
        if wait > 0:
            time.sleep(wait)
        try:
            if m.group('w'):
                cam.set_frame_size(int(m.group('w')), int(m.group('h')))
            elif m.group('pf'):
                cam.set_image_mode(
                    ImageMode.SCIENTIFIC if m.group('pf') != 'Mono8' else ImageMode.EIGHT_BIT
                )
            elif m.group('us'):
                cam.set_exposure_ms(int(m.group('us')) / 1000.0)
            elif m.group('gain'):
                cam.set_gain_db(float(m.group('gain')))
            elif m.group('black'):
                cam.set_black_level(float(m.group('black')))
            elif m.group('bin'):
                cam.set_binning(int(m.group('bin')))
        except (
            Exception
        ) as e:  # a refusal is recorded and the replay goes on: the stops are the question
            refused.append(f'{n}: {m.group(0)[13:]} -> {type(e).__name__}: {e}')
    figure('replay_refused', refused)
    figure('replay_took_s', round(time.monotonic() - start, 1))


# The frames the camera characterization sets (its 2026-11-12 run under
# CompleteExposure: 321 stops, none slow), none of them the camera's smallest.
_CHARACTERIZATION_SIZES = ((2100, 2100), (1052, 1050))


def _frame_only(scope, sizes=None, leg='F'):
    """Legs F and S: one restart per change, ``GAP_S`` after each, so every stop can have frames behind it.

    The frame alone, at the binning it holds: a frame write restarts the
    grab once, where a binning write restarts it twice in one call (the
    binning, then the frame), the second stop coming before the first frame.
    Leg F goes between the smallest frame and the one the scope came up
    with; leg S between the characterization's two (``sizes``).
    """
    cam = scope.camera
    if sizes is None:
        held = cam.read_frame_size()
        smallest = scope.capabilities.camera_limits.frame_min
        sizes = (
            (smallest.width + smallest.width % 2, smallest.height + smallest.height % 2),
            (held.width, held.height),
        )
    for n in range(1, 31):
        for size in sizes:
            made = time.strftime('%H:%M:%S')
            cam.set_frame_size(*size)
            time.sleep(GAP_S)
            figure(f'{leg}_c{n}_{size[0]}x{size[1]}', made)


def main():
    if HARDWARE:
        with hardware_session(objective=OBJECTIVE) as (session, _runner):
            _measure(session)
        return
    session, _live = make_session('probe_geometry_restart_pace')
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
