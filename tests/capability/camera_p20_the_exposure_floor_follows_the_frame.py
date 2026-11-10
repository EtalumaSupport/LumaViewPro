"""Camera P20: whether the camera's exposure floor moves with the frame, and whether the API knows it did.

The API holds one exposure floor, `capabilities.camera_limits.exposure_min_ms`,
copied from the camera at connect. Its range check (`set_exposure_ms`) and its
answer to "what does this stored exposure become on the camera"
(`applied_exposure_ms_for`) both read that copy. On IDS bodies the camera's
own floor (the ExposureTime node's minimum) is documented to rise with frame
area, frame rate and pixel clock, and the driver clamps an exposure below it
to it (`read_exposure_min_ms` reads the live floor). If the floor moves, an
exposure at the connect floor is applied higher than asked while
`applied_exposure_ms_for` answers `capped=False`: the API says one exposure
and the camera takes another.

Each step reads the connect floor (C), the camera's live floor (L), the frame
rate the settings allow, then sets the exposure to C through the API and
records the exposure the camera answers it took (E) beside
`applied_exposure_ms_for(C)`. The steps, every one through the API: the
geometry the scope came up with; binning 1 at the largest frame; binning 1 at
the smallest frame; binning 2. The geometry the scope came up with is put back
at the end (in the session only: nothing is saved).

The check at each step: when the camera took more than C, the API says the
exposure was capped. A failure is the defect shown; a pass at every step with
L equal to C is the defect ruled out on this body.

    python tests/capability/camera_p20_the_exposure_floor_follows_the_frame.py             # simulator
    python tests/capability/camera_p20_the_exposure_floor_follows_the_frame.py --hardware  # the connected scope

The simulated camera's floor never moves, so on the simulator whether the
floor moves is a void and the run proves only that the probe runs. Nothing
moves beyond the bring-up home.
"""

import sys
import traceback

from harness import HARDWARE, check, figure, hardware_session, make_session, report, void

# A relative tolerance for "the same exposure": the camera quantizes an
# exposure to its line time, a few microseconds, far under 0.1 % of any floor.
_SAME = 1e-3


def _above(value, floor):
    return value is not None and floor is not None and value > floor * (1 + _SAME)


def _step(scope, label, rows):
    cam = scope.camera
    limits = scope.capabilities.camera_limits
    connect_floor = limits.exposure_min_ms if limits else None
    live_floor = cam.read_exposure_min_ms()
    taken = cam.set_exposure_ms(connect_floor) if connect_floor is not None else None
    applied = cam.applied_exposure_ms_for(connect_floor) if connect_floor is not None else None
    row = {
        'binning': cam.read_binning(),
        'frame': cam.read_frame_size().to_dict(),
        'frame_rate_fps': cam.read_frame_rate_fps(),
        'connect_floor_ms': connect_floor,
        'live_floor_ms': live_floor,
        'exposure_taken_ms': taken,
        'api_applied_ms': applied.applied_ms if applied else None,
        'api_capped': applied.capped if applied else None,
    }
    figure(label, row)
    rows.append(row)
    check(f'{label}: the scope publishes an exposure floor', connect_floor is not None)
    check(f'{label}: the camera answers its live floor', live_floor is not None)
    check(
        f'{label}: the camera takes no exposure under its live floor',
        taken is not None and live_floor is not None and taken >= live_floor * (1 - _SAME),
        f'took {taken} ms, live floor {live_floor} ms',
    )
    if applied is not None:
        check(
            f'{label}: when the camera took more than the connect floor, the API says it was capped',
            not _above(taken, connect_floor) or applied.capped,
            f'asked {connect_floor} ms, the camera took {taken} ms, the API says applied '
            f'{applied.applied_ms} ms, capped={applied.capped}',
        )


def _measure(session):
    scope = session.scope
    cam = scope.camera
    limits = scope.capabilities.camera_limits
    figure('camera_model', scope.capabilities.camera_model)
    check('the scope publishes camera limits', limits is not None)
    sizes = scope.capabilities.camera_binning_sizes
    figure('binning_sizes', sizes)
    held_binning = cam.read_binning()
    held_frame = cam.read_frame_size()
    rows = []

    _step(scope, 'S0_as_brought_up', rows)

    cam.set_binning(1)
    largest = limits.frame_max
    check('the scope publishes its largest frame', largest is not None)
    if largest is not None:
        cam.set_frame_size(largest.width, largest.height)
        _step(scope, 'S1_binning_1_largest_frame', rows)

    smallest = limits.frame_min
    if smallest is None:
        void(
            'S2: the scope publishes its smallest frame',
            False,
            'this camera declares no minimum frame',
        )
    else:
        # Even sides: the API floors a request to them.
        cam.set_frame_size(
            smallest.width + smallest.width % 2, smallest.height + smallest.height % 2
        )
        _step(scope, 'S2_binning_1_smallest_frame', rows)

    if sizes and 2 in sizes:
        cam.set_binning(2)
        _step(scope, 'S3_binning_2', rows)
    else:
        void('S3: the camera offers 2x binning', False, f'binning sizes {sizes}')

    cam.set_binning(held_binning)
    cam.set_frame_size(held_frame.width, held_frame.height)
    check(
        'the geometry the scope came up with is back',
        cam.read_binning() == held_binning and cam.read_frame_size() == held_frame,
        f'binning {cam.read_binning()} (was {held_binning}), frame {cam.read_frame_size()} (was {held_frame})',
    )

    moved = any(_above(r['live_floor_ms'], r['connect_floor_ms']) for r in rows)
    figure('the_live_floor_moved_above_the_connect_floor', moved)
    if not HARDWARE:
        void(
            'the live floor moves with the frame', moved, "the simulated camera's floor never moves"
        )


def main():
    if HARDWARE:
        with hardware_session() as (session, _runner):
            _measure(session)
        return
    session, _live = make_session('probe_exposure_floor')
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
