"""Camera P20: whether the camera's exposure floor moves with its readout, and whether the API follows it.

The API holds the camera's exposure range, `scope.camera.setting_limits`,
read from the camera at connect and again after each readout change (binning,
frame size, pixel format, conversion gain, line-noise reduction). Its range
check (`set_exposure_ms`) and its answer to "what does this stored exposure
become on the camera" (`applied_exposure_ms_for`) both read it. The floor is
known to move: Mono8 to Mono12 raises it on the a2A3536 and the daA3840, and
binning 2 lowers it on the IDS U3-34LxXLS-M.

Each step reads the API's floor (F), the frame rate the settings allow, then
sets the exposure to F through the API and records the exposure the camera
answers it took (E). The steps, every one through the API: the geometry and
mode the scope came up with; binning 1 at the largest frame; binning 1 at the
smallest frame; binning 2; the other image mode (8-bit or 12-bit). What the
scope came up with is put back at the end (the image mode is saved again as
it was).

The check at each step: the camera takes the API's floor, so the API refuses
nothing the camera would take and passes nothing it would refuse. The figure
`the_floor_moved` records whether any step's floor differs from the first's:
the bench record of which settings move it.

    python tests/capability/camera_p20_the_exposure_floor_follows_the_frame.py             # simulator
    python tests/capability/camera_p20_the_exposure_floor_follows_the_frame.py --hardware  # the connected scope

The simulated camera's floor follows its pixel format alone (14 us in Mono8,
29 us in Mono12, the daA3840's). Nothing moves beyond the bring-up home.
"""

import sys
import traceback

from harness import HARDWARE, check, figure, hardware_session, make_session, report, void

from _lumascope.modules.image_mode import ImageMode

# A relative tolerance for "the same exposure": the camera quantizes an
# exposure to its line time, a few microseconds, far under 0.1 % of any floor.
_SAME = 1e-3


def _differs(value, other):
    return value is not None and other is not None and abs(value - other) > other * _SAME


def _step(scope, label, rows):
    cam = scope.camera
    floor = cam.setting_limits.exposure_min_ms
    taken = cam.set_exposure_ms(floor) if floor is not None else None
    row = {
        'binning': cam.read_binning(),
        'frame': cam.read_frame_size().to_dict(),
        'pixel_format': cam.read_pixel_format(),
        'frame_rate_fps': cam.read_frame_rate_fps(),
        'floor_ms': floor,
        'exposure_taken_ms': taken,
    }
    figure(label, row)
    rows.append(row)
    check(f'{label}: the scope publishes an exposure floor', floor is not None)
    check(
        f'{label}: the camera takes the floor the API holds',
        taken is not None and not _differs(taken, floor),
        f'asked {floor} ms, the camera took {taken} ms',
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
    held_mode = cam.image_mode
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
    other_mode = ImageMode.SCIENTIFIC if held_mode == ImageMode.EIGHT_BIT else ImageMode.EIGHT_BIT
    cam.set_image_mode(other_mode)
    _step(scope, f'S4_{other_mode.value}', rows)
    cam.set_image_mode(held_mode)
    check(
        'what the scope came up with is back',
        cam.read_binning() == held_binning
        and cam.read_frame_size() == held_frame
        and cam.image_mode == held_mode,
        f'binning {cam.read_binning()} (was {held_binning}), frame {cam.read_frame_size()} '
        f'(was {held_frame}), mode {cam.image_mode} (was {held_mode})',
    )

    moved = any(_differs(r['floor_ms'], rows[0]['floor_ms']) for r in rows)
    figure('the_floor_moved', moved)
    if not HARDWARE:
        check('the simulated floor moves with the pixel format', moved)


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
