"""Camera P09: the frame limits a scope publishes, brought up at the binning the camera holds.

`capabilities.camera_limits.frame_max` is the largest unbinned frame. It is
built at connect from the camera's reported maximum, which a Basler body
reports after binning (`Width.Max` follows `BinningHorizontal`), so the build
un-bins it by the binning the camera holds then and takes the smallest of
that, the documented sensor size and the catalogue's ceiling. A rule that
forgot the binning would publish about half the sensor on a scope brought up
at 2x. `frame_min` is the camera's reported minimum, recorded unscaled; whether
it moves with binning is not known from code.

Each run reads the limits the scope came up with and the binning and frame the
camera holds. On hardware the binning is the installation's own, so the read
is made once with the scope set to 1x and once with it set to 2x (set in
LumaViewPro, which then closes); the two runs' `frame_max` must agree, and
their `frame_min` figures say whether the minimum follows binning. The
simulator brings a session up at each of 1x and 2x in one run.

    python tests/capability/camera_p09_frame_limits_at_the_binning_held.py             # simulator, 1x and 2x
    python tests/capability/camera_p09_frame_limits_at_the_binning_held.py --hardware  # the connected scope, at its binning

The check in each run: the largest frame is at least the delivered frame
times the binning, which a maximum left binned fails whenever the delivered
frame is wider than half the sensor. Nothing is saved and nothing moves
beyond the bring-up home.
"""

import sys
import traceback

from harness import HARDWARE, check, figure, hardware_session, make_session, report


def _read(session, label):
    scope = session.scope
    limits = scope.capabilities.camera_limits
    check(f'{label}: the scope publishes camera limits', limits is not None)
    binning = scope.camera.read_binning()
    delivered = scope.camera.read_frame_size()
    figure(
        label,
        {
            'binning': binning,
            'delivered': delivered.to_dict(),
            'frame_max': limits.frame_max.to_dict() if limits.frame_max else None,
            'frame_min': limits.frame_min.to_dict() if limits.frame_min else None,
            'camera': scope.capabilities.camera_model,
        },
    )
    check(
        f'{label}: the largest frame is at least the delivered frame times the binning',
        limits.frame_max is not None
        and limits.frame_max.width >= delivered.width * binning
        and limits.frame_max.height >= delivered.height * binning,
        f'frame_max {limits.frame_max}, delivered {delivered} at {binning}x',
    )


def main():
    if HARDWARE:
        with hardware_session() as (session, _runner):
            _read(session, 'held')
        return
    for size in ('1x1', '2x2'):
        session, _live = make_session(f'probe_frame_limits_{size}', binning={'size': size})
        try:
            _read(session, f'at_{size}')
        finally:
            session.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        check('probe completed without an unexpected raise', False)
    sys.exit(report())
