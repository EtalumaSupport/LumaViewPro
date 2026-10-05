"""Camera P02: the pixel formats the camera offers, and the grid its frame windows sit on.

Two facts about the camera body, read at connect: the formats it offers, as
the scope's capabilities publish them (`capabilities.camera_pixel_formats`)
and as the imaging API asks the camera (`get_supported_pixel_formats`); and
the frame-size grid, the increment of the camera's Width / Height nodes. The
API delivers any even frame size by acquiring the next window up and
cropping, so no API member states the grid: it is read from the driver's
`_frame_grid`, the one place the camera's own increment is asked. Reads
only; nothing moves and no light is turned on.

    python tests/capability/camera_p02_formats_and_grid.py             # a simulated LS850T
    python tests/capability/camera_p02_formats_and_grid.py --hardware  # the connected scope
"""

import sys
import traceback

from harness import HARDWARE, check, figure, hardware_session, make_session, report


def _read(session):
    scope = session.scope
    published = tuple(scope.capabilities.camera_pixel_formats)
    asked = tuple(scope.imaging.get_supported_pixel_formats())
    figure('camera', scope.imaging.camera_identity)
    figure('formats the capabilities publish', published)
    figure('formats the camera reports', asked)
    check('the camera reports its formats', bool(asked))
    check('the capabilities publish the formats the camera reports', published == asked)

    grid = scope.imaging._driver._frame_grid()
    check('the camera reports its frame grid', grid is not None)
    if grid is None:
        return
    figure('frame grid step (width, height)', grid.step)
    figure('frame size minimum (width, height)', grid.size_min)
    figure('frame size maximum (width, height)', grid.max_size)
    figure('binning', scope.imaging.get_binning_size())


def main():
    if HARDWARE:
        with hardware_session() as (session, _runner):
            _read(session)
        return
    session, _live = make_session('probe_camera', microscope='LS850T')
    try:
        _read(session)
    finally:
        session.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        check('probe completed without an unexpected raise', False)
    sys.exit(report())
