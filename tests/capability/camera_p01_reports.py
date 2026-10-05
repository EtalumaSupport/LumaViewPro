"""Camera P01: the camera reports its black level, frame rate and link, and every frame records the black level.

The figures a camera states about itself, read through the API: the black
level (`get_black_level`) and the one each captured frame's record carries,
the frame rate its settings allow (`get_resulting_frame_rate`; the FX2
reports none), the frame rate the stream delivers (frame arrivals counted
by a frame listener, at the session's exposure and at 1 ms), and the link
(`diagnostics.get_camera_link_info`). Reads only; a scope with a stage does
not move.

    python tests/capability/camera_p01_reports.py             # an LS620 simulator
    python tests/capability/camera_p01_reports.py --hardware  # the connected scope
"""

import contextlib
import sys
import threading
import time

from harness import HARDWARE, check, figure, hardware_session, report

DELIVERY_WINDOW_S = 5.0


@contextlib.contextmanager
def _ls620_simulator():
    """An LS620 on the simulator: the production FX2 drivers over a simulated FX2."""
    from modules.scope_session import ScopeSession

    from tests.settings_fixtures import complete_settings

    session = ScopeSession.create(
        complete_settings(microscope='LS620'), simulate=True, warn_pre_release=False
    )
    try:
        yield session, None
    finally:
        session.shutdown()


def _delivered_fps(imaging):
    """Frames per second reaching the host over the window, by arrival."""
    arrivals = []
    lock = threading.Lock()

    def arrived(image, timestamp, chunks):
        with lock:
            arrivals.append(time.monotonic())

    imaging.add_frame_listener(arrived, name='camera_p01_delivery')
    try:
        time.sleep(DELIVERY_WINDOW_S)
    finally:
        imaging.remove_frame_listener(arrived)
    with lock:
        if len(arrivals) < 2:
            return None
        return round((len(arrivals) - 1) / (arrivals[-1] - arrivals[0]), 2)


def main():
    session_cm = hardware_session() if HARDWARE else _ls620_simulator()
    with session_cm as (session, _runner):
        scope = session.scope
        imaging = scope.imaging
        figure('camera', imaging.camera_identity)

        level = imaging.get_black_level()
        figure('black level', level)
        imaging.capture_and_wait(accept_dark=True, timeout_s=5.0)
        record = (imaging.last_capture_info or {}).get('frame_record')
        recorded = None if record is None else record.black_level
        figure('black level in the frame record', recorded)
        check(
            'a captured frame records the black level the camera reports',
            recorded == level,
            recorded,
        )

        figure('resulting frame rate (camera)', imaging.get_resulting_frame_rate())
        exposure = imaging.get_exposure_ms()
        figure('exposure ms', exposure)
        figure('delivered fps at the session exposure', _delivered_fps(imaging))
        imaging.set_exposure_ms(1.0)
        try:
            figure('resulting frame rate (camera) at 1 ms', imaging.get_resulting_frame_rate())
            figure('delivered fps at 1 ms', _delivered_fps(imaging))
        finally:
            imaging.set_exposure_ms(exposure)

        link = scope.diagnostics.get_camera_link_info()
        figure('link', link)
        check('the link is reported', link is not None, link)
        if link is not None:
            check(
                'a link speed carries its unit',
                (link['link_speed'] is None) == (link['link_speed_unit'] is None),
                link,
            )
    sys.exit(report())


if __name__ == '__main__':
    main()
