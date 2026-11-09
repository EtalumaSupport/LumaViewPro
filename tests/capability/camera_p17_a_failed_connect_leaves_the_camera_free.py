"""Camera P17: a Pylon connect that fails after the camera opened leaves the camera free to open again.

`PylonCamera.connect`'s failure handler sets `self.active = None` and stops
its frame worker, with no `Close` or `DestroyDevice` of the camera it opened.
Whether the camera is then left held by the process (a second open refused
as in use) is the SDK's behaviour, which Basler's emulator cannot show: its
control opened a second camera while the first was held. This probe asks the
body. A driver connects as it is constructed (`Camera.__init__`), so each
step is a construction:

1. The control: one driver connects; a second, built while the first holds
   the camera, must be refused. A second open that succeeds here means this
   camera cannot show a hold, and the probe stops.
2. The first disconnects, and a driver is built every half second until one
   connects, timing the release (a reopen half a second after a disconnect
   was refused on the daA3840). Then a driver is built with its
   configuration step made to raise, so its connect fails after `Open()`.
3. A driver is built the same way. Never connecting within 10 s: the failed
   connect left the camera held. Connecting: it did not, and the time beside
   the normal disconnect's says whether its release is any slower.

    python tests/capability/camera_p17_a_failed_connect_leaves_the_camera_free.py --hardware

Hardware only, with LumaViewPro and pylon Viewer closed: no simulated camera
runs the Pylon SDK. The drivers are built directly, since the question is the
driver's own teardown; nothing is lit, nothing moves, and the configuration
step that raises is the only change, made for one construction and put back.
"""

import sys
import time
import traceback

from harness import HARDWARE, check, figure, report, void

from _lumascope.drivers.pyloncamera import PylonCamera

POLL_S = 0.5
FREE_WITHIN_S = 10.0


def _build():
    """A driver built (and so connected); (the driver or None, the failure's words or None)."""
    try:
        camera = PylonCamera()
    except Exception as ex:
        return None, f'{type(ex).__name__}: {ex}'
    if not camera.found:
        return None, 'constructed with found=False'
    return camera, None


def _seconds_until_free():
    """How long until a driver connects again, building one every POLL_S; None past FREE_WITHIN_S.

    A camera released asynchronously is free after a delay; one left held
    by this process is never free while it runs.
    """
    t0 = time.monotonic()
    while time.monotonic() - t0 < FREE_WITHIN_S:
        camera, _why = _build()
        if camera is not None:
            camera.disconnect()
            return round(time.monotonic() - t0, 2)
        time.sleep(POLL_S)
    return None


def _fail_after_the_open(self):
    raise RuntimeError('camera_p17: the configuration step fails on purpose, after Open()')


def _probe():
    first, why = _build()
    check('the control: the first driver connects', first is not None, str(why))
    if first is None:
        return
    second, held_why = _build()
    figure(
        'the control: a second driver while the first holds the camera',
        {'connected': second is not None, 'failure': held_why},
    )
    check(
        'the control: a second driver is refused while the camera is held',
        second is None,
        str(held_why),
    )
    if second is not None:
        second.disconnect()
        first.disconnect()
        return
    first.disconnect()
    after_disconnect = _seconds_until_free()
    figure('seconds until free after a normal disconnect', after_disconnect)
    check(
        'the camera is free after a normal disconnect',
        after_disconnect is not None,
        f'not within {FREE_WITHIN_S} s',
    )
    if after_disconnect is None:
        return
    # The free check's driver has just disconnected too.
    _seconds_until_free()

    # A stand-in by design: a configuration step that fails after Open() is a
    # fault no simulated camera produces, since none runs the Pylon SDK.
    real_step = PylonCamera.init_camera_config
    PylonCamera.init_camera_config = _fail_after_the_open
    try:
        failing, failed_why = _build()
    finally:
        PylonCamera.init_camera_config = real_step
    figure(
        'the driver made to fail after the open',
        {'connected': failing is not None, 'failure': failed_why},
    )
    # Its failure is the configuration step's, not an open refused for another reason.
    check(
        'the driver made to fail after the open fails at its configuration step',
        failing is None and 'camera_p17' in str(failed_why),
        str(failed_why),
    )
    after_failure = _seconds_until_free()
    figure('seconds until free after the failed connect', after_failure)
    check(
        'the camera is free after the failed connect: it was not left held',
        after_failure is not None,
        f'not within {FREE_WITHIN_S} s',
    )


def main():
    if not HARDWARE:
        void(
            'a failed connect leaves the camera free for the next one',
            False,
            'no simulated camera runs the Pylon SDK; a Basler body (--hardware) decides it',
        )
        return
    _probe()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        check('probe completed without an unexpected raise', False)
    sys.exit(report())
