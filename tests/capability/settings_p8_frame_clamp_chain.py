"""Probe 8 -- where a typed frame size of 13414 actually goes.

The API sets a frame side past the camera's limits at the limit
(CameraAPI.set_frame_size), saves it, answers the frame delivered, and
reports the clamp in one ValueClampedNotice, not in its return value alone.
"""

import sys

import harness as _common

from _lumascope.modules.exceptions import ValueClampedNotice
from _lumascope.modules.notification_center import Severity, notifications

s, live = _common.make_session()
try:
    im = s.scope.imaging
    im_camera = s.scope.camera
    largest = s.scope.capabilities.camera_limits.frame_max
    print('largest frame     :', largest)
    told = []
    notifications.add_listener(told.append, min_severity=Severity.DEBUG)
    delivered = im_camera.set_frame_size(13414, 13414)
    _common.ok(
        '13414 is set at the largest frame',
        delivered == im_camera.frame_size
        and delivered.width <= largest.width
        and delivered.height <= largest.height,
        f'delivered {delivered} against {largest}',
    )
    clamps = [post for post in told if getattr(post, 'reason', None) == ValueClampedNotice.reason]
    _common.ok('the clamp is told once', len(clamps) == 1, f'{len(clamps)} clamp notices')
except Exception:
    import traceback

    traceback.print_exc()
    # Re-raised: a probe that crashed has no verdict, and exiting 0
    # here read as a pass.
    raise
finally:
    s.shutdown()

# The recorded results are the probe's verdict, so they have to reach the
# exit code -- without this the slice printed FAIL and exited 0.
sys.exit(_common.report())
