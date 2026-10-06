"""Probe 8 -- where a typed frame size of 13414 actually goes.

The API refuses a frame the camera cannot take
(ImagingAPI.set_frame_size raises CameraSettingOutOfRangeError) rather than
clamping it and reporting the clamp only in its return value.
"""

import sys

import harness as _common

from modules.exceptions import CameraSettingOutOfRangeError

s, live = _common.make_session()
try:
    im = s.scope.imaging
    before = im.frame_size_cached
    print('largest frame     :', s.scope.capabilities.camera_max_frame_size)
    try:
        delivered = im.set_frame_size(13414, 13414)
        _common.ok('13414 is REFUSED by the API', False, f'delivered {delivered}')
    except CameraSettingOutOfRangeError as e:
        _common.ok('13414 is REFUSED by the API', True, str(e))
    _common.ok(
        'a refused frame leaves the frame as it was',
        im.frame_size_cached == before,
        f'{before} -> {im.frame_size_cached}',
    )
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
