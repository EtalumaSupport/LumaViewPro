"""Probe 8 -- where a typed frame size of 13414 actually goes.

Two clamps sit in series, neither of which refuses:
  1. the GUI's own, ui/microscope_settings.py:1152 ->
     modules/binning.displayed_to_native(..., cap=native_max)
  2. the driver's, behind ImagingAPI.set_frame_size, which RETURNS the
     delivered geometry (modules/lumascope_api/imaging.py:1339 contract).
"""

import sys

import harness as _common
from modules import binning

s, live = _common.make_session()
try:
    im = s.scope.imaging
    native_max = im.get_native_resolution()
    typed = {'width': 13414, 'height': 13414}
    alignment = im.get_pixel_alignment()
    native = binning.displayed_to_native(typed, 1, native_max)
    displayed = binning.native_to_displayed(native, 1, alignment)
    print('native_max        :', native_max)
    print('pixel alignment   :', alignment)
    print('GUI clamp of 13414:', native, '->', displayed)
    delivered = im.set_frame_size(typed['width'], typed['height'])
    print('API delivered     :', delivered)
    _common.ok('the two clamps agree', displayed == delivered, f'{displayed} vs {delivered}')
    _common.void(
        '13414 is REFUSED somewhere in the chain',
        False,
        'both layers clamp; the GUI clamp is silent, the API clamp is '
        'reported only in the return value',
    )
except Exception:
    import traceback

    traceback.print_exc()
finally:
    s.shutdown()

# The recorded results are the probe's verdict, so they have to reach the
# exit code -- without this the slice printed FAIL and exited 0.
sys.exit(_common.report())
