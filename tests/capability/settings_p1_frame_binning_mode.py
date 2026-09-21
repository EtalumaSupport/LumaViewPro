"""Probe 1 -- camera geometry: frame size, binning, image mode (pixel format).

GUI entry points probed headlessly:
  frame size   ui/lumaviewpro.kv:2223,2240 -> ui/microscope_settings.py:1108
  binning      ui/lumaviewpro.kv:2455      -> ui/microscope_settings.py:825
  image mode   ui/lumaviewpro.kv:2413      -> ui/microscope_settings.py:559
"""

import sys

import harness as _common
from modules.lumascope_api._constants import *  # noqa: F403  (no-op if absent)

s, live = _common.make_session()
im = s.scope.imaging
try:
    native = im.get_native_resolution()
    print('native resolution     :', native)
    print('available binning     :', im.get_available_binning_sizes())
    print('supported pix formats :', im.get_supported_pixel_formats())
    print('start w/h             :', im.get_width(), im.get_height())

    # --- frame size, in range -------------------------------------------
    delivered = im.set_frame_size(800, 600)
    _common.void(
        'frame 800x600 applied', delivered == {'width': 800, 'height': 600}, str(delivered)
    )
    _common.void(
        'frame read-back',
        (im.get_width(), im.get_height()) == (800, 600),
        f'{im.get_width()}x{im.get_height()}',
    )
    print('frame_size_cached     :', im.frame_size_cached)

    # --- frame size, OUT OF RANGE (the 13414 shape) ----------------------
    try:
        out = im.set_frame_size(13414, 13414)
        print('set_frame_size(13414,13414) RETURNED:', out)
        _common.void(
            '13414 refused (not silently clamped)',
            False,
            f'clamped to {out} and returned as delivered -- no raise',
        )
    except Exception as e:
        _common.void('13414 refused', True, f'{type(e).__name__}: {e}')
    print('after oversize, read-back:', im.get_width(), im.get_height())

    # --- frame size, zero / negative ------------------------------------
    for bad in ((0, 0), (-10, 600)):
        try:
            out = im.set_frame_size(*bad)
            print(f'set_frame_size{bad} RETURNED:', out)
        except Exception as e:
            print(f'set_frame_size{bad} raised {type(e).__name__}: {e}')

    # --- binning ---------------------------------------------------------
    supported = im.get_available_binning_sizes()
    got = im.set_binning_size(2)
    _common.ok(
        'binning 2 applied',
        bool(got) and im.get_binning_size() == 2,
        f'ret={got} read-back={im.get_binning_size()}',
    )
    bad_bin = next(x for x in (3, 5, 7, 16) if x not in supported)
    try:
        out = im.set_binning_size(bad_bin)
        _common.void(
            f'unsupported binning {bad_bin} refused',
            not out,
            f'returned {out}; read-back {im.get_binning_size()}',
        )
    except Exception as e:
        _common.void(f'unsupported binning {bad_bin} refused', True, f'{type(e).__name__}: {e}')
    im.set_binning_size(1)

    # --- pixel format (image mode) --------------------------------------
    fmts = im.get_supported_pixel_formats()
    if fmts:
        r = im.set_pixel_format(fmts[0])
        _common.ok(
            f'pixel format {fmts[0]} applied',
            bool(r),
            f'ret={r} read-back={im.pixel_format_cached}',
        )
    try:
        out = im.set_pixel_format('Mono99')
        _common.ok('bogus pixel format refused', not out, f'returned {out}')
    except Exception as e:
        _common.ok('bogus pixel format refused', True, f'{type(e).__name__}: {e}')

    # --- is the image MODE (8/12-bit display mode) an API concept? -------
    snap = s.get_settings_snapshot()
    print("settings['image_mode'] :", snap.get('image_mode'))
    print('session has set_image_mode?', hasattr(s, 'set_image_mode'))
    print('imaging has set_image_mode?', hasattr(im, 'set_image_mode'))
finally:
    s.shutdown()

# The recorded results are the probe's verdict, so they have to reach the
# exit code -- without this the slice printed FAIL and exited 0.
sys.exit(_common.report())
