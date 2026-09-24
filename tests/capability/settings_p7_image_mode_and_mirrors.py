"""Probe 7 -- image mode as a user capability, and the settings mirrors.

GUI entry point: ui/lumaviewpro.kv:2413 -> ui/microscope_settings.py:559
select_image_mode(): resolves the mode to a capture depth, picks a pixel
format the sensor supports, pushes it, and commits settings['image_mode']
+ scope_display.image_mode.

Also asks the mirror question for frame size and binning: after an API
apply, does the session's settings dict follow, or only the camera?
"""

import sys

import harness as _common
from modules import image_mode

s, live = _common.make_session()
try:
    im = s.scope.imaging
    print(
        'modes available with this camera:',
        image_mode.available_modes(im.get_supported_pixel_formats()),
    )

    # The whole capability, done by the caller (there is no API for it)
    mode = '12bit_scientific'
    depth = image_mode.resolve_image_mode(mode)['capture_depth']
    target = image_mode.select_capture_pixel_format(depth, im.get_supported_pixel_formats())
    print(f'mode {mode!r} -> capture_depth {depth} -> pixel format {target!r}')
    _common.ok(
        '12-bit pixel format applied',
        bool(im.set_pixel_format(target)),
        f'cached={im.pixel_format_cached}',
    )
    s.update_settings('image_mode', mode)
    _common.ok('image_mode recorded in the store', s.get_settings_snapshot()['image_mode'] == mode)
    print('significant_bits after the mode change:', im.significant_bits)

    # A bogus mode: refused by the resolver, or accepted by the store?
    try:
        image_mode.resolve_image_mode('37bit')
        _common.ok('bogus image mode refused by resolve_image_mode', False)
    except Exception as e:
        _common.ok(
            'bogus image mode refused by resolve_image_mode', True, f'{type(e).__name__}: {e}'
        )
    s.update_settings('image_mode', '37bit')
    _common.void(
        'bogus image mode refused by the store',
        False,
        f'stored {s.get_settings_snapshot()["image_mode"]!r}',
    )
    s.update_settings('image_mode', mode)

    # --- TWO ANSWERERS, not a stale mirror ------------------------------
    # This probe used to demand that settings follow a camera apply. That
    # write-through was refuted twice by execution and then on principle by
    # Eric ("SSOT"): a binning change always issues a frame-size apply, so
    # recomputing the native ROI there re-introduces #683, and
    # _set_binning_size_impl refreshes by a READ that bypasses the commit
    # chokepoint entirely. The defect is that config_helpers answers a
    # question ImagingAPI already owns -- so what this records is that the
    # API knows the geometry, which is what makes the second answerer a
    # duplicate rather than the API being ignorant.
    before = s.get_settings_snapshot()
    im.set_frame_size(1024, 768)
    after = s.get_settings_snapshot()
    _common.ok(
        'the API knows the delivered frame size after an apply',
        (im.get_width(), im.get_height()) != (0, 0),
        f'API {im.get_width()}x{im.get_height()}, settings {after["frame"]}',
    )
    im.set_binning_size(2)
    _common.ok(
        'the API knows the delivered binning after an apply',
        im.get_binning_size() == 2,
        f'API {im.get_binning_size()}, settings {s.get_settings_snapshot()["binning"]}',
    )
    print("settings['frame'] native keys:", dict(after['frame']))
except Exception:
    import traceback

    traceback.print_exc()
finally:
    s.shutdown()

# The recorded results are the probe's verdict, so they have to reach the
# exit code -- without this the slice printed FAIL and exited 0.
sys.exit(_common.report())
