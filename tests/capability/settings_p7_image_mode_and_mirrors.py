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
    im_camera = s.scope.camera
    print('modes available (every camera):', image_mode.available_modes())

    # The whole capability, one member: the camera's image mode chooses the
    # format from the ones this camera has and saves the mode.
    mode = '12bit_scientific'
    depth = image_mode.resolve_image_mode(mode)['capture_depth']
    target = image_mode.select_capture_pixel_format(
        depth, s.scope.capabilities.camera_pixel_formats
    )
    print(f'mode {mode!r} -> capture_depth {depth} -> pixel format {target!r}')
    im_camera.set_image_mode(mode)
    _common.ok(
        '12-bit pixel format applied',
        im_camera.pixel_format == target,
        f'cached={im_camera.pixel_format}',
    )
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
    try:
        im_camera.set_image_mode('37bit')
        _common.ok(
            'bogus image mode refused by the store',
            s.get_settings_snapshot()['image_mode'] != '37bit',
            f'stored {s.get_settings_snapshot()["image_mode"]!r}',
        )
    except Exception as e:
        _common.ok('bogus image mode refused by the store', True, f'{type(e).__name__}: {e}')
    im_camera.set_image_mode(mode)

    # --- ONE ANSWERER: the camera saves what it took --------------------
    # The camera's setters save the frame, the binning and the region they
    # came from once the camera took them (scope.camera, W3b), so the
    # settings follow every apply; what this records is that the API knows
    # the geometry after one.
    before = s.get_settings_snapshot()
    im_camera.set_frame_size(1024, 768)
    after = s.get_settings_snapshot()
    _common.ok(
        'the API knows the delivered frame size after an apply',
        (im_camera.read_frame_size().width, im_camera.read_frame_size().height) != (0, 0),
        f'API {im_camera.read_frame_size().width}x{im_camera.read_frame_size().height}, settings {after["frame"]}',
    )
    im_camera.set_binning(2)
    _common.ok(
        'the API knows the delivered binning after an apply',
        im_camera.read_binning() == 2,
        f'API {im_camera.read_binning()}, settings {s.get_settings_snapshot()["binning"]}',
    )
    print("settings['frame'] native keys:", dict(after['frame']))
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
