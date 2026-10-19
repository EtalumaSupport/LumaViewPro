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
im_camera = s.scope.camera
try:
    caps = s.scope.capabilities
    native = caps.camera_limits.frame_max
    print('largest frame         :', native)
    print('available binning     :', caps.camera_binning_sizes)
    print('supported pix formats :', caps.camera_pixel_formats)
    print(
        'start w/h             :',
        im_camera.read_frame_size().width,
        im_camera.read_frame_size().height,
    )

    # --- frame size, in range -------------------------------------------
    delivered = im_camera.set_frame_size(800, 600)
    _common.check(
        'frame 800x600 applied', delivered == {'width': 800, 'height': 600}, str(delivered)
    )
    _common.check(
        'frame read-back',
        (im_camera.read_frame_size().width, im_camera.read_frame_size().height) == (800, 600),
        f'{im_camera.read_frame_size().width}x{im_camera.read_frame_size().height}',
    )
    print('frame_size_cached     :', im_camera.frame_size.to_dict())

    # --- frame size, OUT OF RANGE (the 13414 shape) ----------------------
    from modules.exceptions import CameraSettingOutOfRangeError, CameraSettingUnsupportedError

    try:
        out = im_camera.set_frame_size(13414, 13414)
        print('set_frame_size(13414,13414) RETURNED:', out)
        _common.check(
            '13414 refused (not silently clamped)',
            False,
            f'clamped to {out} and returned as delivered -- no raise',
        )
    except CameraSettingOutOfRangeError as e:
        _common.check('13414 refused (not silently clamped)', True, str(e))
    print(
        'after oversize, read-back:',
        im_camera.read_frame_size().width,
        im_camera.read_frame_size().height,
    )

    # --- frame size, zero / negative ------------------------------------
    for bad in ((0, 0), (-10, 600)):
        try:
            out = im_camera.set_frame_size(*bad)
            print(f'set_frame_size{bad} RETURNED:', out)
        except Exception as e:
            print(f'set_frame_size{bad} raised {type(e).__name__}: {e}')

    # --- binning ---------------------------------------------------------
    supported = caps.camera_binning_sizes
    im_camera.set_binning(2)
    _common.ok(
        'binning 2 applied',
        im_camera.read_binning() == 2,
        f'read-back={im_camera.read_binning()}',
    )
    bad_bin = next(x for x in (3, 5, 7, 16) if x not in supported)
    try:
        im_camera.set_binning(bad_bin)
        _common.check(
            f'unsupported binning {bad_bin} refused',
            False,
            f'accepted; read-back {im_camera.read_binning()}',
        )
    except CameraSettingUnsupportedError as e:
        _common.check(f'unsupported binning {bad_bin} refused', True, f'{e.reason}: {e}')
    im_camera.set_binning(1)

    # --- pixel format, chosen by the image mode -------------------------
    from modules import image_mode

    fmts = caps.camera_pixel_formats
    if fmts:
        chosen = image_mode.select_capture_pixel_format(12, fmts)
        im_camera.set_image_mode('12bit_scientific')
        _common.ok(
            f'pixel format {chosen} applied by the full-depth mode',
            im_camera.pixel_format == chosen,
            f'read-back={im_camera.pixel_format}',
        )
    try:
        im_camera.set_image_mode('Mono99')
        _common.ok('bogus image mode refused', False, 'accepted')
    except Exception as e:
        _common.ok('bogus image mode refused', True, f'{type(e).__name__}: {e}')

    # --- is the image MODE (8/12-bit display mode) an API concept? -------
    snap = s.get_settings_snapshot()
    print("settings['image_mode'] :", snap.get('image_mode'))
    print('camera has set_image_mode?', hasattr(im_camera, 'set_image_mode'))
finally:
    s.shutdown()

# The recorded results are the probe's verdict, so they have to reach the
# exit code -- without this the slice printed FAIL and exited 0.
sys.exit(_common.report())
