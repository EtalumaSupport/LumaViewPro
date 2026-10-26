"""P3: gain, exposure, auto-gain through ImagingAPI, read back."""

import traceback
import harness
from tests.stock_channels import LED0

s, live = harness.make_session('p3')
im = s.scope.imaging
im_camera = s.scope.camera


def step(label, fn):
    try:
        print(label, '->', fn(), flush=True)
    except Exception as e:
        print(label, '-> RAISED', type(e).__name__, e, flush=True)


try:
    im.enable_streaming()
    step(
        'set_gain_db(7.5)',
        lambda: (im_camera.set_gain_db(7.5), im_camera.read_gain_db(), im_camera.gain_db),
    )
    step(
        'set_exposure_ms(123.0)',
        lambda: (
            im_camera.set_exposure_ms(123.0),
            im_camera.read_exposure_ms(),
            im_camera.exposure_ms,
        ),
    )
    step('set_gain_db(1e6)', lambda: (im_camera.set_gain_db(1e6), im_camera.read_gain_db()))
    print('camera_supports_auto_gain:', s.scope.capabilities.camera_supports_auto_gain, flush=True)
    ag = s.get_auto_gain_settings()
    print('session.get_auto_gain_settings():', ag, flush=True)
    step('set_auto_gain(True, ag)', lambda: im_camera.set_auto_gain(True, ag))
    step('lock_auto_gain()', lambda: im_camera.lock_auto_gain())
    step(
        f'camera.apply({LED0})',
        lambda: im_camera.apply(
            channel=LED0, gain_db=3.0, exposure_ms=50.0, auto_gain=False, auto_gain_settings=None
        ),
    )
    print(
        'after apply: gain',
        im_camera.read_gain_db(),
        'exp',
        im_camera.read_exposure_ms(),
        flush=True,
    )
    harness.assert_no_ui()
except Exception:
    traceback.print_exc()
    # Re-raised: a probe that crashed has no verdict, and exiting 0
    # here read as a pass.
    raise
finally:
    s.shutdown()
