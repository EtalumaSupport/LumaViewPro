"""P3: gain, exposure, auto-gain through ImagingAPI, read back."""

import traceback
import harness

s, live = harness.make_session('p3')
im = s.scope.imaging


def step(label, fn):
    try:
        print(label, '->', fn(), flush=True)
    except Exception as e:
        print(label, '-> RAISED', type(e).__name__, e, flush=True)


try:
    im.start_streaming()
    step('set_gain_db(7.5)', lambda: (im.set_gain_db(7.5), im.get_gain_db(), im.gain_db_cached))
    step(
        'set_exposure_ms(123.0)',
        lambda: (im.set_exposure_ms(123.0), im.get_exposure_ms(), im.exposure_ms_cached),
    )
    step('set_gain_db(1e6)', lambda: (im.set_gain_db(1e6), im.get_gain_db()))
    print('camera_supports_auto_gain:', s.scope.capabilities.camera_supports_auto_gain, flush=True)
    ag = s.get_auto_gain_settings()
    print('session.get_auto_gain_settings():', ag, flush=True)
    step('set_auto_gain(True, ag)', lambda: im.set_auto_gain(True, ag))
    step('lock_auto_gain()', lambda: im.lock_auto_gain())
    step(
        'apply_layer_camera_settings(Blue)',
        lambda: im.apply_layer_camera_settings(
            layer='Blue', gain_db=3.0, exposure_ms=50.0, auto_gain=False, auto_gain_settings=None
        ),
    )
    print('after apply: gain', im.get_gain_db(), 'exp', im.get_exposure_ms(), flush=True)
    harness.assert_no_ui()
except Exception:
    traceback.print_exc()
finally:
    s.shutdown()
