"""Probe 6 -- the layer tab (accordion drawer) switch, headless.

GUI entry point: a person opening a layer drawer (ui/ui_helpers.py
LoggedAccordionItem.person_opened -> ui/image_settings.py
_show_channel_for_person), which asks session.show_channel.

The user-visible capability is "make channel X the shown one": the other
channels' LEDs go off, X lit if the shown channel was, and X's stored
gain/exposure applied to the camera. This probe does it through the API.
"""

import sys

import harness as _common

from modules.lumascope_api.illumination import LedState
from tests.stock_channels import LED1

s, live = _common.make_session()
try:
    ill, im = s.scope.illumination, s.scope.camera
    cfgs = s.get_layer_configs()
    print('layers known to the session:', sorted(cfgs))
    target = LED1 if LED1 in cfgs else sorted(cfgs)[0]
    cfg = cfgs[target]
    print(f'{target} config:', {k: cfg[k] for k in list(cfg)[:6]})

    # The drawer's ask, through the API: the shown channel lit, then moved
    s.light_shown_channel(True)
    s.show_channel(target)
    lit = [r.name for r in ill.get_led_states() if r.state is LedState.LIT]
    _common.ok('exactly the target layer is lit', lit == [target], f'lit={lit}')
    _common.ok(
        'camera took the layer gain',
        abs(im.read_gain_db() - float(cfg['gain_db'])) < 0.51,
        f'gain={im.read_gain_db()} wanted={cfg["gain_db"]}',
    )
    _common.ok(
        'camera took the layer exposure',
        abs(im.read_exposure_ms() - float(cfg['exposure_ms'])) < 0.51,
        f'exp={im.read_exposure_ms()} wanted={cfg["exposure_ms"]}',
    )
    s.light_shown_channel(False)

    # Is there a single API call for "make this layer active"?
    names = [
        n for n in dir(s) + dir(ill) + dir(im) if not n.startswith('_') and ('layer' in n.lower())
    ]
    print('session/API members mentioning "layer":', sorted(set(names)))

    # Range refusal on the per-layer values the drawer renders
    limits = s.scope.capabilities.camera_limits
    print('max gain / max exposure:', limits.gain_max_db, limits.exposure_max_ms)
    # The camera's range refuses each of them (CameraSettingOutOfRangeError).
    for fn, bad, label in (
        (im.set_gain_db, 999.0, 'gain 999 dB'),
        (im.set_exposure_ms, 1e7, 'exposure 10,000,000 ms'),
        (im.set_gain_db, -50.0, 'gain -50 dB'),
    ):
        try:
            r = fn(bad)
            _common.ok(f'{label} refused', False, f'returned {r}')
        except Exception as e:
            _common.ok(f'{label} refused', True, f'{type(e).__name__}: {e}')
    print('gain/exposure after the bad writes:', im.read_gain_db(), im.read_exposure_ms())
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
