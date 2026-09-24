"""Probe 6 -- the layer tab (accordion drawer) switch, headless.

GUI entry point: ui/lumaviewpro.kv:323,331,339,347 (on_collapse) and
ui/lumaviewpro.kv:365 (panel toggle) -> ui/image_settings.py:159
set_expanded_layer / :688 _do_accordion_collapse.

The user-visible capability is "make channel X the active one": the other
channels' LEDs go off, and X's stored gain/exposure/illumination are
applied to the camera.  That reconcile loop lives in the GUI
(image_settings.py:737-752); this probe does the same thing through the API.
"""

import sys

import harness as _common

s, live = _common.make_session()
try:
    ill, im = s.scope.illumination, s.scope.imaging
    cfgs = s.get_layer_configs()
    print('layers known to the session:', sorted(cfgs))
    target = 'Green' if 'Green' in cfgs else sorted(cfgs)[0]
    cfg = cfgs[target]
    print(f'{target} config:', {k: cfg[k] for k in list(cfg)[:6]})

    # The GUI's reconcile, expressed through the API
    for layer in cfgs:
        if layer == target:
            continue
        st = ill.get_led_state(channel=layer)
        if st.get('enabled'):
            ill.led_off(layer, owner='probe')
    im.apply_layer_camera_settings(
        gain_db=float(cfg['gain_db']),
        exposure_ms=float(cfg['exposure_ms']),
        layer=target,
    )
    ill.led_on(target, float(cfg['illumination_ma']), owner='probe')
    states = ill.get_led_states()
    lit = [k for k, v in states.items() if v.get('enabled')]
    _common.ok('exactly the target layer is lit', lit == [target], f'lit={lit}')
    _common.ok(
        'camera took the layer gain',
        abs(im.get_gain_db() - float(cfg['gain_db'])) < 0.51,
        f'gain={im.get_gain_db()} wanted={cfg["gain_db"]}',
    )
    _common.ok(
        'camera took the layer exposure',
        abs(im.get_exposure_ms() - float(cfg['exposure_ms'])) < 0.51,
        f'exp={im.get_exposure_ms()} wanted={cfg["exposure_ms"]}',
    )
    ill.led_off(target, owner='probe')

    # Is there a single API call for "make this layer active"?
    names = [
        n for n in dir(s) + dir(ill) + dir(im) if not n.startswith('_') and ('layer' in n.lower())
    ]
    print('session/API members mentioning "layer":', sorted(set(names)))

    # Range refusal on the per-layer values the drawer renders
    print('max gain / max exposure cached:', im.max_gain_db_cached, im.max_exposure_ms_cached)
    # Exposure IS refused; both gain bounds are not, so they are voids and
    # exposure stays a check. One loop, per-value disposition.
    for fn, bad, label, is_void in (
        (im.set_gain_db, 999.0, 'gain 999 dB', True),
        (im.set_exposure_ms, 1e7, 'exposure 10,000,000 ms', False),
        (im.set_gain_db, -50.0, 'gain -50 dB', True),
    ):
        record = _common.void if is_void else _common.ok
        try:
            r = fn(bad)
            record(f'{label} refused', False, f'returned {r}')
        except Exception as e:
            record(f'{label} refused', True, f'{type(e).__name__}: {e}')
    print('gain/exposure after the bad writes:', im.get_gain_db(), im.get_exposure_ms())
except Exception:
    import traceback

    traceback.print_exc()
finally:
    s.shutdown()

# The recorded results are the probe's verdict, so they have to reach the
# exit code -- without this the slice printed FAIL and exited 0.
sys.exit(_common.report())
