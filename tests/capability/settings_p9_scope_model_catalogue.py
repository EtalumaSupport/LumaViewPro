"""Probe 9 -- the scope-model picker's catalogue, headless.

GUI entry point: ui/advanced_settings.py:484-485 (its OWN inline kv) ->
load_scopes() :296 (reads MicroscopeSettings.scopes, loaded by
ui/microscope_settings.py:146-173) and select_scope() :300.
"""

import sys

import harness as _common

s, live = _common.make_session()
try:
    from modules.layer_record import load_scope_models

    models = load_scope_models()
    print('models a script can list:', sorted(models))
    _common.ok('the catalogue is reachable below the GUI', bool(models))
    print(
        'same load duplicated in ui/microscope_settings.py:146-173 '
        "(json.load(...)['Models'] + _validate_scopes)"
    )
    # selecting one
    s.update_settings('microscope', sorted(models)[0])
    _common.ok(
        'model selection recorded', s.get_settings_snapshot()['microscope'] == sorted(models)[0]
    )
    s.update_settings('microscope', 'NotAScope')
    _common.void(
        'a model outside the catalogue is refused',
        False,
        f'stored {s.get_settings_snapshot()["microscope"]!r} -- neither '
        'update_settings nor the GUI checks membership',
    )
    # capabilities still follow the hardware, not the stored model
    print(
        'capabilities after a bogus model: '
        f'has_turret={s.scope.capabilities.has_turret} '
        f'has_xy_stage={s.scope.capabilities.has_xy_stage}'
    )
except Exception:
    import traceback

    traceback.print_exc()
finally:
    s.shutdown()

# The recorded results are the probe's verdict, so they have to reach the
# exit code -- without this the slice printed FAIL and exited 0.
sys.exit(_common.report())
