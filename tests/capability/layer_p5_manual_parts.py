"""P5: which STEP of ui/composite_capture.py::_live_capture_impl has no API path?"""

import traceback
import harness

s, live = harness.make_session('p5')
scope = s.scope


def step(label, fn):
    try:
        print(f'{label:52s} OK   ', fn(), flush=True)
    except Exception as e:
        print(f'{label:52s} FAIL {type(e).__name__}: {e}', flush=True)


try:
    import modules.common_utils as cu
    from modules import config_helpers

    step('runtime_state.get_well_label()', lambda: scope.runtime_state.get_well_label())
    step(
        'config_helpers.get_image_capture_config_from_settings',
        lambda: config_helpers.get_image_capture_config_from_settings(s.get_settings_snapshot()),
    )
    step(
        'session.get_layer_configs([Blue])',
        lambda: sorted(s.get_layer_configs(['Blue'])['Blue'].keys()),
    )
    step(
        'cu.resolve_channel_identity(illum, None)',
        lambda: cu.resolve_channel_identity(scope.illumination, None),
    )
    step(
        'cu.build_step_name(...)',
        lambda: cu.build_step_name(
            cu.StepNameComponents(custom_prefix='A1_Blue', turret_position=None)
        ),
    )
    step(
        'cu.get_opened_layer(<needs ctx.image_settings widget>)', lambda: cu.get_opened_layer(None)
    )
    # the two overlay flags
    try:
        import modules.scope_display  # noqa

        print('overlay flags below ui/: modules.scope_display EXISTS')
    except ImportError as e:
        print('overlay flags below ui/: NONE --', e)
    hits = [a for a in dir(scope.imaging) if 'bullseye' in a.lower() or 'crosshair' in a.lower()]
    print('imaging API overlay attrs:', hits or 'NONE')
    hits2 = [a for a in dir(s) if 'bullseye' in a.lower() or 'crosshair' in a.lower()]
    print('session overlay attrs:', hits2 or 'NONE')
    harness.assert_no_ui()
except Exception:
    traceback.print_exc()
finally:
    s.shutdown()
