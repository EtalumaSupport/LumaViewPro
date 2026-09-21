"""P9: the remaining panel capabilities -- sum, composite threshold, autofocus
flag, acquire mode, video duration, stim params, and the three focus buttons."""

import time
import traceback
import harness

s, live = harness.make_session('p9')
sc = s.scope


def show(label, v):
    print(f'{label:46s} {v}', flush=True)


try:
    lay = 'Blue'
    # --- settings-backed capabilities: write the one store, read it back ---
    with s.settings_lock:
        s.settings[lay]['sum'] = 4
        s.settings[lay]['composite_brightness_threshold'] = 42.0
        s.settings[lay]['autofocus'] = True
        s.settings[lay]['acquire'] = 'video'
        s.settings[lay]['video_config']['duration'] = 7
        s.settings[lay]['stim_config'].update(
            {'frequency': 3, 'pulse_width': 25, 'pulse_count': 9, 'illumination_ma': 77}
        )
    lc = s.get_layer_configs([lay])[lay]
    show('sum', lc['sum'])
    show('autofocus', lc['autofocus'])
    show('acquire', lc['acquire'])
    show('video duration', lc['video_config']['duration'])
    show('stim params', lc['stim_config'])
    from modules import config_helpers

    show(
        'composite blend thresholds', config_helpers.get_composite_blend_thresholds(s.settings)[lay]
    )

    # --- the three focus buttons ---
    sc.motion.home('ALL')
    time.sleep(2)
    z = sc.motion.get_current_position('Z')
    show('motion.get_current_position(Z)', z)
    with s.settings_lock:
        s.settings[lay]['focus'] = z  # == save_focus's layer write
    show('settings[Blue][focus]', s.get_settings_snapshot()[lay]['focus'])
    proto = sc.protocols.create_protocol(input_config=s.get_sequenced_capture_config())
    show('protocol.modify_step_z_height', proto.modify_step_z_height(step_idx=0, z=z + 5))
    show('step 0 Z after modify', proto.step(idx=0)['Z'])
    show(
        'protocol.apply_focus_all_layer_steps',
        proto.apply_focus_all_layer_steps(layer=lay, z=z + 9),
    )
    show('step Blue Z after apply', proto.steps()[proto.steps()['Color'] == lay]['Z'].iloc[0])
    # goto_focus == a Z move to the stored focus
    show(
        'motion.move_absolute(Z, focus)',
        sc.motion.move_absolute('Z', z + 9, wait_until_complete=True),
    )
    show('Z after goto', sc.motion.get_current_position('Z'))
    harness.assert_no_ui()
except Exception:
    traceback.print_exc()
finally:
    s.shutdown()
