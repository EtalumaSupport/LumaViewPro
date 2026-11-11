"""P9: the remaining panel capabilities -- sum, composite threshold, autofocus
flag, acquire mode, video duration, stim params, and the three focus buttons."""

import time
import traceback
import harness
from tests.stock_channels import LED0

s, live = harness.make_session('p9')
sc = s.scope


def show(label, v):
    print(f'{label:46s} {v}', flush=True)


try:
    lay = LED0
    # --- settings-backed capabilities: write the one store, read it back ---
    s.update_settings(f'{lay}.sum', 4)
    s.update_settings(f'{lay}.autofocus', True)
    s.set_channel_acquire(lay, 'video')
    s.update_settings(f'{lay}.video_config.duration', 7)
    for key, value in {
        'frequency': 3,
        'pulse_width': 25,
        'pulse_count': 9,
        'illumination_ma': 77,
    }.items():
        s.update_settings(f'{lay}.stim_config.{key}', value)
    lc = s.get_channel_configs([lay])[lay]
    show('sum', lc['sum'])
    show('autofocus', lc['autofocus'])
    show('acquire', lc['acquire'])
    show('video duration', lc['video_config']['duration'])
    # A layer's stimulation is its settings' alone: the channel configs a
    # protocol is built from carry none.
    show('stim params', s.settings[lay]['stim_config'])
    from _lumascope.modules import config_helpers

    show(
        'composite blend thresholds', config_helpers.get_composite_blend_thresholds(s.settings)[lay]
    )

    # --- the three focus buttons ---
    sc.motion.home('ALL')
    time.sleep(2)
    z = sc.motion.get_current_position('Z')
    show('motion.get_current_position(Z)', z)
    s.save_channel_focus(lay, z)  # == save_focus's layer write
    show(f'settings[{LED0}][focus]', s.get_settings_snapshot()[lay]['focus'])
    proto = sc.protocols.create_protocol(input_config=s.get_sequenced_capture_config())
    show('protocol.modify_step_z_height', proto.modify_step_z_height(step_index=0, z=z + 5))
    show('step 0 Z after modify', proto.step_row(step_index=0)['Z'])
    show(
        'protocol.apply_focus_all_channel_steps',
        proto.apply_focus_all_channel_steps(channel=lay, z=z + 9),
    )
    show(f'step {LED0} Z after apply', proto.steps()[proto.steps()['channel'] == lay]['Z'].iloc[0])
    # goto_focus == a Z move to the stored focus
    show(
        'motion.move_z(focus)',
        sc.motion.move_z(z + 9),
    )
    show('Z after goto', sc.motion.get_current_position('Z'))
    harness.assert_no_ui()
except Exception:
    traceback.print_exc()
    # Re-raised: a probe that crashed has no verdict, and exiting 0
    # here read as a pass.
    raise
finally:
    s.shutdown()
