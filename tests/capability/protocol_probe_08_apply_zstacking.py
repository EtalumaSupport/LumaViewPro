"""Probe 08 -- 'Apply Z-Stacking' (ui/lumaviewpro.kv:1428 -> apply_zstacking,
ui/protocol_settings.py:639): expand every step of a protocol into slices.

Distinct from the z-stack Acquire button (one field, one layer, run now).
"""

from harness import make_session, banner

ZS = {'step_size': 5.0, 'range': 20.0, 'position': 'Current Position at Center'}
session, live = make_session('applyzs', home=True, zstack=dict(ZS))
try:
    import modules.config_helpers as config_helpers

    settings = session.settings
    cfg = config_helpers.get_sequenced_capture_config_from_settings(
        settings,
        objective_helper=session.objective_helper,
        wellplate_loader=session.wellplate_loader,
        tiling='1x1',
        use_zstacking=False,
    )
    cfg['layer_configs'] = {'BF': cfg['layer_configs']['BF']}
    cfg['layer_configs']['BF']['acquire'] = 'image'
    cfg['labware_id'] = 'Center Plate'
    p = session.scope.protocols.create_protocol(input_config=cfg)
    before = p.num_steps()

    banner('apply_zstacking')
    zp = config_helpers.get_zstack_params_from_settings(settings)
    print('zstack params:', zp)
    status = p.apply_zstacking(zstack_params=zp, axes_config=session.scope.motion.get_axes_config())
    print('status:', status, f'{before} -> {p.num_steps()} steps')
    print('Z-Slice column:', p.steps()['Z-Slice'].tolist())
    print('Z column      :', p.steps()['Z'].tolist())
    print('ASSERT expanded:', 'PASS' if p.num_steps() > before else 'FAIL')

    banner('the zero-extent refusal the GUI owns (ui/protocol_settings.py:653-675)')
    p2 = session.scope.protocols.create_protocol(input_config=cfg)
    try:
        st = p2.apply_zstacking(
            zstack_params={'range': 0.0, 'step_size': 0.0, 'z_reference': zp['z_reference']},
            axes_config=session.scope.motion.get_axes_config(),
        )
        print(
            'NO RAISE:',
            st,
            'steps:',
            p2.num_steps(),
            '-> the range/step > 0 refusal lives ONLY in the widget',
        )
    except Exception as e:
        print(f'RAISED {type(e).__name__}: {e}')
finally:
    session.shutdown()
print('\nPROBE 08 DONE')
