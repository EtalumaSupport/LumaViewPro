"""Probe 04 -- edit a protocol, navigate to a step, tile it, pick a plate.

All headless: Protocol / ScopeSession only, no ui. import.
"""

from harness import make_session, banner

session, live = make_session('edit', home=True)
try:
    import modules.config_helpers as config_helpers
    from modules.tiling_config import TilingConfig

    settings = session.settings
    banner('base protocol')
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
    print('steps:', p.num_steps())

    banner('ADD a step (insert_step)')
    pos = session.get_current_plate_position()
    obj_id, _ = session.get_current_objective_info()
    lc = config_helpers.get_layer_configs(settings, specific_layers=['BF'])['BF']
    lc['acquire'] = 'image'
    idx = p.insert_step(
        step_name=None,
        layer='BF',
        layer_config=lc,
        stim_configs=config_helpers.get_stim_configs(settings),
        plate_position=pos,
        objective_id=obj_id,
        before_step=None,
        after_step=0,
    )
    print('insert -> idx', idx, 'steps now', p.num_steps())

    banner('EDIT a step (modify_step)')
    lc2 = dict(lc)
    lc2['gain_db'] = 7.5
    lc2['exposure_ms'] = 42.0
    p.modify_step(
        step_idx=1,
        label='probe_renamed',
        layer='BF',
        layer_config=lc2,
        stim_configs=config_helpers.get_stim_configs(settings),
        plate_position=pos,
        objective_id=obj_id,
    )
    s = p.step(idx=1)
    print('step1 Name/Gain/Exposure:', s['Name'], s['Gain'], s['Exposure'])
    print('ASSERT edit landed:', 'PASS' if (s['Gain'] == 7.5 and s['Exposure'] == 42.0) else 'FAIL')

    banner('DELETE a step')
    before = p.num_steps()
    p.delete_step(step_idx=1)
    print(f'{before} -> {p.num_steps()}', 'PASS' if p.num_steps() == before - 1 else 'FAIL')

    banner('NAVIGATE to a step -- is there an API for it?')
    names = [n for n in dir(session) if 'step' in n.lower()]
    print('session members containing "step":', names)
    from modules.protocol_runner import ProtocolRunner

    print(
        'ProtocolRunner members containing "step":',
        [n for n in dir(ProtocolRunner) if 'step' in n.lower()],
    )
    # What a script must do instead: read the step and drive motion itself.
    st = p.step(idx=0)
    print('step0 X/Y/Z/Color/Objective:', st['X'], st['Y'], st['Z'], st['Color'], st['Objective'])
    session.scope.motion.move_absolute(
        axis='X', position=st['X'], frame='plate', wait_until_complete=True
    )
    session.scope.motion.move_absolute(
        axis='Y', position=st['Y'], frame='plate', wait_until_complete=True
    )
    session.scope.motion.move_absolute(axis='Z', position=st['Z'], wait_until_complete=True)
    now = session.get_current_plate_position()
    print('landed at:', now)
    print(
        'ASSERT stage at step:',
        'PASS' if abs(now['x'] - st['X']) < 0.05 and abs(now['y'] - st['Y']) < 0.05 else 'FAIL',
    )

    banner('SET TILING')
    # TilingConfig takes its catalogue path; the API owns it.
    # tiling_configs_path() is a METHOD on ProtocolsAPI, not a property.
    tc = TilingConfig(tiling_configs_file_loc=session.scope.protocols.tiling_configs_path())
    print('tiling catalogue:', tc.available_configs())
    print('no-tiling label :', tc.no_tiling_label())
    before = p.num_steps()
    status = p.apply_tiling(
        tiling='2x2',
        frame_dimensions=config_helpers.get_frame_dimensions_from_settings(settings),
        binning_size=config_helpers.get_binning_from_settings(settings),
        curr_step_idx=0,
        axes_config=session.scope.motion.get_axes_config(),
        labware=config_helpers.get_selected_labware_from_settings(
            settings, session.wellplate_loader
        )[1],
        stage_offset=settings['stage_offset'],
        overlap_percent=settings['tiling_overlap_percent'],
        capabilities=session.scope.capabilities,
    )
    print('tile status:', status, f'{before} -> {p.num_steps()} steps')
    print('ASSERT tiled 4x:', 'PASS' if p.num_steps() == before * 4 else f'CHECK ({p.num_steps()})')
    print('Tile column:', p.steps()['Tile'].tolist()[:6])

    banner('LABWARE / PLATE SELECTION')
    plates = session.wellplate_loader.get_plate_list()
    print('plates offered:', len(plates), plates[:4])
    before_plate = settings['protocol']['labware']
    target = next(x for x in plates if x != before_plate)
    ok = session.select_labware(target)
    after_plate = settings['protocol']['labware']
    print(f'{before_plate!r} -> select_labware({target!r}) = {ok} -> {after_plate!r}')
    print('ASSERT plate changed:', 'PASS' if after_plate == target else 'FAIL')
    try:
        session.select_labware('No Such Plate')
        print('bogus plate: NOT REFUSED (FAIL)')
    except Exception as e:
        print(f'bogus plate refused: {type(e).__name__}: {e}')
finally:
    session.shutdown()
print('\nPROBE 04 DONE')
