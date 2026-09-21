"""Probe 02 -- the z-stack Acquire button vs ProtocolRunner.run_zstack.

Part A: itemise the difference by capturing the kwargs each lane hands to
        SequencedCaptureRunner.prepare(). The GUI lane is replayed here in
        its settings-side equivalent (ui/zstack.py:150-314 with every
        `*_from_ui` getter swapped for the `*_from_settings` function it
        delegates to -- see modules/config_ui_getters.py, each of which is
        a one-line adapter).
Part B: RUN run_zstack for real and assert the slices landed on disk.
"""

import pathlib
import time

from harness import make_session, banner

ZS = {'step_size': 5.0, 'range': 20.0, 'position': 'Current Position at Center'}

session, live = make_session('zstack', home=True, zstack=dict(ZS))
try:
    import pathlib as _pl

    import modules.config_helpers as config_helpers
    from modules.protocol_runner import ProtocolRunner
    from modules.sequenced_capture_runner import SequencedCaptureRunMode
    from modules.tiling_config import TilingConfig
    from modules.zstack_config import ZStackConfig

    runner = ProtocolRunner(session)
    settings = session.settings
    engine = session.sequenced_capture_runner

    captured = {}

    def capture(tag):
        """Capture prepare()'s kwargs without committing a run."""
        real = engine.prepare

        def spy(**kw):
            captured[tag] = kw
            raise _StopError()

        return real, spy

    class _StopError(Exception):
        pass

    # ---------------- Lane 1: what ui/zstack.py assembles ----------------
    banner('A1. GUI lane (ui/zstack.py) prepare kwargs')
    labware_id, _lw = config_helpers.get_selected_labware_from_settings(
        settings, session.wellplate_loader
    )
    objective_id, _ = session.get_current_objective_info()
    zstack_params = config_helpers.get_zstack_params_from_settings(settings)
    zc = ZStackConfig(
        range=zstack_params['range'],
        step_size=zstack_params['step_size'],
        current_z_reference=zstack_params['z_reference'],
        current_z_value=session.scope.motion.get_current_position('Z'),
    )
    zstack_positions_valid = zc.number_of_steps() > 0
    print('zstack positions valid:', zstack_positions_valid, 'slices:', zc.number_of_steps())

    active_layer = 'BF'
    active_layer_config = config_helpers.get_layer_configs(
        settings, specific_layers=[active_layer]
    )[active_layer]
    active_layer_config['acquire'] = 'image'
    active_layer_config['autofocus'] = False

    curr_position = session.get_current_plate_position()
    curr_position.update({'name': 'ZStack'})
    tiling_config = TilingConfig(
        tiling_configs_file_loc=_pl.Path(session.source_path) / 'data' / 'tiling.json'
    )
    gui_cfg = config_helpers.build_sequenced_capture_config(
        {
            'labware_id': labware_id,
            'positions': [curr_position],
            'objective_id': objective_id,
            'zstack_params': zstack_params,
            'use_zstacking': True,
            'tiling': tiling_config.no_tiling_label(),
            'tiling_overlap_percent': 0.0,
            'layer_configs': {active_layer: active_layer_config},
            'period': None,
            'duration': None,
            'frame_dimensions': config_helpers.get_frame_dimensions_from_settings(settings),
            'binning_size': config_helpers.get_binning_from_settings(settings),
            'stim_config': config_helpers.get_stim_configs(settings),
        }
    )
    gui_seq = session.scope.protocols.create_protocol(input_config=gui_cfg)
    print('GUI-lane protocol steps:', gui_seq.num_steps())

    gui_kwargs = dict(
        protocol=gui_seq,
        run_mode=SequencedCaptureRunMode.SINGLE_ZSTACK,
        run_trigger_source='zstack',
        max_scans=1,
        sequence_name='zstack',
        parent_dir=_pl.Path(settings['live_folder']).resolve() / 'Manual' / 'Z-Stacks',
        image_capture_config=config_helpers.get_image_capture_config_from_settings(settings),
        enable_image_saving=True,  # is_image_saving_enabled() -- GUI checkbox
        autogain_settings=config_helpers.get_auto_gain_settings(settings),
        callbacks={'ui': 'x10 GUI callbacks'},
        return_to_position=session.get_current_plate_position(),
        leds_state_at_end='return_to_original',
        engineering_mode=session.engineering_mode,
        autofocus_snapshot=config_helpers.autofocus_snapshot_from_settings(
            settings, session.settings_lock
        ),
        **config_helpers.get_sequenced_run_settings(
            settings, run_mode=SequencedCaptureRunMode.SINGLE_ZSTACK
        ),
    )
    captured['gui'] = gui_kwargs
    for k in sorted(gui_kwargs):
        print(f'  {k:28s} = {gui_kwargs[k]!r}'[:160])

    # ---------------- Lane 2: ProtocolRunner.run_zstack ----------------
    banner('A2. run_zstack prepare kwargs (spied, no run committed)')
    real_prepare = engine.prepare

    def spy(**kw):
        captured['api'] = kw
        raise _StopError()

    engine.prepare = spy
    try:
        runner.run_zstack(layer='BF')
    except _StopError:
        pass
    finally:
        engine.prepare = real_prepare
    for k in sorted(captured['api']):
        print(f'  {k:28s} = {captured["api"][k]!r}'[:160])

    banner('A3. itemised diff')
    keys = sorted(set(captured['gui']) | set(captured['api']))
    for k in keys:
        g = captured['gui'].get(k, '<ABSENT>')
        a = captured['api'].get(k, '<ABSENT>')
        if k == 'protocol':
            gs, as_ = g.num_steps(), a.num_steps()
            same = gs == as_
            print(f'  protocol.num_steps  GUI={gs} API={as_} {"SAME" if same else "DIFFER"}')
            gcols = g.steps()
            acols = a.steps()
            print('    columns equal     :', list(gcols.columns) == list(acols.columns))
            for col in gcols.columns:
                gv = gcols[col].tolist()
                av = acols[col].tolist()
                if gv != av:
                    print(f'    step col DIFFERS  : {col}: GUI={gv[:3]} API={av[:3]}')
            continue
        if repr(g) == repr(a):
            print(f'  SAME    {k}')
        else:
            print(f'  DIFFER  {k}\n            GUI: {g!r}'[:200])
            print(f'            API: {a!r}'[:200])

    # ---------------- Part B: actually run it ----------------
    banner('B. run_zstack for real, assert slices on disk')
    outcome = runner.run_zstack(layer='BF', sequence_name='probe_zstack')
    settled = outcome.wait(timeout_s=300)
    print('status :', settled.status, settled.reason)
    run_dir = runner.run_dir()
    print('run_dir:', run_dir)
    deadline = time.time() + 90
    while time.time() < deadline and session.file_io_executor.is_protocol_queue_active():
        time.sleep(0.5)
    files = sorted(p for p in pathlib.Path(run_dir).rglob('*') if p.is_file())
    imgs = [p for p in files if p.suffix.lower() in ('.tiff', '.tif')]
    print('slice files:', len(imgs))
    for p in imgs[:10]:
        print('   ', p.name, p.stat().st_size)
    print('expected slices:', zc.number_of_steps())
    print('ASSERT slices == expected:', 'PASS' if len(imgs) == zc.number_of_steps() else 'FAIL')
finally:
    session.shutdown()
print('\nPROBE 02 DONE')
