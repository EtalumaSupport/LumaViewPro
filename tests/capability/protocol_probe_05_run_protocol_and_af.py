"""Probe 05 -- run a FULL protocol headlessly, and autofocus all steps.

Full protocol: ProtocolRunner.run_protocol (named API method).
Autofocus All Steps: NO named API method exists -- run_autofocus() is a
one-position one-layer focus, not "AF every step of the loaded protocol".
This probe reproduces the GUI handler's assembly
(ui/protocol_settings.py:1897-1921) through ProtocolRunner.prepare/start.
"""

import copy
import datetime
import pathlib
import time

from harness import make_session, banner


def images_under(d):
    if not d:
        return []
    return [p for p in pathlib.Path(d).rglob('*') if p.suffix.lower() in ('.tiff', '.tif')]


session, live = make_session('runproto', home=True)
try:
    import modules.config_helpers as config_helpers
    from modules.protocol_runner import ProtocolRunner
    from modules.sequenced_capture_runner import SequencedCaptureRunMode

    settings = session.settings
    runner = ProtocolRunner(session)

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
    protocol = session.scope.protocols.create_protocol(input_config=cfg)
    print('steps:', protocol.num_steps())

    banner('A. RUN FULL PROTOCOL (2 scans, 10s period)')
    protocol.modify_time_params(
        period=datetime.timedelta(seconds=10),
        duration=datetime.timedelta(seconds=15),
    )
    print('period/duration:', protocol.period(), protocol.duration())
    outcome = runner.run_protocol(
        protocol,
        sequence_name='probe_full_protocol',
        image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
    )
    settled = outcome.wait(timeout_s=300)
    print('status:', settled.status, settled.reason)
    run_dir = runner.run_dir()
    deadline = time.time() + 90
    while time.time() < deadline and session.file_io_executor.is_protocol_queue_active():
        time.sleep(0.5)
    imgs = images_under(run_dir)
    print('run_dir:', run_dir)
    print('images :', len(imgs), [p.name for p in imgs][:6])
    print('ASSERT >=2 scans worth of images:', 'PASS' if len(imgs) >= 2 else f'FAIL ({len(imgs)})')

    runner.wait_for_run_idle(timeout_s=60)

    banner('B. AUTOFOCUS ALL STEPS -- is there a named API method?')
    print(
        'ProtocolRunner methods with "autofocus":',
        [n for n in dir(runner) if 'autofocus' in n.lower()],
    )
    import inspect

    print('run_autofocus signature:', inspect.signature(ProtocolRunner.run_autofocus))
    print(
        '-> takes a LAYER, not a protocol: it is the standalone AF button, '
        'not "Autofocus All Steps".'
    )

    banner('B2. replaying the GUI handler through prepare/start')
    z_before = session.scope.motion.get_current_position('Z')
    sequence = copy.deepcopy(protocol)
    sequence.modify_autofocus_all_steps(enabled=True)
    print('AF flags after modify_autofocus_all_steps:', sequence.steps()['Auto_Focus'].tolist())
    plan = runner.prepare(
        protocol=sequence,
        run_mode=SequencedCaptureRunMode.SINGLE_AUTOFOCUS_SCAN,
        run_trigger_source='autofocus_scan',
        max_scans=1,
        sequence_name='af_scan',
        parent_dir=None,
        image_capture_config=config_helpers.get_image_capture_config_from_settings(settings),
        enable_image_saving=False,
        autogain_settings=config_helpers.get_auto_gain_settings(settings),
        callbacks={},
        update_z_pos_from_autofocus=True,
        leds_state_at_end='off',
        engineering_mode=session.engineering_mode,
        autofocus_snapshot=config_helpers.autofocus_snapshot_from_settings(
            settings, session.settings_lock
        ),
        **config_helpers.get_sequenced_run_settings(
            settings, run_mode=SequencedCaptureRunMode.SINGLE_AUTOFOCUS_SCAN
        ),
    )
    af_outcome = runner.start(plan)
    af = af_outcome.wait(timeout_s=300)
    print('status:', af.status, af.reason)
    z_after = session.scope.motion.get_current_position('Z')
    print('Z before/after:', z_before, z_after)
    print('protocol step Z values after AF:', sequence.steps()['Z'].tolist())
    print('ASSERT AF run completed:', 'PASS' if af.status == 'completed' else 'FAIL')
    print(
        'kwargs a script had to supply by hand:',
        len(plan.__dict__) if hasattr(plan, '__dict__') else '?',
    )

    banner('C. does a headless run move the TURRET per step?')
    import modules.protocol_step_runner as psr
    import modules.protocol_run_loop as prl

    src = pathlib.Path(psr.__file__).read_text() + pathlib.Path(prl.__file__).read_text()
    print("'turret' in the run modules:", 'turret' in src.lower())
    print('has_turret on this sim scope:', session.scope.capabilities.has_turret)
finally:
    session.shutdown()
print('\nPROBE 05 DONE')
