"""Probe 05 -- run a FULL protocol headlessly, and autofocus all steps.

Full protocol: ProtocolRunner.run_protocol (named API method).
Autofocus All Steps: ProtocolRunner.run_autofocus_all_steps, which writes
the focused Z into the protocol it is given.
"""

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

    settings = session.settings
    runner = ProtocolRunner(session)

    cfg = config_helpers.get_sequenced_capture_config_from_settings(
        settings,
        objective_helper=session.objective_helper,
        wellplate_loader=session.wellplate_loader,
        current_z=session.get_current_plate_position()['z'],
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
    # Until the run has let go of the scope AND its files have landed.
    while time.time() < deadline and session.run_lockout:
        time.sleep(0.5)
    imgs = images_under(run_dir)
    print('run_dir:', run_dir)
    print('images :', len(imgs), [p.name for p in imgs][:6])
    print('ASSERT >=2 scans worth of images:', 'PASS' if len(imgs) >= 2 else f'FAIL ({len(imgs)})')

    runner.wait_for_run_idle(timeout_s=60)

    banner('B. AUTOFOCUS ALL STEPS through ProtocolRunner.run_autofocus_all_steps')
    z_before = session.scope.motion.get_current_position('Z')
    z_in = protocol.steps()['Z'].tolist()
    af = runner.run_autofocus_all_steps(protocol).wait(timeout_s=300)
    print('status:', af.status, af.reason, 'focus_written:', af.focus_written)
    z_after = session.scope.motion.get_current_position('Z')
    print('Z before/after:', z_before, z_after)
    print('protocol step Z values before/after AF:', z_in, protocol.steps()['Z'].tolist())
    print('ASSERT AF run completed:', 'PASS' if af.status == 'completed' else 'FAIL')
    print('ASSERT focus written:', 'PASS' if af.focus_written is True else 'FAIL')

    banner('C. does a headless run move the TURRET per step?')
    import modules.protocol_step_runner as psr
    import modules.protocol_run_loop as prl

    src = pathlib.Path(psr.__file__).read_text() + pathlib.Path(prl.__file__).read_text()
    print("'turret' in the run modules:", 'turret' in src.lower())
    print('has_turret on this sim scope:', session.scope.capabilities.has_turret)
finally:
    session.shutdown()
print('\nPROBE 05 DONE')
