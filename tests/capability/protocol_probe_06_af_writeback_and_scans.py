"""Probe 06 -- two follow-ups from probe 05.

1. Does a headless 'Autofocus All Steps' get the focused Z back? The
   run writes it into the caller's protocol before run_complete is sent.
2. Multi-scan: how many scans does a period/duration pair actually run?
"""

import datetime
import pathlib
import time

from harness import make_session, banner

session, live = make_session('afwb', home=True)
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

    banner('1. AF-all-steps write-back')
    seen = {}

    def on_complete(**kw):
        seen['z_on_mine'] = protocol.steps()['Z'].tolist()

    z_in = protocol.steps()['Z'].tolist()
    af = runner.run_autofocus_all_steps(protocol, callbacks={'run_complete': on_complete}).wait(
        timeout_s=300
    )
    print('status / focus_written :', af.status, af.focus_written)
    print('Z handed in            :', z_in)
    print('Z on MY protocol at run_complete:', seen.get('z_on_mine'))
    print('Z on MY protocol after :', protocol.steps()['Z'].tolist())
    print(
        'ASSERT the focused Z is on the caller protocol by run_complete:',
        'PASS'
        if af.focus_written and seen.get('z_on_mine') == protocol.steps()['Z'].tolist()
        else 'FAIL',
    )
    runner.wait_for_run_idle(timeout_s=60)

    banner('2. multi-scan full protocol: period 5s, duration 20s')
    protocol.modify_time_params(
        period=datetime.timedelta(seconds=5), duration=datetime.timedelta(seconds=20)
    )
    out = runner.run_protocol(
        protocol,
        sequence_name='multi',
        image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
    )
    print('remaining_scans at start:', runner.remaining_scans())
    print('protocol_interval       :', runner.protocol_interval())
    s = out.wait(timeout_s=300)
    print('status:', s.status, s.reason)
    rd = runner.run_dir()
    deadline = time.time() + 90
    # Until the run has let go of the scope AND its files have landed.
    while time.time() < deadline and session.run_lockout:
        time.sleep(0.5)
    imgs = sorted(p.name for p in pathlib.Path(rd).rglob('*') if p.suffix.lower() == '.tiff')
    print('images:', len(imgs), imgs[:8])
    print('ASSERT multiple scans:', 'PASS' if len(imgs) > 1 else f'FAIL ({len(imgs)})')
finally:
    session.shutdown()
print('\nPROBE 06 DONE')
