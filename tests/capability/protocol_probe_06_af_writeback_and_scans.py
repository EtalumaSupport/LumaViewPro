"""Probe 06 -- two follow-ups from probe 05.

1. Does a headless 'Autofocus All Steps' get the focused Z back? The
   write-back the GUI does (ui/protocol_settings.py:1702-1703) is
   `self._protocol.steps()['Z'] = kwargs['protocol'].steps()['Z']`,
   guarded on status == 'completed'. Nothing in modules/ does it.
2. Multi-scan: how many scans does a period/duration pair actually run?
"""

import copy
import datetime
import pathlib
import time

from harness import make_session, banner

session, live = make_session('afwb', home=True)
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

    banner('1. AF-all-steps write-back')
    seen = {}

    def on_complete(**kw):
        seen['status'] = kw.get('status')
        p = kw.get('protocol')
        seen['z'] = None if p is None else p.steps()['Z'].tolist()

    sequence = copy.deepcopy(protocol)
    sequence.modify_autofocus_all_steps(enabled=True)
    z_in = sequence.steps()['Z'].tolist()
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
        callbacks={'run_complete': on_complete},
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
    runner.start(plan).wait(timeout_s=300)
    time.sleep(2)
    print('callback status      :', seen.get('status'))
    print('Z handed in          :', z_in)
    print('Z on callback protocol:', seen.get('z'))
    print('Z on MY protocol obj  :', sequence.steps()['Z'].tolist())
    changed = seen.get('z') != z_in
    print(
        'ASSERT focused Z only reachable via the run_complete callback:',
        'PASS'
        if changed and sequence.steps()['Z'].tolist() == z_in
        else f'CHECK (cb_changed={changed})',
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
    while time.time() < deadline and session.file_io_executor.is_protocol_queue_active():
        time.sleep(0.5)
    imgs = sorted(p.name for p in pathlib.Path(rd).rglob('*') if p.suffix.lower() == '.tiff')
    print('images:', len(imgs), imgs[:8])
    print('ASSERT multiple scans:', 'PASS' if len(imgs) > 1 else f'FAIL ({len(imgs)})')
finally:
    session.shutdown()
print('\nPROBE 06 DONE')
