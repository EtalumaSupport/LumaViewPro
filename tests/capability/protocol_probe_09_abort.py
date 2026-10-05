"""Probe 09 -- stop a running protocol from a script (the Stop half of the
run buttons: ui/protocol_settings.py _press_panel_run -> run.stop(),
ui/zstack.py likewise). A stop goes through the handle the run's call
returned."""

import datetime
import time

from harness import make_session, banner

session, live = make_session('abort', home=True)
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
    p = session.scope.protocols.create_protocol(input_config=cfg)
    p.modify_time_params(
        period=datetime.timedelta(seconds=5), duration=datetime.timedelta(seconds=120)
    )

    banner('an earlier scan, run to its end: its handle is the stale one below')
    earlier = runner.run_single_scan(
        p,
        sequence_name='earlier',
        image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
    )
    print('earlier:', earlier.wait(timeout_s=120).status)
    deadline = time.time() + 90
    # A new run is refused while the earlier run's files still write.
    while time.time() < deadline and session.run_lockout:
        time.sleep(0.5)

    banner('start a long protocol, then stop it')
    out = runner.run_protocol(
        p,
        sequence_name='abortme',
        image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
    )
    time.sleep(3)
    print('is_protocol_running:', session.is_protocol_running, '| out.is_live:', out.is_live)

    banner('a stop through the handle of a run that is not the live one must be refused')
    try:
        earlier.stop()
        print('NOT REFUSED (FAIL)')
    except Exception as e:
        print(f'refused: {type(e).__name__}: {getattr(e, "reason", "")}: {e}')

    banner('the run is stopped by its own handle')
    out.stop()
    s = out.wait(timeout_s=120)
    print('status:', s.status, s.reason)
    print('ASSERT stopped:', 'PASS' if not session.is_protocol_running else 'FAIL')
finally:
    session.shutdown()
print('\nPROBE 09 DONE')
