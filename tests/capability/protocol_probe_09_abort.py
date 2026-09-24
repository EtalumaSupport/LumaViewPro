"""Probe 09 -- stop a running protocol from a script (the Stop half of the
run buttons: ui/protocol_settings.py _cleanup_at_end_of_protocol ->
reset(run), ui/zstack.py likewise). A stop names the run by the handle its
call returned."""

import datetime
import time

from harness import make_session, banner

session, live = make_session('abort', home=True)
try:
    import modules.config_helpers as config_helpers
    from modules.protocol_runner import ProtocolRunner
    from modules.run_outcome import PendingRunOutcome

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
    p = session.scope.protocols.create_protocol(input_config=cfg)
    p.modify_time_params(
        period=datetime.timedelta(seconds=5), duration=datetime.timedelta(seconds=120)
    )

    banner('start a long protocol, then stop it')
    out = runner.run_protocol(
        p,
        sequence_name='abortme',
        image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
    )
    time.sleep(3)
    print('is_running:', runner.is_running(), '| trigger:', runner.run_trigger_source())
    print('is_live_run(out):', runner.is_live_run(out))

    banner('a stop naming a run that is not the live one must be refused')
    try:
        runner.abort(PendingRunOutcome())
        print('NOT REFUSED (FAIL)')
    except Exception as e:
        print(f'refused: {type(e).__name__}: {getattr(e, "reason", "")}: {e}')

    banner('the run is stopped by its own handle')
    runner.abort(out)
    s = out.wait(timeout_s=120)
    print('status:', s.status, s.reason)
    print('idle  :', runner.wait_for_run_idle(timeout_s=60))
    print('ASSERT stopped:', 'PASS' if not runner.is_running() else 'FAIL')
finally:
    session.shutdown()
print('\nPROBE 09 DONE')
