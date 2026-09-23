"""Probe 09 -- stop a running protocol from a script (the Stop half of the
run buttons: ui/protocol_settings.py:2500 _cleanup_at_end_of_protocol ->
reset(requester), ui/zstack.py:134 likewise)."""

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

    banner('a NON-owner reset must be refused')
    try:
        runner.abort(requester='zstack')
        print('NOT REFUSED (FAIL)')
    except Exception as e:
        print(f'refused: {type(e).__name__}: {e}')

    banner('the owner stops it')
    runner.abort(requester='api_protocol')
    s = out.wait(timeout_s=120)
    print('status:', s.status, s.reason)
    print('idle  :', runner.wait_for_run_idle(timeout_s=60))
    print('ASSERT stopped:', 'PASS' if not runner.is_running() else 'FAIL')
finally:
    session.shutdown()
print('\nPROBE 09 DONE')
