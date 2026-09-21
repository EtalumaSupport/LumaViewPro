"""Probe 00 -- does a headless session come up at all, and do the two
flagged ProtocolRunner facade methods work?"""

from harness import make_session, banner

session, live = make_session('smoke', home=True)
try:
    from modules.protocol_runner import ProtocolRunner

    runner = ProtocolRunner(session)
    banner('session up')
    print('live_folder      :', live)
    print('is_running       :', runner.is_running())
    print('run_trigger_src  :', runner.run_trigger_source())

    banner('flagged facade methods')
    for name in ('video_drain_busy', 'video_pending_writes', 'discard_video_pending'):
        try:
            v = getattr(runner, name)()
            print(f'{name:22s} -> OK {v!r}')
        except Exception as e:
            print(f'{name:22s} -> {type(e).__name__}: {e}')

    banner('engine members (what the facade calls)')
    eng = session.sequenced_capture_runner
    print('type(engine.video_drain_busy)    :', type(eng.video_drain_busy))
    print('type(engine.video_pending_writes):', type(eng.video_pending_writes))
finally:
    session.shutdown()
print('\nPROBE 00 DONE')
