"""Probe 00 -- does a headless session come up at all, and do the
session's video close members answer?"""

from harness import make_session, banner

session, live = make_session('smoke', home=True)
try:
    banner('session up')
    print('live_folder      :', live)
    print('is_protocol_running:', session.is_protocol_running)
    print('exclusive_activity :', session.exclusive_activity)

    banner('video close members')
    print('close_drain_pending:', session.close_drain_pending)
    print('close_drain_frames :', session.close_drain_frames)
    session.discard_close_drain()
    print('discard_close_drain -> OK')

    banner('engine members (what the session reads)')
    eng = session.sequenced_capture_runner
    print('type(engine.video_drain_busy)    :', type(eng.video_drain_busy))
    print('type(engine.video_pending_writes):', type(eng.video_pending_writes))
finally:
    session.shutdown()
print('\nPROBE 00 DONE')
