"""P7: composite capture through ProtocolRunner.start_composite (no GUI)."""

import pathlib
import traceback
import harness

s, live = harness.make_session('p7')
try:
    # two channels acquiring an image is the composite's own precondition
    with s.settings_lock:
        for lay in ('Blue', 'Green'):
            s.settings[lay]['acquire'] = 'image'
            s.settings[lay]['illumination_ma'] = 50
    runner = s.create_protocol_runner()
    parent = pathlib.Path(live) / 'Manual' / 'Composites'
    # (a) not streaming -- does the ENGINE refuse, or only the GUI pre-check?
    print(
        'streaming?', s.scope.imaging.is_streaming(), 'active_cached', s.scope.imaging.active_cached
    )
    try:
        out = runner.start_composite(sequence_name='composite', parent_dir=parent)
        print('a: start_composite while NOT streaming -> ACCEPTED', out)
        res = runner.wait_for_completion(timeout=90)
        print('a: outcome', res)
    except Exception as e:
        print('a: start_composite while NOT streaming -> REFUSED', type(e).__name__, e)
    # (b) streaming
    s.scope.imaging.start_streaming()
    try:
        out = runner.start_composite(sequence_name='composite2', parent_dir=parent)
        print('b: start_composite streaming -> ACCEPTED', out)
        res = runner.wait_for_completion(timeout=120)
        print('b: outcome', res)
    except Exception as e:
        print('b: REFUSED', type(e).__name__, e)
    import time

    time.sleep(3)
    files = sorted(str(p.relative_to(live)) for p in live.rglob('*') if p.is_file())
    print('files produced:', files[:25], '...' if len(files) > 25 else '', 'count', len(files))
    print('composites:', [f for f in files if 'omposite' in f][:10])
    harness.assert_no_ui()
except Exception:
    traceback.print_exc()
finally:
    s.shutdown()
