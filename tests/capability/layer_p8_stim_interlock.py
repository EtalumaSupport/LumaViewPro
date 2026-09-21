"""P8: the stim/acquire interlock -- can a headless caller construct the
combination the GUI makes unrepresentable (stim enabled AND acquire set)?"""

import pathlib
import time
import traceback
import harness

s, live = harness.make_session('p8')
try:
    from modules import config_helpers

    lay = 'Blue'
    with s.settings_lock:
        s.settings['stimulation_enabled'] = True
        s.settings[lay]['acquire'] = 'image'
        s.settings[lay]['stim_config']['enabled'] = True  # the GUI forbids this pair
        s.settings['Green']['acquire'] = 'image'
    snap = s.get_settings_snapshot()[lay]
    print(
        f'1: stored pair -> acquire={snap["acquire"]!r} '
        f'stim.enabled={snap["stim_config"]["enabled"]!r}'
    )
    print(
        '2: session.get_layer_configs:',
        s.get_layer_configs([lay])[lay]['acquire'],
        s.get_layer_configs([lay])[lay]['stim_config']['enabled'],
    )
    print('3: session.get_enabled_stim_configs:', list(s.get_enabled_stim_configs().keys()))
    # does any builder / prepare refuse or normalise the pair?
    cfg = config_helpers.get_composite_capture_config_from_settings(
        s.settings, s.objective_helper, position=s.get_current_plate_position()
    )
    proto = s.scope.protocols.create_protocol(input_config=cfg)
    steps = proto.steps()
    row = steps[steps['Color'] == lay]
    _acq = row['Acquire'].iloc[0] if len(row) else None
    _stim = row['Stim_Config'].iloc[0].get(lay) if len(row) else None
    print(f'4: protocol step for Blue -> Acquire={_acq!r} Stim_Config={_stim!r}')
    # run it and see whether prepare() refuses the illegal pair
    s.scope.motion.home('ALL')
    time.sleep(2)
    s.scope.imaging.start_streaming()
    runner = s.create_protocol_runner()
    try:
        runner.start_composite(sequence_name='stimpair', parent_dir=pathlib.Path(live) / 'C')
        print('5: prepare()/start_composite with the illegal pair -> ACCEPTED')
        print('5: outcome', runner.wait_for_completion(timeout=180).status)
    except Exception as e:
        print('5: REFUSED', type(e).__name__, e)
    print(
        '6: any stim EXECUTOR below ui/?',
        [m for m in ('fire_stim', 'run_stim', 'stimulate') if hasattr(s.scope.illumination, m)]
        or 'NONE',
    )
    harness.assert_no_ui()
except Exception:
    traceback.print_exc()
finally:
    s.shutdown()
