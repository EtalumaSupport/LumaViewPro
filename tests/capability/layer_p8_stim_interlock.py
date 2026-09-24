"""P8: the stim/acquire interlock -- can a headless caller construct the
combination the GUI makes unrepresentable (stim enabled AND acquire set)?

Two voids, both measured by running the pair through the API:
  * nothing below ui/ refuses the pair -- it is stored, assembled, built into
    a protocol, accepted by prepare() and run to completion;
  * nothing below ui/ executes stimulation at all.
Stimulation is not available in the GUI either until the new firmware ships
(Eric, 2026-09-21), so the second void is expected to stay open until then.
"""

import pathlib
import sys
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
    cfg = config_helpers.get_composite_capture_config_from_settings(
        s.settings, s.objective_helper, position=s.get_current_plate_position()
    )
    proto = s.scope.protocols.create_protocol(input_config=cfg)
    steps = proto.steps()
    row = steps[steps['Color'] == lay]
    harness.check('a script can build a protocol from settings holding the pair', len(row) > 0)
    _acq = row['Acquire'].iloc[0] if len(row) else None
    _stim = row['Stim_Config'].iloc[0].get(lay) if len(row) else None
    print(f'4: protocol step for Blue -> Acquire={_acq!r} Stim_Config={_stim!r}')

    # run it and see whether prepare() refuses the illegal pair
    s.scope.motion.home('ALL')
    time.sleep(2)
    s.scope.imaging.start_streaming()
    runner = s.create_protocol_runner()
    refused = False
    try:
        runner.start_composite(sequence_name='stimpair', parent_dir=pathlib.Path(live) / 'C')
        print('5: prepare()/start_composite with the illegal pair -> ACCEPTED')
        print('5: outcome', runner.wait_for_completion(timeout=180).status)
    except Exception as e:
        refused = True
        print('5: REFUSED', type(e).__name__, e)
    harness.void(
        'prepare() refuses a layer set to both acquire and stimulate',
        refused,
        'the interlock lives only in the GUI; the API accepts and runs the pair',
    )

    executors = [
        m for m in ('fire_stim', 'run_stim', 'stimulate') if hasattr(s.scope.illumination, m)
    ]
    print('6: any stim EXECUTOR below ui/?', executors or 'NONE')
    harness.void(
        'a stimulation executor exists below ui/',
        bool(executors),
        'none until the new firmware ships; the GUI has no stimulation either',
    )
    harness.assert_no_ui()
except Exception:
    traceback.print_exc()
    harness.check('probe completed without an unexpected raise', False)
finally:
    s.shutdown()
sys.exit(harness.report())
