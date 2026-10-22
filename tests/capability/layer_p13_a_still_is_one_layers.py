"""P13: a still's name, its exposure and gain, and the layer the camera held are one layer's.

Asked of the API alone, as a script or REST would, and read back from each
saved file: `manual_capture.capture` names the still for the lit LED, else
the shown layer, and the camera holds that layer's stored settings for the
frame, whatever ran before it.

1. a run, then a still: named for the layer shown before the run, at its
   stored settings (the run's end shows it again);
2. a go-to, then a still: named for the step's layer, at its settings;
3. a typed value, then a still: the shown layer's exposure written through
   `update_settings`, and the still taken at the new value;
4. a still asked while a go-to to another layer's step, elsewhere on the
   plate, is under way, at three delays: whichever layer it is named for,
   its exposure, gain and held layer are that layer's.

A layer with auto-gain on is not compared for exposure and gain (the camera
then holds what the loop reached); its name and held layer still are.

    python tests/capability/layer_p13_a_still_is_one_layers.py             # simulator
    python tests/capability/layer_p13_a_still_is_one_layers.py --hardware  # the connected scope
"""

import sys
import threading
import time
import traceback

import tifffile

import harness
from harness import HARDWARE, check, figure, hardware_session, report, void

# A camera writes exposure in its own steps and reads gain back in its own;
# the file carries what the camera reported, rounded.
EXPOSURE_TOLERANCE_MS = 0.05
GAIN_TOLERANCE_DB = 0.05
# How far the travel case's steps lie from the start, in um of X.
TRAVEL_UM = 5000.0
TRAVEL_DELAYS_S = (0.0, 0.05, 0.15)


def _file_record(path):
    from modules.image_utils import _structured_metadata

    with tifffile.TiffFile(str(path)) as tif:
        return _structured_metadata(tif)


def _still(session):
    stills = session.manual_capture.capture(false_color_on=False).result(timeout=60)
    return stills[0].file


def judge(session, label, path, expect=None):
    """Check one saved still: one layer's name, settings and held layer."""
    meta = _file_record(path)
    if meta is None:
        check(f'{label}: the file carries its record', False, str(path))
        return None
    plane = meta.get('Plane', {})
    channel = (meta.get('Channel', {}).get('Name') or [None])[0]
    held = plane.get('CameraHeldLayer')
    got = (plane.get('ExposureTime'), plane.get('Gain'))
    stored = session.get_settings_snapshot()[channel]
    figure(f'{label}: file', {'name': path.name, 'channel': channel, 'held': held, 'file': got})
    figure(f'{label}: {channel} stored', (stored['exposure_ms'], stored['gain_db']))
    check(f'{label}: the name is the channel', channel in path.name, path.name)
    check(
        f'{label}: the camera held the named layer',
        held == channel,
        f'held {held}, named {channel}',
    )
    if expect is not None:
        check(f'{label}: named {expect}', channel == expect, f'named {channel}')
    if stored['auto_gain']:
        void(f'{label}: exposure and gain are the layer stored', False, f'{channel} auto-gain on')
    else:
        check(
            f'{label}: exposure and gain are the layer stored',
            got[0] is not None
            and got[1] is not None
            and abs(got[0] - stored['exposure_ms']) <= EXPOSURE_TOLERANCE_MS
            and abs(got[1] - stored['gain_db']) <= GAIN_TOLERANCE_DB,
            f'file {got}, stored {(stored["exposure_ms"], stored["gain_db"])}',
        )
    return channel


def _steps(session, protocol, colors):
    return {
        protocol.step_row(step_index=i)['Color']: i
        for i in range(protocol.num_steps())
        if protocol.step_row(step_index=i)['Color'] in colors
    }


def body(session, runner):
    scope = session.scope
    layers = list(session.get_layer_configs())
    fluor = 'Blue' if 'Blue' in layers else next(layer for layer in layers if layer != 'BF')
    pair = ('BF', fluor)
    figure('layers compared', pair)
    snapshot = session.get_settings_snapshot()
    acquire_before = {layer: snapshot[layer]['acquire'] for layer in layers}
    exposure_before = snapshot['BF']['exposure_ms']
    x0 = scope.motion.get_target_position('X')
    try:
        for layer in layers:
            session.set_layer_acquire(layer, 'image' if layer in pair else None)
        here = scope.protocols.create_protocol(empty_config=session.get_sequenced_capture_config())
        session.add_step(here, before_step_index=0)
        scope.motion.move_xy(x_um=x0 + TRAVEL_UM)
        there = scope.protocols.create_protocol(empty_config=session.get_sequenced_capture_config())
        session.add_step(there, before_step_index=0)
        scope.motion.move_xy(x_um=x0)
        at_here = _steps(session, here, pair)
        at_there = _steps(session, there, pair)
        figure(
            'steps here (index, X)',
            {c: (i, here.step_row(step_index=i)['X']) for c, i in at_here.items()},
        )
        figure(
            'steps there (index, X)',
            {c: (i, there.step_row(step_index=i)['X']) for c, i in at_there.items()},
        )

        # 1. A run, then a still.
        session.show_channel(fluor)
        session.light_shown_channel(False)
        outcome = runner.run_single_scan(here, sequence_name='p13_run', timeout_s=300)
        figure('run outcome', (outcome.status, outcome.reason))
        deadline = time.time() + 90
        while time.time() < deadline and session.controls_lock.run_lockout:
            time.sleep(0.25)
        judge(session, 'run then still', _still(session), expect=fluor)

        # 2. A go-to, then a still.
        session.go_to_step(here, at_here['BF'])
        judge(session, 'go-to then still', _still(session), expect='BF')

        # 3. A typed value, then a still.
        session.show_channel('BF')
        typed = round(exposure_before + 5.0, 2)
        session.update_settings('BF.exposure_ms', typed)
        check(
            'typed value stored',
            session.get_settings_snapshot()['BF']['exposure_ms'] == typed,
            str(typed),
        )
        judge(session, 'typed value then still', _still(session), expect='BF')
        session.update_settings('BF.exposure_ms', exposure_before)

        # 4. A still while a go-to to the other layer's step is under way.
        named = []
        for n, delay in enumerate(TRAVEL_DELAYS_S):
            # From the other end's BF step, so the go-to travels and changes the layer.
            (start, start_steps), (protocol, steps) = (
                ((here, at_here), (there, at_there))
                if n % 2 == 0
                else ((there, at_there), (here, at_here))
            )
            session.go_to_step(start, start_steps['BF'])
            ended = threading.Event()
            failure = []

            def go(protocol=protocol, idx=steps[fluor], failure=failure, ended=ended):
                try:
                    session.start_go_to_step(protocol, idx).result(timeout=60)
                except BaseException as e:
                    failure.append(e)
                finally:
                    ended.set()

            threading.Thread(target=go, name='p13-go-to', daemon=True).start()
            time.sleep(delay)
            path = _still(session)
            ended.wait(60)
            check(f'travel {delay:.2f} s: the go-to ended', not failure, repr(failure))
            named.append(judge(session, f'still {delay:.2f} s into a go-to', path))
        figure('layers the travel stills were named for', named)
    finally:
        scope.illumination.leds_off()
        for layer, mode in acquire_before.items():
            session.set_layer_acquire(layer, mode)
        session.update_settings('BF.exposure_ms', exposure_before)
        scope.motion.move_xy(x_um=x0)


def main():
    if HARDWARE:
        with hardware_session() as (session, runner):
            body(session, runner)
        sys.exit(report())
    from tests.scope_fakes import TEST_TURRET_OBJECTIVES

    session, _live = harness.make_session(
        'layer_p13', home=True, microscope='LS850T', turret_objectives=dict(TEST_TURRET_OBJECTIVES)
    )
    try:
        body(session, session.create_protocol_runner())
    except BaseException:
        traceback.print_exc()
        check('probe completed without an unexpected raise', False)
    finally:
        session.shutdown()
    sys.exit(report())


if __name__ == '__main__':
    main()
