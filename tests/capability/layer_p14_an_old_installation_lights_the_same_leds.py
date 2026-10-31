"""P14: an installation saved before the excitation keys lights the same LEDs on the release.

Run twice on one scope, each time from a checkout holding a copy of the same
pre-release ``current.json``: first at a commit before the keys were named
for their excitation, saving a protocol with a step per fluorescence channel;
then at the release, loading that protocol and comparing against the first
run's verdict. Each run records, per fluorescence LED address, the channel
the scope names for it and that channel's stored current, exposure and gain;
and per step, its channel and the LED addresses lit (with their current)
when the step is gone to with the preview on, and the frame mean lit and dark.

    python tests/capability/layer_p14_an_old_installation_lights_the_same_leds.py --hardware --save DIR
    python tests/capability/layer_p14_an_old_installation_lights_the_same_leds.py --hardware \
        --compare BEFORE_VERDICT.json DIR

Without ``--hardware`` it runs the save leg on a simulated LS850, which
proves the probe runs; the comparison is the bench's, and the simulator's
answer to it is pinned by ``tests/test_an_old_protocol_runs_the_leds_it_was_made_with.py``.
"""

import contextlib
import json
import pathlib
import sys

from harness import HARDWARE, check, figure, hardware_session, make_session, probe_dir, report

SAVE = '--compare' not in sys.argv
PROTOCOL_NAME = 'old_installation.tsv'


def _arg(flag, offset=1):
    return sys.argv[sys.argv.index(flag) + offset]


def _fluorescence(scope):
    """The scope's fluorescence channels, by key, with their LED addresses."""
    return [c for c in scope.channel_identity.channels if c.excitation_nm is not None]


def _stored(session, scope):
    """Per LED address: the channel named for it and its stored settings."""
    settings = session.get_settings_snapshot()
    stored = {}
    for channel in _fluorescence(scope):
        block = settings[channel.key_name]
        for address in channel.led_addresses:
            stored[str(address)] = {
                'key': channel.key_name,
                'illumination_ma': float(block['illumination_ma']),
                'exposure_ms': float(block['exposure_ms']),
                'gain_db': float(block['gain_db']),
            }
    return stored


def _mean(scope):
    image = scope.imaging.capture_still(accept_dark=True, timeout_s=10.0).image
    return round(float(image.mean()), 2)


def _steps(session, scope, protocol):
    """Each step's channel, what going to it lights, and the frame lit and dark."""
    illumination = scope.illumination
    steps = []
    for index in range(protocol.num_steps()):
        session.go_to_step(protocol, index)
        lit = sorted(
            [r.address, r.illumination_ma]
            for r in illumination.get_led_states()
            if r.state.name == 'LIT'
        )
        lit_mean = _mean(scope)
        illumination.leds_off()
        dark_mean = _mean(scope)
        steps.append(
            {
                'color': str(protocol.steps()['Color'].iloc[index]),
                'lit': lit,
                'lit_mean': lit_mean,
                'dark_mean': dark_mean,
            }
        )
        print(f'--- step {index}: {steps[-1]} ---', flush=True)
    return steps


@contextlib.contextmanager
def _ls850_simulator():
    session, _live = make_session(
        'layer_p14',
        home=True,
        microscope='LS850',
        objective_id='10x Oly',
        protocol={'labware': 'Center Plate'},
    )
    # As the bench's installation stood: one plate position, two
    # fluorescence channels acquiring.
    for channel in _fluorescence(session.scope)[:2]:
        session.set_channel_acquire(channel.key_name, 'image')
    try:
        yield session, None
    finally:
        session.shutdown()


def _folder():
    if '--save' in sys.argv:
        return pathlib.Path(_arg('--save'))
    if '--compare' in sys.argv:
        return pathlib.Path(_arg('--compare', 2))
    return probe_dir('layer_p14')


def main():
    folder = _folder()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / PROTOCOL_NAME
    session_cm = hardware_session() if HARDWARE else _ls850_simulator()
    with session_cm as (session, _runner):
        scope = session.scope
        session.update_settings('protocol_led_on', True)
        stored = figure('stored by LED address', _stored(session, scope))
        if SAVE:
            protocol = session.new_protocol()
            session.add_step(protocol, channels=[c.key_name for c in _fluorescence(scope)])
            session.save_protocol(protocol, path)
            figure('protocol saved', str(path))
        else:
            protocol = session.load_protocol(str(path))
        steps = figure('steps', _steps(session, scope, protocol))
        scope.illumination.leds_off()

    for at, step in enumerate(steps):
        check(
            f'step {at} ({step["color"]}) lights one LED, brighter than dark',
            len(step['lit']) == 1 and step['lit_mean'] > step['dark_mean'],
            str(step),
        )
    if SAVE:
        return
    before = json.loads(pathlib.Path(_arg('--compare')).read_text())['figures']
    old_names = {s['key'] for s in before['stored by LED address'].values()}
    check(
        'each LED address holds the settings it held before',
        {a: {k: v for k, v in s.items() if k != 'key'} for a, s in stored.items()}
        == {
            a: {k: v for k, v in s.items() if k != 'key'}
            for a, s in before['stored by LED address'].items()
        },
        f'before {before["stored by LED address"]}, now {stored}',
    )
    check(
        'each step lights the LED and current it lit before',
        [s['lit'] for s in steps] == [s['lit'] for s in before['steps']],
        f'before {[s["lit"] for s in before["steps"]]}, now {[s["lit"] for s in steps]}',
    )
    check(
        'each step is named for the channel the scope now names for its LED',
        [s['color'] for s in steps]
        == [stored[str(s['lit'][0][0])]['key'] for s in before['steps']],
        str([s['color'] for s in steps]),
    )
    check(
        'no step keeps an old name',
        not {s['color'] for s in steps} & old_names - {s['key'] for s in stored.values()},
        str([s['color'] for s in steps]),
    )


if __name__ == '__main__':
    main()
    sys.exit(report())
