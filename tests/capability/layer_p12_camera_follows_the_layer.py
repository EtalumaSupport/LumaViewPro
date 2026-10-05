"""P12: the camera holds a layer's exposure and gain after bring-up and after going to a step.

Asked of the API alone, as a script or REST would: bring-up puts BF's stored
settings on the camera, and `go_to_step` the step's layer's
(`ScopeSession.apply_layer_camera`). Before that member, only the GUI's
layer control set the camera, and the LS850T read the Pylon default
(10 ms / 0 dB) after bring-up and after each go.

1. after bring-up, the camera against BF's stored exposure and gain (not
   asked when BF's auto-gain is on: the camera then holds what the auto loop
   reached);
2. after `go_to_step` to a Blue step, then a BF step, built from the layers'
   own settings, the camera against that step's exposure and gain.

A capture after each go reads the frame record, the values a saved file
would carry.

    python tests/capability/layer_p12_camera_follows_the_layer.py             # simulator
    python tests/capability/layer_p12_camera_follows_the_layer.py --hardware  # the connected scope
"""

import sys

from harness import HARDWARE, check, figure, hardware_session, make_session, report


def _camera(scope):
    imaging = scope.imaging
    return round(float(imaging.get_exposure_ms()), 3), round(float(imaging.get_gain_db()), 3)


def _record(scope):
    imaging = scope.imaging
    imaging.capture_and_wait(accept_dark=True, timeout_s=5.0)
    record = (imaging.last_capture_info or {}).get('frame_record')
    if record is None:
        return None
    return {k: getattr(record, k, None) for k in ('exposure_ms', 'gain_db')}


def _sim():
    session, _live = make_session('p12', home=True)
    return session


def body(session):
    scope = session.scope
    settings = session.settings
    layers = list(session.get_layer_configs())

    after_bring_up = _camera(scope)
    figure('camera after bring-up (exposure ms, gain dB)', after_bring_up)
    for layer in layers:
        figure(f'{layer} stored', (settings[layer]['exposure_ms'], settings[layer]['gain_db']))
    matching = [
        layer
        for layer in layers
        if after_bring_up
        == (
            round(float(settings[layer]['exposure_ms']), 3),
            round(float(settings[layer]['gain_db']), 3),
        )
    ]
    figure('layers the camera matches after bring-up', matching)
    bf = (
        round(float(settings['BF']['exposure_ms']), 3),
        round(float(settings['BF']['gain_db']), 3),
    )
    if settings['BF']['auto_gain']:
        figure(
            'bring-up check',
            "not asked: BF's auto-gain is on, so the camera holds what the loop reached",
        )
    else:
        check(
            'after bring-up the camera holds BF',
            after_bring_up == bf,
            f'camera {after_bring_up}, BF {bf}',
        )

    for layer in layers:
        settings[layer]['acquire'] = None
    settings['BF']['acquire'] = 'image'
    settings['Blue']['acquire'] = 'image'
    protocol = scope.protocols.create_protocol(empty_config=session.get_sequenced_capture_config())
    session.add_step(protocol, before_step=0)
    colors = [protocol.step(idx=i)['Color'] for i in range(protocol.num_steps())]
    figure('steps', colors)

    for color in ('Blue', 'BF'):
        idx = colors.index(color)
        step = protocol.step(idx=idx)
        want = (round(float(step['Exposure']), 3), round(float(step['Gain']), 3))
        session.go_to_step(protocol, idx)
        got = _camera(scope)
        figure(f'{color} step (exposure ms, gain dB)', want)
        figure(f'camera after go_to_step({color})', got)
        figure(f'frame record after go_to_step({color})', _record(scope))
        check(
            f'after go_to_step({color}) the camera holds the step exposure and gain',
            got == want,
            f'camera {got}, step {want}',
        )
    scope.illumination.leds_off()


def main():
    if HARDWARE:
        with hardware_session() as (session, _runner):
            body(session)
    else:
        session = _sim()
        try:
            body(session)
        finally:
            session.shutdown()
    sys.exit(report())


if __name__ == '__main__':
    main()
