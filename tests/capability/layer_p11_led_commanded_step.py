"""P11: an LED request lights at the board's step, and the scope records the current commanded.

On an FX2 scope one step is one brightness byte (840 / 255 mA): a BF request
of 1.0 mA lights at byte 1 (~3.29 mA) and 5.0 mA at byte 2 (~6.59 mA); 0 mA
is dark with the channel still on at 0 mA. On a serial LED board the step is
whole mA. The probe reads what the API returns, stores and puts in the frame
record, and the frame brightness at each step, so the operator can watch the
light at the optics while it runs.

    python tests/capability/layer_p11_led_commanded_step.py             # an LS620 simulator
    python tests/capability/layer_p11_led_commanded_step.py --hardware  # the connected scope
    ... --layer Green   # a channel other than BF (the default)

The frames are taken at the layer's own saved exposure and gain.
"""

import contextlib
import sys
import time

from harness import HARDWARE, check, figure, hardware_session, report

LAYER = sys.argv[sys.argv.index('--layer') + 1] if '--layer' in sys.argv else 'BF'
HOLD_S = 3.0  # time to look at the optics at each step


def _frame(scope):
    imaging = scope.imaging
    image = imaging.capture_and_wait(accept_dark=True, timeout_s=5.0)
    record = (imaging.last_capture_info or {}).get('frame_record')
    mean = None if image is None else round(float(image.mean()), 2)
    return mean, record


@contextlib.contextmanager
def _ls620_simulator():
    """An LS620 on the simulator: the production FX2 drivers over a simulated FX2.

    Not the harness's headless session, which sets timing on a simulated
    serial LED board an FX2 scope does not have.
    """
    from modules.scope_session import ScopeSession

    from tests.settings_fixtures import complete_settings

    session = ScopeSession.create(
        complete_settings(microscope='LS620'), simulate=True, warn_pre_release=False
    )
    try:
        yield session, None
    finally:
        session.shutdown()


def main():
    session_cm = hardware_session() if HARDWARE else _ls620_simulator()
    with session_cm as (session, _runner):
        scope = session.scope
        illumination = scope.illumination
        step = illumination._driver.commanded_ma(1e-9)
        figure('one step, mA', round(step, 4))

        saved = session.settings[LAYER]
        figure(f'{LAYER} exposure ms', scope.imaging.set_exposure_ms(float(saved['exposure_ms'])))
        figure(f'{LAYER} gain dB', scope.imaging.set_gain_db(float(saved['gain_db'])))

        illumination.leds_off()
        dark, _ = _frame(scope)
        figure('dark frame mean', dark)

        means = {}
        for request in (1.0, 5.0, 0.0):
            expected = illumination._driver.commanded_ma(request)
            print(f'--- {LAYER} {request} mA: look at the optics now ---', flush=True)
            commanded = illumination.led_on(LAYER, request, block=True)
            time.sleep(HOLD_S)
            state = illumination.get_led_state(LAYER)
            mean, record = _frame(scope)
            means[request] = mean
            figure(f'{request} mA: commanded', commanded)
            figure(f'{request} mA: state', state)
            figure(f'{request} mA: frame mean', mean)
            recorded = None if record is None else record.illumination_ma.get(LAYER)
            figure(f'{request} mA: frame record', recorded)
            check(f'{request} mA is commanded on the step', commanded == expected, commanded)
            check(
                f'{request} mA: the state holds the commanded current, on',
                state == {'enabled': True, 'illumination_ma': expected},
                state,
            )
            if request > 0:
                check(f'{request} mA lights at one step or more', expected >= step, expected)
                check(f'{request} mA: the frame records it', recorded == expected, recorded)
                check(
                    f'{request} mA: the frame is brighter than the dark frame',
                    None not in (mean, dark) and mean > dark,
                    (mean, dark),
                )
            else:
                check('0 mA: the frame records no current', recorded is None, recorded)

        check(
            '5 mA is at least as bright as 1 mA',
            None not in (means[5.0], means[1.0]) and means[5.0] >= means[1.0],
            means,
        )
        illumination.leds_off()
    sys.exit(report())


if __name__ == '__main__':
    main()
