"""P1: per-channel LED on/off + illumination current, read back through the API."""

import harness

from modules.lumascope_api.illumination import LedState

s, live = harness.make_session('p1')
ill = s.scope.illumination
try:
    print('available:', [(r.address, r.name) for r in ill.get_led_states()])
    # on at 123 mA
    ill.led_on('Blue', 123.0)
    st = ill.get_led_state('Blue')
    print('after led_on(Blue,123):', st)
    assert st.state is LedState.LIT and st.illumination_ma == 123.0, st
    # change current while on
    ill.led_on('Blue', 45.0)
    print('after led_on(Blue,45):', ill.get_led_state('Blue'))
    assert ill.get_led_state('Blue').illumination_ma == 45.0
    # off
    ill.led_off('Blue')
    print(
        'after led_off(Blue):',
        ill.get_led_state('Blue'),
        'state=',
        ill.get_led_state('Blue').state,
    )
    assert ill.get_led_state('Blue').state is LedState.DARK
    # out-of-range refusal?
    try:
        ill.led_on('Blue', 99999.0)
        print('led_on 99999 mA -> ACCEPTED, state:', ill.get_led_state('Blue'))
    except Exception as e:
        print('led_on 99999 mA -> REFUSED', type(e).__name__, e)
    ill.leds_off()
    print('PASS')
    harness.assert_no_ui()
finally:
    s.shutdown()
