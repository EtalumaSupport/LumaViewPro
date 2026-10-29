"""P1: per-channel LED on/off + illumination current, read back through the API."""

import harness

from modules.lumascope_api.illumination import LedState
from tests.stock_channels import LED0
from tests.scope_fakes import channel_reading

s, live = harness.make_session('p1')
ill = s.scope.illumination
try:
    print('available:', [(r.address, r.name) for r in ill.get_led_states()])
    # on at 123 mA
    ill.led_on(LED0, 123.0)
    st = channel_reading(ill, LED0)
    print(f'after led_on({LED0},123):', st)
    assert st.state is LedState.LIT and st.illumination_ma == 123.0, st
    # change current while on
    ill.led_on(LED0, 45.0)
    print(f'after led_on({LED0},45):', channel_reading(ill, LED0))
    assert channel_reading(ill, LED0).illumination_ma == 45.0
    # off
    ill.led_off(LED0)
    print(
        f'after led_off({LED0}):',
        channel_reading(ill, LED0),
        'state=',
        channel_reading(ill, LED0).state,
    )
    assert channel_reading(ill, LED0).state is LedState.DARK
    # out-of-range refusal?
    try:
        ill.led_on(LED0, 99999.0)
        print('led_on 99999 mA -> ACCEPTED, state:', channel_reading(ill, LED0))
    except Exception as e:
        print('led_on 99999 mA -> REFUSED', type(e).__name__, e)
    ill.leds_off()
    print('PASS')
    harness.assert_no_ui()
finally:
    s.shutdown()
