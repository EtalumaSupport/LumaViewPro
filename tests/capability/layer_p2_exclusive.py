"""P2: does the API enforce one-LED-at-a-time (GUI's disable_leds_for_other_layers)?"""

import harness

from modules.lumascope_api.illumination import LedState

s, live = harness.make_session('p2')
ill = s.scope.illumination
try:
    ill.led_on('Blue', 50.0)
    ill.led_on('Green', 60.0)
    states = {r.name: r for r in ill.get_led_states() if r.state is LedState.LIT}
    print('lit after two led_on calls:', states)
    print('EXCLUSIVITY ENFORCED BY API:', len(states) <= 1)
    ill.leds_off()
    print('after leds_off:', [r for r in ill.get_led_states() if r.state is LedState.LIT])
finally:
    s.shutdown()
