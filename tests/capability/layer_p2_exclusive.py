"""P2: does the API enforce one-LED-at-a-time (GUI's disable_leds_for_other_layers)?"""

import harness

s, live = harness.make_session('p2')
ill = s.scope.illumination
try:
    ill.led_on('Blue', 50.0, block=True)
    ill.led_on('Green', 60.0, block=True)
    states = {c: v for c, v in ill.get_led_states().items() if v['enabled']}
    print('lit after two led_on calls:', states)
    print('EXCLUSIVITY ENFORCED BY API:', len(states) <= 1)
    ill.leds_off()
    print('after leds_off:', {c: v for c, v in ill.get_led_states().items() if v['enabled']})
finally:
    s.shutdown()
