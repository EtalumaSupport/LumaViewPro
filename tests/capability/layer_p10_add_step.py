"""P10: adding a step -- can a script add a step to a protocol the way Add Step does?

One stage. The Session composes the add from its own settings (layer
configs, stim configs, plate position, objective, channel order) and the
protocols API performs it, refusing when nothing would be added. Before the
API member existed the GUI handler owned the whole decision and called the
protocol module directly, so a script had no way to add a step at all.
"""

import sys
import traceback

import harness

s, live = harness.make_session('p10', home=True)
try:
    scope = s.scope
    for layer in s.get_layer_configs():
        s.settings[layer]['acquire'] = None
    s.settings['BF']['acquire'] = 'image'

    protocol = scope.protocols.create_protocol(empty_config=s.get_sequenced_capture_config())
    names = s.add_step(protocol, before_step=0)
    print('P10: add_step ->', names, 'steps now', protocol.num_steps())
    harness.check(
        'a script can add a step through the Session',
        protocol.num_steps() == 1 and protocol.step(idx=0)['Color'] == 'BF',
    )

    from modules.exceptions import ProtocolRunRefusedError

    s.settings['BF']['acquire'] = None
    refused = False
    try:
        s.add_step(protocol, before_step=0)
    except ProtocolRunRefusedError as e:
        refused = e.reason == 'no_acquiring_layer'
    harness.check('an add that would add nothing is refused at the API', refused)
    harness.check('the refusal added nothing', protocol.num_steps() == 1)
    harness.assert_no_ui()
except Exception:
    traceback.print_exc()
    harness.check('probe completed without an unexpected raise', False)
finally:
    s.shutdown()
sys.exit(harness.report())
