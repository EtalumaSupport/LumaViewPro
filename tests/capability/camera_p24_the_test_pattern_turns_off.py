"""Camera P24: what a Basler's TestPattern offers, and what turning the pattern off leaves.

`set_test_pattern(enabled, pattern)` turns the pattern on when `enabled`
and writes `'Off'` when not, one base body for every camera. Whether a
Basler body offers `'Off'`, and whether the node lands there, is the
camera's answer. This probe reads it:

- the `TestPattern` node's value as found, and the entries it offers;
- `set_test_pattern(True, 'Black')`, then the node read;
- `set_test_pattern(False)`, then the node read.

    python tests/capability/camera_p24_the_test_pattern_turns_off.py --hardware  # the connected scope

The simulator keeps its own pattern and has no `TestPattern` node, so the
probe is a bench read only. The node is put back as it was found. Live view
stays off, nothing is lit, and the stage does not move beyond the bring-up
home.
"""

import sys
import traceback

from harness import HARDWARE, check, figure, hardware_session, report

NODE = 'TestPattern'


def _read(scope):
    answer = scope.diagnostics.read_camera_node(NODE)
    return None if answer is None else answer.value


def _probe(session):
    scope = session.scope
    driver = scope._camera_driver
    figure('camera', {'model': scope.capabilities.camera_model, 'driver': type(driver).__name__})
    found = _read(scope)
    figure('as found', found)
    entries = driver._entries(NODE)
    figure('entries', list(entries))
    check(f'{NODE} offers an Off entry', 'Off' in entries, str(list(entries)))
    try:
        driver.set_test_pattern(True, 'Black')
        on = _read(scope)
        figure("after set_test_pattern(True, 'Black')", on)
        check("set_test_pattern(True, 'Black') leaves Black", on == 'Black', str(on))
        driver.set_test_pattern(False)
        off = _read(scope)
        figure('after set_test_pattern(False)', off)
        check('set_test_pattern(False) leaves the pattern off', off == 'Off', str(off))
    finally:
        if found is not None:
            driver.write_node(NODE, found)
            figure('put back', _read(scope))


def main():
    if not HARDWARE:
        print(
            'REFUSED: P24 reads a Basler node the simulator does not have; run it with --hardware.'
        )
        sys.exit(2)
    with hardware_session() as (session, _runner):
        _probe(session)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        check('probe completed without an unexpected raise', False)
    sys.exit(report())
