"""Camera P24: what a Basler's TestPattern offers, and what turning the pattern off leaves.

`set_test_pattern(enabled, pattern)` turns the pattern on when `enabled`
and writes `'Off'` when not, one base body for every camera. Whether a
Basler body offers `'Off'`, and whether the node lands there, is the
camera's answer. This probe reads it:

- the `TestPattern` node's value as found, and the entries it offers;
- `set_test_pattern(True, 'Black')`, then the node read;
- `set_test_pattern(False)`, then the node read.

    python tests/capability/camera_p24_the_test_pattern_turns_off.py             # simulator
    python tests/capability/camera_p24_the_test_pattern_turns_off.py --hardware  # the connected scope

The simulated camera keeps its own pattern and has no named nodes: there
the pattern is turned on and off through the same member, and the node read
is refused as unsupported (`camera_node_unsupported`). On a scope the node is
put back as it was found. Live view stays off, nothing is lit, and the stage
does not move beyond the bring-up home.
"""

import sys
import traceback

from harness import HARDWARE, check, figure, hardware_session, make_session, report
from _lumascope.modules.scope_parts import Board

NODE = 'TestPattern'


def _read(scope):
    answer = scope.diagnostics.read_camera_node(NODE)
    return None if answer is None else answer.value


def _simulated(session):
    from _lumascope.modules.exceptions import CameraSettingUnsupportedError

    scope = session.scope
    driver = scope._camera_driver
    figure('camera', {'model': scope.capabilities.camera_model, 'driver': type(driver).__name__})
    driver.set_test_pattern(True, 'Black')
    driver.set_test_pattern(False)
    try:
        scope.diagnostics.read_camera_node(NODE)
    except CameraSettingUnsupportedError as refused:
        reason = refused.reason
    else:
        reason = None
    check(
        f'the simulated camera refuses a read of {NODE} as unsupported',
        reason == 'camera_node_unsupported',
        str(reason),
    )


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
    if HARDWARE:
        with hardware_session(needs=(Board.CAMERA,)) as (session, _runner):
            _probe(session)
        return
    session, _live = make_session('probe_test_pattern')
    try:
        _simulated(session)
    finally:
        session.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        check('probe completed without an unexpected raise', False)
    sys.exit(report())
