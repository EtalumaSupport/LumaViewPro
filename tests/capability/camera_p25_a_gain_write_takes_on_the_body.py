"""Camera P25: a gain write through the API takes on the body, idle and while it grabs.

`CameraAPI.set_gain_db` answers the gain in effect, and a write the camera
does not apply raises (its docstring; "Camera setting bounds" in the owner
table). On a Pylon body the driver writes `GainSelector` `'All'` first where
the camera has the selector (`PylonCamera.gain`), and since build 3d-4 a
selector the camera refuses raises where it was swallowed, so a body that
refuses `'All'` would refuse every gain write. Which bodies have the
selector, and whether one refuses `'All'`, was never read on a body. This
probe records the camera's `GainSelector` (absent, or its entry), then sets
two gains idle and two while live view grabs, each answered, read back
from the camera and read by its own node name, and puts the gain it came up
with back at the end.

    python tests/capability/camera_p25_a_gain_write_takes_on_the_body.py             # simulator
    python tests/capability/camera_p25_a_gain_write_takes_on_the_body.py --hardware  # a Basler scope or camera

Needs only the camera; nothing moves beyond the bring-up home, no light is
lit, nothing is saved.
"""

import sys
import time
import traceback

from harness import HARDWARE, check, figure, hardware_session, make_session, report
from _lumascope.modules.scope_parts import Board

# Two gains on each side of the walk, both inside every Basler body's range
# and apart by more than any body's gain increment.
TARGETS_DB = (3.0, 6.0)
# How close an answer must be to the gain asked: the dart's and ace 2's
# increment is under this.
WITHIN_DB = 0.05
GRAB_WARMUP_S = 2.0


def _node(scope, name):
    """The camera's own answer for `name`, as a dict, or None when it has no such node."""
    got = scope.diagnostics.read_camera_node(name)
    return None if got is None else {'value': got.value, 'unit': got.unit}


def _set_and_read(scope, when):
    camera = scope.camera
    for target in TARGETS_DB:
        label = f'{when}: gain {target} dB'
        try:
            answered = camera.set_gain_db(target)
        except Exception as ex:
            check(
                f'{label} is taken', False, f'{type(ex).__name__}: {getattr(ex, "reason", "")} {ex}'
            )
            continue
        read = camera.read_gain_db()
        node = _node(scope, 'Gain') if HARDWARE else None
        figure(label, {'answered': answered, 'read_gain_db': read, 'Gain node': node})
        check(
            f'{label} is answered as the gain asked',
            answered is not None and abs(answered - target) <= WITHIN_DB,
            str(answered),
        )
        check(
            f'{label} reads back from the camera',
            abs(read - target) <= WITHIN_DB,
            str(read),
        )
        if HARDWARE:
            check(
                f"{label} is the camera's own Gain node",
                node is not None and abs(float(node['value']) - target) <= WITHIN_DB,
                str(node),
            )


def _probe(session):
    scope = session.scope
    figure('camera', {'model': scope.capabilities.camera_model})
    if HARDWARE:
        figure('GainSelector', _node(scope, 'GainSelector'))
    camera = scope.camera
    came_up_with = camera.read_gain_db()
    figure('gain at bring-up', came_up_with)

    imaging = scope.imaging
    try:
        if imaging.read_streaming():
            imaging.disable_streaming()
        _set_and_read(scope, 'idle')
        imaging.enable_streaming()
        time.sleep(GRAB_WARMUP_S)
        _set_and_read(scope, 'grabbing')
        check('grabbing: the camera still grabs after the writes', imaging.read_streaming())
    finally:
        imaging.disable_streaming()
        camera.set_gain_db(came_up_with)
    check(
        'the gain it came up with is back',
        abs(camera.read_gain_db() - came_up_with) <= WITHIN_DB,
        str(camera.read_gain_db()),
    )


def main():
    if HARDWARE:
        with hardware_session(needs=(Board.CAMERA,)) as (session, _runner):
            _probe(session)
        return
    session, _live = make_session('probe_gain_write')
    try:
        _probe(session)
    finally:
        session.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        check('probe completed without an unexpected raise', False)
    sys.exit(report())
