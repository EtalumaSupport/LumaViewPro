#!/usr/bin/env python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Refusals, and a fault the simulator injects, on the simulator.

Everything the API refuses or fails at is an exception under
``LumascopeError`` carrying a ``reason`` to branch on, and a refusal a
``cause``: whether the same request could succeed later. This provokes one
refusal of each cause -- a move before the scope is homed (the state), a move
past the stage's travel (the request), and a still after the simulator has
pulled the camera's cable (nothing until a restart) -- and prints each.

Run it from a LumaViewPro installation folder:

    python3 docs/api_examples/refusals_and_faults.py

The same refusals over REST are ``refusals_and_faults.sh``.
"""

import pathlib
import sys

# The installation folder, importable when the script is run by its path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from modules.exceptions import LumascopeError
from modules.scope_session import ScopeSession


def refused(attempt):
    """The refusal ``attempt`` raises, printed; it is expected to raise one."""
    try:
        attempt()
    except LumascopeError as refusal:
        print(f'{refusal.reason} ({refusal.cause}): {refusal}')
        return refusal
    raise AssertionError('the attempt was not refused')


def main():
    session = ScopeSession.create(ScopeSession.load_user_settings('.'), simulate=True)
    try:
        scope = session.scope

        # The scope's state: until it is homed its position is unknown, so
        # a move is refused rather than driven blind. Retry once it is homed.
        before_home = refused(lambda: scope.motion.move_z(4000.0))
        session.run_application_session()

        # The request: no state of the scope lets this move succeed.
        past_travel = refused(lambda: scope.motion.move_z(1_000_000.0))

        # A fault the simulator injects through the same driver path a real
        # one takes, on every tier: the camera's cable pulled. What needs the
        # camera is refused until the application restarts.
        scope.simulation.remove_camera()
        print(f'Not connected: {scope.unconnected_parts()}')
        scope.illumination.led_on('BF', 5)
        no_camera = refused(
            lambda: session.manual_capture.capture(false_color_on=False).result(timeout=30)
        )
        scope.illumination.leds_off()
        return before_home, past_travel, no_camera
    finally:
        session.shutdown()


if __name__ == '__main__':
    main()
