#!/usr/bin/env python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Protocol execution example.

Demonstrates:
- Choosing the layers a run captures (ScopeSession.set_channel_acquire)
- Building a Protocol from the session's own configuration, without a file
- Running one scan through the session's protocol runner and reading how
  it ended

This example builds a protocol of one well and two channels, runs it once
on the simulated scope, and prints what the run captured.
"""

import sys
import pathlib
import tempfile

# Make the repo root importable when run standalone
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

# This example runs the SAME code path two ways:
#   standalone: python3 docs/api_examples/protocol_execution.py  (the real installed deps)
#   in-suite:   tests/test_api_examples.py runs main() under the heavy-dep
#               mocks the test conftest installs before collection
# The sys.path line serves the standalone form; in-suite it is a no-op.

from modules.scope_session import ScopeSession

LAYERS = ('BF', 'Ex405')
# One well, in plate millimetres: well A1 of the selected plate.
WELL = 'A1'


def main():
    # The user's configuration, prepared as the GUI prepares it (read,
    # checked against the shipped template, repaired, merged). A root with
    # no usable settings is refused, never quietly replaced by the template.
    settings = ScopeSession.load_user_settings('.')

    # The factory is the one door to a scope: it wires the simulated
    # drivers on the installation's data folder, configures the scope from
    # the settings, starts the camera feed and the executor lanes.
    #
    # A real script names its own folder; this one leaves nothing behind.
    # The folder outlives the session: a run's result is in before its files
    # are written, and shutdown waits for them.
    with tempfile.TemporaryDirectory() as live:
        session = ScopeSession.create(settings, simulate=True)
        try:
            run_one_scan(session, live)
        finally:
            # The factory built the scope, so shutdown disconnects it.
            session.shutdown()
    print('Session shut down')


def run_one_scan(session, live):
    session.set_live_folder(live)

    # Every run moves the stage, so it is refused (position_unknown) until
    # the axes have been homed.
    session.scope.motion.home('ALL')

    # The layers set to acquire are the ones a protocol builds steps for.
    for layer in LAYERS:
        session.set_channel_acquire(layer, 'image')

    # A protocol with a step per acquiring layer at one well, from the
    # session's settings: the plate, the objective and each layer's
    # settings. session.new_protocol() builds one at every well instead.
    protocol = session.create_empty_protocol()
    session.scope.motion.move_to_well(WELL)
    session.add_step(protocol)
    print(f'Protocol: {protocol.num_steps()} steps, layers {list(LAYERS)}')

    # The run captures in the session's image mode and output formats
    # (session.scope.camera.set_image_mode(...) chooses the mode).
    runner = session.create_protocol_runner()
    pending = runner.start_single_scan(protocol, sequence_name='example_scan')

    # Wait for the run to end, then read HOW it ended: status is
    # completed/incomplete/aborted/failed/failed_at_start, reason the
    # machine-readable cause, captures what was captured of what was asked.
    # A bound that expires first raises and leaves the run going;
    # pending.stop() stops it.
    result = pending.result(timeout=300)
    print(f'Ended: {result.status} ({result.reason}) -- {result.message}')
    print(f'Captured {result.captures.captured} of {result.captures.asked}')

    # runner.start_protocol(protocol, sequence_name=...) runs the full timed
    # protocol, a scan every period for its duration, and returns the same
    # kind of operation.


if __name__ == '__main__':
    main()
