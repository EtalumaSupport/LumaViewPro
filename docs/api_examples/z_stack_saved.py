#!/usr/bin/env python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A saved z-stack, on the simulator.

A z-stack is a run: its range, step and reference -- whether the stage's
position is the stack's top, centre or bottom -- are settings, and it saves a
slice per step around where the stage is. This sets a 20 um stack in 5 um
steps centred on the stage, runs it on brightfield over well B2, waits for
its slices to be written, and prints each slice's file. Each file carries
its own record, its Z among it, which Fiji and other Bio-Formats readers show.

Run it from a LumaViewPro installation folder:

    python3 docs/api_examples/z_stack_saved.py

The same stack over REST is ``z_stack_saved.sh``.
"""

import pathlib
import sys

# The installation folder, importable when the script is run by its path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from modules.scope_session import ScopeSession


def main():
    session = ScopeSession.create(ScopeSession.load_user_settings('.'), simulate=True)
    try:
        session.run_application_session()
        scope = session.scope
        scope.motion.move_to_well('B2')

        # The stack's extent and step, and where the stage sits in it.
        session.update_settings('zstack.range', 20.0)
        session.update_settings('zstack.step_size', 5.0)
        session.update_settings('zstack.position', 'center')
        session.update_settings('BF.illumination_ma', 5)

        runner = session.create_protocol_runner()
        stacking = runner.start_zstack('BF', sequence_name='example_stack')
        outcome = stacking.result(timeout=120)
        files = stacking.files.result(timeout=60)
        run_dir = stacking.progress().run_dir
        print(f'Ended: {outcome.status} ({outcome.reason}) -- {outcome.message}')
        print(f'Slices written: {files.written}, in {run_dir}')

        slices = sorted(run_dir.glob('*.tiff'))
        for path in slices:
            print(f'  {path.name}')
        return outcome, files, slices
    finally:
        session.shutdown()


if __name__ == '__main__':
    main()
