#!/usr/bin/env python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A scan of two wells whose images are kept, on the simulator.

Builds a protocol with a brightfield step at each of two wells, runs it once,
and reads how it ended. A run lets go of the scope before its images are all
on disk, so this waits for its files too, then lists the run's folder under
the live folder. Prints what it captured and the files it kept.

Run it from a LumaViewPro installation folder:

    python3 docs/api_examples/scan.py

The same scan over REST is ``scan.sh``.
"""

import pathlib
import sys

# The installation folder, importable when the script is run by its path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from modules.scope_session import ScopeSession

WELLS = ('B2', 'B3')


def main():
    session = ScopeSession.create(ScopeSession.load_user_settings('.'), simulate=True)
    try:
        session.run_application_session()
        scope = session.scope

        # The channels set to acquire are the ones a step captures; a step
        # is added where the stage is, on the protocol's plate.
        session.set_channel_acquire('BF', 'image')
        session.update_settings('BF.illumination_ma', 5)
        protocol = session.create_empty_protocol()
        for well in WELLS:
            scope.motion.move_to_well(well)
            session.add_step(protocol, channels=('BF',))

        runner = session.create_protocol_runner()
        started = runner.start_single_scan(protocol, sequence_name='example_scan')

        # A run ends with an outcome; it does not raise. Its images keep
        # writing after it lets go of the scope, and the next run waits for
        # them, so the files are their own wait.
        outcome = started.result(timeout=120)
        files = started.files.result(timeout=60)
        run_dir = started.progress().run_dir
        print(f'Ended: {outcome.status} ({outcome.reason}) -- {outcome.message}')
        print(f'Captured {outcome.captures.captured} of {outcome.captures.asked}')
        print(f'Images written: {files.written}, in {run_dir}')
        for path in sorted(run_dir.iterdir()):
            print(f'  {path.name}')
        return outcome, files, run_dir
    finally:
        session.shutdown()


if __name__ == '__main__':
    main()
