#!/usr/bin/env python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Stopping a run, on the simulator.

Starts an hour-long timed protocol over three wells, a scan a minute, and
stops it once its first scan has ended. A stop only asks: the run unwinds,
puts the scope back and ends ``aborted``, and the images it took are still
written. Prints how it ended and what it kept.

Run it in a LumaViewPro source folder (how, step by step: README.md here):

    python3 docs/api_examples/stop_a_run.py

The same stop over REST is ``stop_a_run.sh``.
"""

import pathlib
import sys
import threading

# The installation folder, importable when the script is run by its path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from modules.run_events import RunEvents
from modules.scope_session import ScopeSession

WELLS = ('B2', 'B3', 'B4')


def main():
    session = ScopeSession.create(ScopeSession.load_user_settings('.'), simulate=True)
    try:
        session.run_application_session()
        scope = session.scope

        # A brightfield step at each well, a scan every 60 s for an hour.
        session.set_channel_acquire('BF', 'image')
        session.update_settings('BF.illumination_ma', 5)
        protocol = session.create_empty_protocol()
        for well in WELLS:
            scope.motion.move_to_well(well)
            session.add_step(protocol, channels=('BF',))
        session.set_protocol_schedule(protocol, period_s=60.0, duration_s=3600.0)
        print(f'Protocol: {protocol.num_steps()} steps, {protocol.num_scans()} scans')

        # A run's events arrive on the run's threads: this one only says so.
        first_scan_ended = threading.Event()
        events = RunEvents(scan_ended=lambda scan, remaining, period_s: first_scan_ended.set())
        runner = session.create_protocol_runner()
        running = runner.start_protocol(protocol, sequence_name='stopped', events=events)

        first_scan_ended.wait(timeout=60)
        running.stop()

        # The run ends on its own once it has unwound; its images keep
        # writing after it lets go of the scope, so they are waited for too.
        outcome = running.result(timeout=60)
        files = running.files.result(timeout=60)
        print(f'Ended: {outcome.status} ({outcome.reason}) -- {outcome.message}')
        print(f'Captured {outcome.captures.captured} of {outcome.captures.asked}')
        print(f'Images written: {files.written}, in {running.progress().run_dir}')
        return outcome, files
    finally:
        session.shutdown()


if __name__ == '__main__':
    main()
