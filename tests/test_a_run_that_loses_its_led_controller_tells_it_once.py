# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run that loses its LED controller tells the person each thing once.

The cleanup's end-of-run LED transition waits for its answer and reports a
refusal in its one summary: with a channel believed lit and no controller
to darken it, the summary says the LEDs were not restored. The transition's
task was not silent, so the IO lane reported the same refusal a second
time, as its own "Not Connected" popup. A waited transition's task is now
silent, so its waiter is its one reporter; the run's abort and the
cleanup's summary are what the person sees.

The loss is an FX2 pulled (the LS620's camera and LED are one USB device,
whose removal the driver detects). A serial LED board unplugged is not
used: when the API learns that board is gone depends on which write's
error the serial driver keeps, the serial-exchange row's subject, not this
one's.
"""

from modules.notification_center import Severity
from modules.run_events import RunEvents
from tests.test_a_command_for_absent_motion_hardware_is_refused import _protocol, _session, _wait


def test_an_fx2_pulled_mid_run_is_told_by_the_abort_and_the_cleanup(tmp_path, centre_posts):
    session = _session(tmp_path, 'LS620')
    # Two BF steps, so step 0's light is still on when step 1 begins.
    rows = [{**_protocol().step(idx=0), 'Name': f'A1_s{i}', 'Label': f'A1_s{i}'} for i in range(2)]

    def step_started(step_idx):
        if step_idx == 1:
            session.scope.simulation.remove_camera()

    try:
        centre_posts.clear()
        run = session.create_protocol_runner().start_protocol(
            protocol=_protocol(rows=rows, n_scans=3),
            sequence_name='pulled',
            parent_dir=str(tmp_path / 'run'),
            events=RunEvents(step_started=step_started),
        )

        outcome = _wait(run)

        shown = [(n.title, n.severity) for n in centre_posts if n.shown]
        assert outcome == ('failed', 'hardware_disconnected')
        assert shown == [
            ('Protocol Aborted', Severity.CRITICAL),
            ('Protocol cleanup issues', Severity.ERROR),
        ]
    finally:
        session.shutdown()
