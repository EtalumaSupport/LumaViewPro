"""What a composite run's merge tells the user, and when.

A composite that succeeds is silent: the saved folder and the log line
are the record, and the button already handed the UI back at run end.
A merge that fails tells the user exactly once, from the merge itself,
whatever the failure was -- the post-processor returning nothing usable,
the post-processor raising, or the run's writes never draining. Before
this contract the post-processor's unattended-batch notices did the
talking for the run kind (two modals on every success, one on its own
failure only) and the other failure exits said nothing at all.

The bus is observed through a real listener at NOTICE and above, the
same threshold the GUI popup bridge subscribes at, so a notification that
would never reach the screen cannot satisfy these tests.
"""

import contextlib
import pathlib
import time

import pytest

from modules.exceptions import CaptureError
from modules.notification_center import Severity, notifications
from tests.test_composite_run_e2e import headless_settings, open_composite_session
from tests.test_composite_run_failures import _fail_these_channels


@contextlib.contextmanager
def _bus_at_popup_threshold():
    seen = []
    record = seen.append  # one bound method, so remove_listener finds it
    notifications.add_listener(record, min_severity=Severity.NOTICE)
    try:
        yield seen
    finally:
        notifications.remove_listener(record)


def _about_the_composite(seen):
    return [n for n in seen if 'omposite' in n.title]


class TestASuccessfulCompositeIsSilent:
    def test_no_notification_reaches_the_popup_threshold(self, tmp_path):
        with (
            open_composite_session(headless_settings(tmp_path)) as (_session, runner),
            _bus_at_popup_threshold() as seen,
        ):
            artifact = runner.run_composite(
                sequence_name='quiet', parent_dir=str(tmp_path)
            ).artifact_path

        assert pathlib.Path(artifact).exists()
        assert _about_the_composite(seen) == [], (
            'a successful composite must open no popup; the bus carried '
            f'{[(n.title, n.message) for n in _about_the_composite(seen)]}'
        )


class TestAFailedMergeTellsTheUserOnce:
    def _assert_one_failure_notice(self, seen):
        failures = _about_the_composite(seen)
        assert len(failures) == 1, (
            f'expected exactly one composite notification; saw '
            f'{[(n.title, n.message) for n in failures]}'
        )
        assert failures[0].title == 'Composite Failed'
        assert failures[0].severity >= Severity.ERROR
        return failures[0]

    def test_nothing_usable_to_merge(self, tmp_path):
        # One of two channels comes back black, so the post-processor has
        # no group it can merge and returns its own reason.
        settings = headless_settings(tmp_path, acquiring=('BF', 'Blue'))
        with (
            open_composite_session(settings) as (session, runner),
            _bus_at_popup_threshold() as seen,
            pytest.raises(CaptureError) as excinfo,
        ):
            runner.run_composite(
                sequence_name='no_data',
                parent_dir=str(tmp_path),
                callbacks=_fail_these_channels(
                    session.scope._camera_driver, ('BF', 'Blue'), {'Blue'}
                ),
            )

        # The post-processor's refusal is told as itself: a warning under
        # its own title, never re-typed as a merge failure. The run's missing
        # channel is its own notice, after it.
        assert excinfo.value.reason == 'no_data'
        told = [(n.title, n.severity) for n in seen if n.severity >= Severity.NOTICE]
        assert told == [
            ('Composite Not Possible', Severity.WARNING),
            ('Run Incomplete', Severity.ERROR),
        ], told
        assert 'Blue' in seen[-1].message, 'the shortfall must name the channel that failed'

    def test_the_merge_raises(self, tmp_path, monkeypatch):
        from modules.composite_generation import CompositeGeneration

        def _explode(self, **kwargs):
            raise RuntimeError('synthetic merge crash')

        monkeypatch.setattr(CompositeGeneration, 'load_folder', _explode)
        with (
            open_composite_session(headless_settings(tmp_path)) as (_session, runner),
            _bus_at_popup_threshold() as seen,
            pytest.raises(CaptureError) as excinfo,
        ):
            runner.run_composite(sequence_name='crash', parent_dir=str(tmp_path))

        assert excinfo.value.reason == 'merge_error'
        self._assert_one_failure_notice(seen)

    def test_the_merge_raise_is_logged_once_with_its_traceback(self, tmp_path, monkeypatch, caplog):
        import logging
        from unittest.mock import MagicMock

        import modules.sequenced_capture_runner as scr
        from modules.composite_generation import CompositeGeneration

        def _explode(self, **kwargs):
            raise RuntimeError('synthetic merge crash')

        monkeypatch.setattr(CompositeGeneration, 'load_folder', _explode)
        run_log = MagicMock()
        monkeypatch.setattr(scr, 'logger', run_log)
        with (
            caplog.at_level(logging.ERROR, logger='LVP.outcomes'),
            open_composite_session(headless_settings(tmp_path)) as (_session, runner),
            pytest.raises(CaptureError),
        ):
            runner.run_composite(sequence_name='crash', parent_dir=str(tmp_path))

        # The outcome's own record carries the crash's traceback ...
        outcome_records = [r for r in caplog.records if r.name == 'LVP.outcomes']
        assert len(outcome_records) == 1, [r.getMessage() for r in outcome_records]
        formatted = logging.Formatter().format(outcome_records[0])
        assert 'RuntimeError: synthetic merge crash' in formatted
        assert 'Traceback' in formatted
        # ... so the run does not log the same failure a second time.
        assert run_log.error.call_args_list == []

    def test_the_writes_never_drain(self, tmp_path, monkeypatch):
        import modules.protocol_image_writer as piw
        import modules.sequenced_capture_runner as scr

        real_save = piw.save_image

        def _slow_save(*args, **kwargs):
            time.sleep(0.6)
            return real_save(*args, **kwargs)

        monkeypatch.setattr(piw, 'save_image', _slow_save)
        monkeypatch.setattr(scr, '_POST_RUN_WRITES_WAIT_S', 0.05)
        with (
            open_composite_session(headless_settings(tmp_path)) as (_session, runner),
            _bus_at_popup_threshold() as seen,
            pytest.raises(CaptureError) as excinfo,
        ):
            runner.run_composite(sequence_name='drain', parent_dir=str(tmp_path))

        # The wait's own outcome, told under its own title as the
        # hyperstack build tells it.
        assert excinfo.value.reason == 'write_batch_timeout'
        told = [(n.title, n.severity) for n in seen if n.severity >= Severity.NOTICE]
        assert told == [('Run Images Not Written', Severity.ERROR)], told
