"""A run refused because the last run's file writer stalled names its own remedy.

The remedy is data the refusal carries -- the Session member that answers it
and the words of the offer -- so the one reporter can show the refusal as a
confirmation and a REST or SDK caller receives the same remedy by name. The
Session is the one place a name becomes an action. Before this, the offer
lived in the GUI's press gates: delete them and a person who declined the
drain tick's single offer had no way back but a restart.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.exceptions import ProtocolRunRefusedError, Remedy, RemedyUnknownError
from modules.image_mode import ImageCaptureConfig
from modules.notification_center import NotificationCenter, Severity
from modules.protocol_image_writer import RunWriteBatch
from modules.scope_session import ScopeSession
from modules.sequenced_capture_runner import SequencedCaptureRunMode
from modules.sequential_io_executor import ENQUEUED
from tests.protocol_drives import autofocus_snapshot
from tests.scope_fakes import spec_scope


@pytest.fixture
def session():
    bundle = SimpleNamespace(
        file_io_executor=MagicMock(),
        post_processing_executor=MagicMock(),
        protocol_thread=MagicMock(),
        shutdown=lambda: None,
    )
    return ScopeSession(settings={}, scope=spec_scope(), executor_bundle=bundle)


def _draining(session, *, stalled: bool, writes: int = 1) -> RunWriteBatch:
    """The last run ended with *writes* still to land on the file lane."""
    lane = session.file_io_executor
    lane.put.return_value = ENQUEUED
    lane.in_flight_task_stalled.return_value = stalled
    lane.describe_running_task.return_value = "write_capture 'B2_BF' 45s in flight"
    batch = RunWriteBatch(lane)
    for i in range(writes):
        batch.submit(lambda: None, {}, what=f'The image {i}', pace_until=None)
    batch.close(lambda outcome: None)
    session.sequenced_capture_runner._write_batch = batch
    return batch


def _refused_start(session, tmp_path) -> ProtocolRunRefusedError:
    with pytest.raises(ProtocolRunRefusedError) as refused:
        session.sequenced_capture_runner.prepare(
            protocol=MagicMock(),
            run_trigger_source='test',
            run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
            sequence_name='t',
            image_capture_config=ImageCaptureConfig.from_image_mode('8bit'),
            autogain_settings={},
            parent_dir=tmp_path,
            autofocus_snapshot=autofocus_snapshot(),
        )
    return refused.value


class TestTheRefusalCarriesItsRemedy:
    @pytest.mark.slow
    def test_a_stalled_writer_refusal_names_recovery_and_its_cost(self, session, tmp_path):
        _draining(session, stalled=True, writes=3)

        refusal = _refused_start(session, tmp_path)

        assert refusal.reason == 'files_writing_stalled'
        assert refusal.remedy is not None, 'the refusal names no way out of it'
        assert refusal.remedy.member == 'recover_file_writer'
        assert '3' in refusal.remedy.confirm_text, 'the offer must say what recovering loses'
        assert '3 unsaved image(s)' in refusal.message, 'the prompt must say what recovering loses'

    @pytest.mark.slow
    def test_a_writer_still_moving_offers_nothing(self, session, tmp_path):
        """Healthy drain: the writes will land; recovering would lose them for nothing."""
        _draining(session, stalled=False)

        refusal = _refused_start(session, tmp_path)

        assert refusal.reason == 'files_writing'
        assert refusal.remedy is None


class TestTheSessionAppliesARemedyByName:
    def test_the_stalled_refusal_s_remedy_recovers_the_writer(self, session, tmp_path):
        batch = _draining(session, stalled=True, writes=2)
        refusal = _refused_start(session, tmp_path)

        assert session.apply_remedy(refusal.remedy) == 2
        session.file_io_executor.replace_stuck_worker.assert_called_once()
        assert batch.outcome == 'incomplete'

    def test_a_name_the_session_does_not_offer_is_refused(self, session):
        """A remedy names a member; only the members the Session allows are reachable."""
        with pytest.raises(RemedyUnknownError) as refused:
            session.apply_remedy(Remedy(member='shutdown', confirm_text='Go', cancel_text='No'))

        assert refused.value.reason == 'remedy_unknown'
        assert refused.value.title, 'a refusal is shown under its own title'
        session.file_io_executor.replace_stuck_worker.assert_not_called()


class TestTheReporterCarriesTheRemedyToItsListeners:
    def test_the_notification_of_a_refusal_carries_its_remedy(self):
        # A fresh refusal: one the runner raised has already been shown once,
        # by the reporter it raised through, and is not shown again.
        refusal = ProtocolRunRefusedError(
            reason='files_writing_stalled',
            title='File Writer Stalled',
            message='Recover it.',
            remedy=Remedy(member='recover_file_writer', confirm_text='Go', cancel_text='Wait'),
        )
        center = NotificationCenter()
        heard = []
        center.add_listener(heard.append, min_severity=Severity.DEBUG)

        center.report_outcome(refusal, solicited=True, category='Protocol')

        (shown,) = heard
        assert shown.remedy == refusal.remedy

    def test_a_refusal_without_one_carries_none(self):
        center = NotificationCenter()
        heard = []
        center.add_listener(heard.append, min_severity=Severity.DEBUG)

        center.report_outcome(
            ProtocolRunRefusedError(
                reason='files_writing', title='Files Still Writing', message='Wait.'
            ),
            solicited=True,
            category='Protocol',
        )

        (shown,) = heard
        assert shown.remedy is None
