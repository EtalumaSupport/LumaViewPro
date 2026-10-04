# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Deleting and renaming a step are Session members, refused at the API.

The protocol panel called the protocol's own writers, so a script or REST
caller had no delete or rename, and the panel decided two things itself:
that a Delete or a rename with no step did nothing, and that a name made
only of characters a filename cannot carry kept the old name in silence.
Both are now refusals the API raises, and an edit that leaves a step the
run will refuse is noticed once, as Add and Update are.
"""

from __future__ import annotations

import logging

import pytest

import modules.config_helpers as config_helpers
from modules.exceptions import ProtocolError, Refusal
from modules.notification_center import notifications
from modules.protocol import StepEditRefusedError, StepNotFoundError
from tests.test_adding_a_step_is_an_api_capability import session  # noqa: F401 -- pytest fixture


@pytest.fixture
def reported(monkeypatch):
    reports = []
    real = notifications.report_outcome

    def spy(exc, **kw):
        if type(exc).__name__ == 'ProtocolStepsInvalidNotice':
            reports.append(kw['solicited'])
        return real(exc, **kw)

    monkeypatch.setattr(notifications, 'report_outcome', spy)
    return reports


def _three_bf_steps(session):
    """Three BF steps, at X = 10000, 20000 and 30000 um."""
    for layer in config_helpers.get_layer_configs(session.settings):
        session.settings[layer]['acquire'] = None
    session.settings['BF']['acquire'] = 'image'
    protocol = session.create_empty_protocol()
    for x_um in (10000, 20000, 30000):
        session.scope.motion.move_absolute('X', x_um, wait_until_complete=True)
        session.add_step(protocol, after_step=protocol.num_steps() - 1)
    return protocol


def _names(protocol):
    return [protocol.step(i)['Name'] for i in range(protocol.num_steps())]


class TestDeleteStep:
    def test_the_named_step_goes_and_the_rest_move_up(self, session):
        protocol = _three_bf_steps(session)
        kept = [_names(protocol)[0], _names(protocol)[2]]

        session.delete_step(protocol, 1)

        assert _names(protocol) == kept

    @pytest.mark.parametrize('step_idx', [3, -1])
    def test_a_step_the_protocol_lacks_is_refused_and_nothing_goes(self, session, step_idx):
        protocol = _three_bf_steps(session)
        before = _names(protocol)

        with pytest.raises(ProtocolError):
            session.delete_step(protocol, step_idx)

        assert _names(protocol) == before

    def test_an_empty_protocol_says_it_has_no_steps(self, session):
        protocol = session.create_empty_protocol()

        with pytest.raises(ProtocolError, match='has no steps') as refused:
            session.delete_step(protocol, -1)

        # -1 is the GUI's no-selection index; it means nothing to the person.
        assert '-1' not in str(refused.value)


class TestRenameStep:
    def test_the_step_takes_the_label_and_keeps_it(self, session):
        protocol = _three_bf_steps(session)

        name = session.rename_step(protocol, 1, 'center')

        step = protocol.step(1)
        assert step['Label'] == 'center'
        assert not step['Auto_Named']
        assert name == step['Name']
        assert 'center' in name

    def test_characters_a_filename_cannot_carry_are_removed(self, session):
        protocol = _three_bf_steps(session)

        session.rename_step(protocol, 0, 'my step!')

        assert protocol.step(0)['Label'] == 'mystep'

    def test_a_name_with_nothing_to_keep_is_refused_and_the_step_keeps_its_name(self, session):
        protocol = _three_bf_steps(session)
        before = _names(protocol)

        with pytest.raises(ProtocolError, match='at least one letter'):
            session.rename_step(protocol, 0, '!!!')

        assert _names(protocol) == before

    def test_a_step_the_protocol_lacks_is_refused(self, session):
        protocol = _three_bf_steps(session)

        with pytest.raises(ProtocolError):
            session.rename_step(protocol, 3, 'center')

    def test_an_empty_protocol_says_it_has_no_steps(self, session):
        protocol = session.create_empty_protocol()

        with pytest.raises(ProtocolError, match='has no steps'):
            session.rename_step(protocol, -1, 'center')


class TestARefusedStepEditIsARefusal:
    """Reported as a warning under its own title, never as "Operation failed" with a traceback."""

    def test_a_missing_step_is_a_refusal_titled_no_such_step(self, session):
        protocol = session.create_empty_protocol()

        with pytest.raises(StepNotFoundError) as refused:
            session.delete_step(protocol, -1)

        assert isinstance(refused.value, Refusal)
        assert refused.value.title == 'No Such Step'

    def test_a_name_with_nothing_to_keep_is_a_refusal_titled_step_not_changed(self, session):
        protocol = _three_bf_steps(session)

        with pytest.raises(StepEditRefusedError) as refused:
            session.rename_step(protocol, 0, '!!!')

        assert isinstance(refused.value, Refusal)
        assert refused.value.title == 'Step Not Changed'

    def test_the_reporter_logs_it_once_as_a_warning_with_no_traceback(self, session, caplog):
        protocol = session.create_empty_protocol()
        try:
            session.delete_step(protocol, -1)
        except StepNotFoundError as refused:
            with caplog.at_level(logging.DEBUG):
                notifications.report_outcome(refused, solicited=True, category='UI:DELETE_STEP')

        # The reporter's own record; the interaction log also records the popup.
        records = [
            r
            for r in caplog.records
            if r.name == 'LVP.notifications' and 'has no steps' in r.getMessage()
        ]
        assert [r.levelno for r in records] == [logging.WARNING]
        assert records[0].exc_info is None
        assert 'No Such Step' in records[0].getMessage()


class TestTheInvalidStepNotice:
    def test_an_edit_that_leaves_an_invalid_step_is_noticed_once(self, session, reported):
        protocol = _three_bf_steps(session)
        # Behind the writers: a step the run gate refuses, as a hand-edited file carries one.
        protocol._config['steps'].at[2, 'Exposure'] = 0.0
        reported.clear()

        session.rename_step(protocol, 0, 'first')
        session.delete_step(protocol, 0)

        assert reported == [True, True]

    def test_a_rename_to_the_name_the_step_has_is_not_an_edit(self, session, reported):
        """The GUI's name field renames on every blur; one that changed nothing says nothing."""
        protocol = _three_bf_steps(session)
        session.rename_step(protocol, 0, 'first')
        protocol._config['steps'].at[2, 'Exposure'] = 0.0
        reported.clear()

        name = session.rename_step(protocol, 0, 'first')

        assert reported == []
        assert name == protocol.step(0)['Name']

    def test_renaming_an_auto_named_step_to_its_own_base_is_an_edit(self, session, reported):
        """The step stops being auto-named, so a later channel change keeps the label."""
        protocol = _three_bf_steps(session)
        base = protocol.step(0)['Label']
        assert protocol.step(0)['Auto_Named']
        protocol._config['steps'].at[2, 'Exposure'] = 0.0
        reported.clear()

        session.rename_step(protocol, 0, base)

        assert not protocol.step(0)['Auto_Named']
        assert reported == [True]

    def test_a_valid_edit_reports_nothing(self, session, reported):
        protocol = _three_bf_steps(session)
        reported.clear()

        session.rename_step(protocol, 0, 'first')
        session.delete_step(protocol, 0)

        assert reported == []
