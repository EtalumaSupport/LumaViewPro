# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol file LumaViewPro cannot take is reported once, in words that name it.

The parser logged an ERROR naming the file before each of its checks raised
a bare exception, and a file it could not read became a bare OSError. Shown
through the one reporter, neither carried a kind: each read as a crash, a
traceback in the log beside the parser's own line, and the person who chose
the file was told only that the operation did not complete. A file that
cannot be parsed is now a refusal naming the file and what is wrong with
it, and a file that cannot be read is a fault naming the file and the
operating system's reason; each is logged once, by the reporter.
"""

import logging

import pytest

from modules.exceptions import ProtocolNotLoadedError, Refusal
from modules.notification_center import NotificationCenter, Severity
from modules.protocol import Protocol, ProtocolFormatError
from tests.test_protocol_roundtrip import TILING_CONFIGS


def _load(path):
    return Protocol.from_file(file_path=path, tiling_configs_file_loc=TILING_CONFIGS)


def _report(error):
    """Report *error* through a fresh reporter; return what it showed."""
    centre = NotificationCenter(dedup_window_s=10.0)
    shown = []
    centre.add_listener(shown.append, min_severity=Severity.INFO)
    centre.report_outcome(error, solicited=True, category='UI:LOAD_PROTOCOL')
    return shown


@pytest.fixture
def junk(tmp_path):
    path = tmp_path / 'junk.tsv'
    path.write_text('LumaViewPro Protocol\nVersion\tseven\n')
    return path


class TestAFileThatIsNotAProtocol:
    def test_it_is_a_refusal_naming_the_file_and_what_is_wrong(self, junk):
        with pytest.raises(ProtocolFormatError) as refused:
            _load(junk)

        assert isinstance(refused.value, Refusal)
        assert refused.value.file == junk
        assert str(refused.value) == (
            f"{junk} was not loaded: Invalid 'Version' value in protocol file: "
            'must be a whole number'
        )

    def test_it_is_logged_once_with_no_traceback_and_shown_in_its_words(self, junk, caplog):
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(ProtocolFormatError) as refused:
                _load(junk)
            shown = _report(refused.value)

        logged = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert [(r.levelno, r.exc_info) for r in logged] == [(logging.WARNING, None)]
        assert [(n.title, n.message) for n in shown] == [('Protocol Refused', str(refused.value))]

    def test_a_file_too_large_to_load_is_refused_without_calling_it_damaged(self, tmp_path):
        huge = tmp_path / 'huge.tsv'
        huge.write_bytes(b'x' * (10 * 1024 * 1024 + 1))

        with pytest.raises(ProtocolFormatError, match='exhausting memory') as refused:
            _load(huge)

        assert 'corrupt' not in str(refused.value).lower()


class TestAFileThatCannotBeRead:
    def test_it_names_the_file_and_the_operating_systems_reason(self, tmp_path):
        missing = tmp_path / 'gone.tsv'

        with pytest.raises(ProtocolNotLoadedError) as failed:
            _load(missing)

        assert isinstance(failed.value.__cause__, FileNotFoundError)
        assert failed.value.file == missing
        assert str(failed.value) == (
            f'The protocol at {missing} could not be read (No such file or directory).'
        )

    def test_it_is_shown_in_its_words(self, tmp_path):
        with pytest.raises(ProtocolNotLoadedError) as failed:
            _load(tmp_path / 'gone.tsv')

        shown = _report(failed.value)

        assert [(n.title, n.message) for n in shown] == [('Protocol Not Loaded', str(failed.value))]
