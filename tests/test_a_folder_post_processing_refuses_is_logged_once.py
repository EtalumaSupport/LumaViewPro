# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""A folder post-processing cannot use is a refusal, logged once, as a refusal.

Picking a folder with no protocol in it is an ordinary thing to do. The
loader used to log it at ERROR, the processor logged it again as it built the
refusal, and the reporter then logged the refusal itself: three records, the
first of them an error for something that is not one. The reporter's record
is the one record.
"""

import logging
from unittest.mock import MagicMock

import pandas as pd
import pytest

import modules.protocol_post_processing_helper as helper_module
from modules.exceptions import PostProcessingRefusedError
from modules.notification_center import NotificationCenter
from tests.test_post_processing_tells_whoever_asked import _Processor
from tests.installation_fixtures import shipped_grids


def test_an_empty_folder_is_one_warning_and_no_error(tmp_path, caplog):
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(PostProcessingRefusedError) as raised:
            _Processor([]).load_folder(path=tmp_path, tiling_grids=shipped_grids())
        NotificationCenter().report_outcome(
            raised.value, solicited=True, category='Post-processing', log_only=True
        )

    assert raised.value.reason == 'protocol_data_unreadable'
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
    naming_it = [r for r in caplog.records if 'not found in folder' in r.getMessage()]
    assert [(r.name, r.levelno) for r in naming_it] == [('LVP.outcomes', logging.WARNING)]


def _record(num_records):
    record = MagicMock()
    record.num_records.return_value = num_records
    return record


def _unreadable(**_kwargs):
    raise ValueError('header does not match')


def _locked(*_args):
    raise OSError('in use by another program')


# Each folder the loader cannot process, past the first check: what it
# stubs, and the words of the refusal the caller raises.
_FOLDERS = {
    'record_empty': ({'record': _record(0)}, 'Protocol Execution Record has no records'),
    'post_record_stuck': (
        {'record': _record(1), 'post_record': _unreadable, 'replace': _locked},
        'could not be read',
    ),
    'no_images': ({'record': _record(1)}, 'No image files were found'),
}


# parametrize-justified: each folder has its own stubs and tmp folder
@pytest.mark.parametrize('folder', sorted(_FOLDERS))
def test_every_folder_it_cannot_process_logs_no_error(folder, tmp_path, caplog, monkeypatch):
    stubs, words = _FOLDERS[folder]
    found = {
        'protocol': tmp_path / 'protocol.tsv',
        'protocol_execution_record': tmp_path / 'protocol_record.tsv',
        'protocol_post_record': tmp_path / 'post_record.tsv',
        'protocol_root_dir': tmp_path,
        'execution_record': stubs['record'],
    }
    helper = helper_module.ProtocolPostProcessingHelper
    monkeypatch.setattr(helper, '_find_protocol_tsvs', lambda self, path: found)
    monkeypatch.setattr(helper_module.Protocol, 'from_file', lambda **kw: MagicMock())
    monkeypatch.setattr(
        helper_module.ProtocolPostRecord,
        'from_file',
        stubs.get('post_record', lambda **kw: MagicMock()),
    )
    if 'replace' in stubs:
        monkeypatch.setattr(helper_module.os, 'replace', stubs['replace'])
    monkeypatch.setattr(
        helper, '_get_image_filenames_from_folder', lambda self, **kw: {'raw': [], 'post': []}
    )
    monkeypatch.setattr(helper, '_get_raw_images_df', lambda self, **kw: pd.DataFrame())
    monkeypatch.setattr(helper, '_get_post_images_df', lambda self, **kw: pd.DataFrame())

    with (
        caplog.at_level(logging.DEBUG),
        pytest.raises(PostProcessingRefusedError) as raised,
    ):
        _Processor([]).load_folder(path=tmp_path, tiling_grids=shipped_grids())

    assert words in str(raised.value)
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR] == []
