# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""A folder post-processing cannot use is a refusal, logged once, as a refusal.

Picking a folder with no protocol in it is an ordinary thing to do. The
loader used to log it at ERROR, the processor logged it again as it built the
refusal, and the reporter then logged the refusal itself: three records, the
first of them an error for something that is not one. The reporter's record
is the one record.
"""

import logging

import pytest

import modules.protocol_post_processing_helper as helper_module
import modules.protocol_post_processor as processor_module
from modules.exceptions import PostProcessingRefusedError
from modules.notification_center import NotificationCenter
from tests.test_capture_collision_policy import TILING_CONFIGS
from tests.test_post_processing_tells_whoever_asked import _Processor


def test_an_empty_folder_is_one_warning_and_no_error(tmp_path, caplog, monkeypatch):
    # The suite's lvp_logger is a stand-in that records nothing; the loader
    # and the processor log into real loggers here so their records are seen.
    monkeypatch.setattr(helper_module, 'logger', logging.getLogger('LVP.test.pp_helper'))
    monkeypatch.setattr(processor_module, 'logger', logging.getLogger('LVP.test.pp_processor'))
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(PostProcessingRefusedError) as raised:
            _Processor([]).load_folder(path=tmp_path, tiling_configs_file_loc=TILING_CONFIGS)
        NotificationCenter().report_outcome(
            raised.value, solicited=True, category='Post-processing', log_only=True
        )

    assert raised.value.reason == 'protocol_data_unreadable'
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
    naming_it = [r for r in caplog.records if 'not found in folder' in r.getMessage()]
    assert [(r.name, r.levelno) for r in naming_it] == [('LVP.outcomes', logging.WARNING)]
