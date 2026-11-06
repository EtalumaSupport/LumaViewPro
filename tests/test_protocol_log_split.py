# Copyright Etaluma, Inc.
"""Regression test: protocol-execution detail routes to a dedicated log.

A long protocol soak emitted tens of thousands of per-step lines (step
records, per-channel LED, image-captured events) into the main log,
swamping it. protocol_image_writer now logs through a dedicated
``LVP.protocol`` logger (propagate=False -> protocol.log), mirroring the
serial / camera / metrics split, so the main log stays readable while the
full run history is preserved.

The suite's process declines the log files once, at its start, so the lane's
files are checked in a child that starts a LumaViewPro program's logging.
"""

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


# A LumaViewPro program's start, in a child: logging starts once per process,
# and the suite's has declined its files.
_STARTED = """
import json, logging, os, sys
sys.path.insert(0, 'src')
from _lumascope import lvp_logger
lvp_logger.start_program_logging()
before_a_line = os.path.exists(lvp_logger.PROTOCOL_LOG_FILE)
lane = logging.getLogger('LVP.protocol')
lane.info('a protocol step line')
for handler in (*lane.handlers, lvp_logger.file_handler):
    handler.flush()
print(json.dumps({
    'protocol_log_before_a_line': before_a_line,
    'propagate': lane.propagate,
    'protocol_log': open(lvp_logger.PROTOCOL_LOG_FILE, encoding='utf-8').read(),
    'main_log': open(lvp_logger.LOG_FILE, encoding='utf-8').read(),
}))
"""


@pytest.fixture(scope='module')
def started(tmp_path_factory):
    """What a started program's protocol lane did, read once for the module's tests."""
    folder = tmp_path_factory.mktemp('launch')
    for item in ('src', 'version.txt'):
        (folder / item).symlink_to(REPO_ROOT / item)
    shutil.copytree(REPO_ROOT / 'data', folder / 'data')
    done = subprocess.run(
        [sys.executable, '-c', _STARTED],
        cwd=str(folder),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_a_protocol_line_goes_to_protocol_log_and_not_the_main_log(started):
    """Once logging starts, the LVP.protocol lane writes protocol.log and does
    not propagate into the main log."""
    assert started['propagate'] is False
    assert 'a protocol step line' in started['protocol_log']
    assert 'a protocol step line' not in started['main_log']


def test_protocol_log_file_created_lazily(started):
    """protocol.log appears only when a line goes through the lane.

    An eagerly-created handler left a 0-byte protocol.log on every
    launch, making 'no stills protocol ran this session' (including
    video-only protocol runs) indistinguishable from 'the lane broke'.
    With lazy creation, absence is data -- the same contract autofocus.log
    and serial.log already carry.
    """
    assert started['protocol_log_before_a_line'] is False


def test_protocol_image_writer_uses_protocol_logger():
    """protocol_image_writer -- the per-step narrative bulk -- logs through
    the dedicated protocol lane, so its per-step detail lands in protocol.log."""
    from _lumascope.modules import protocol_image_writer

    assert protocol_image_writer.logger is logging.getLogger('LVP.protocol')
