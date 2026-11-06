# Copyright Etaluma, Inc.
"""Regression test: protocol-execution detail routes to a dedicated log.

A long protocol soak emitted tens of thousands of per-step lines (step
records, per-channel LED, image-captured events) into the main log,
swamping it. protocol_image_writer now logs through a dedicated
``LVP.protocol`` logger (propagate=False -> protocol.log), mirroring the
serial / camera / metrics split, so the main log stays readable while the
full run history is preserved.

The suite's conftest mocks lvp_logger wholesale, so the lane is checked in
a child that starts the real one; the bindings are source scans.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _src(rel):
    return (REPO_ROOT / rel).read_text()


# A LumaViewPro program's start, in a child: the suite's conftest stands a
# MagicMock in for lvp_logger, so only a fresh interpreter starts the real one.
_STARTED = """
import json, logging, sys
sys.path.insert(0, 'src')
from _lumascope import lvp_logger
lvp_logger.start_program_logging()
lane = logging.getLogger('LVP.protocol')
lane.info('a protocol step line')
for handler in (*lane.handlers, lvp_logger.file_handler):
    handler.flush()
print(json.dumps({
    'propagate': lane.propagate,
    'protocol_log': open(lvp_logger.PROTOCOL_LOG_FILE, encoding='utf-8').read(),
    'main_log': open(lvp_logger.LOG_FILE, encoding='utf-8').read(),
}))
"""


def test_a_protocol_line_goes_to_protocol_log_and_not_the_main_log(tmp_path):
    """Once logging starts, the LVP.protocol lane writes protocol.log and does
    not propagate into the main log."""
    folder = tmp_path / 'launch'
    folder.mkdir()
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
    state = json.loads(done.stdout.strip().splitlines()[-1])

    assert state['propagate'] is False
    assert 'a protocol step line' in state['protocol_log']
    assert 'a protocol step line' not in state['main_log']


def test_protocol_image_writer_uses_protocol_logger():
    """protocol_image_writer -- the per-step narrative bulk -- must log
    through the dedicated protocol logger, not the root logger."""
    # pin-justified: import-binding contract; conftest mocks lvp_logger
    # wholesale, so the binding cannot be observed at runtime in pytest.
    src = _src('src/_lumascope/modules/protocol_image_writer.py')
    assert 'from _lumascope.lvp_logger import protocol_logger as logger' in src, (
        'protocol_image_writer must bind its module logger to '
        'protocol_logger so its per-step detail lands in protocol.log'
    )


def test_protocol_log_file_created_lazily():
    """The protocol.log handler must use delay=True so the file appears
    only when a stills protocol actually logs through the lane.

    An eagerly-created handler left a 0-byte protocol.log on every
    launch, making 'no stills protocol ran this session' (including
    video-only protocol runs) indistinguishable from 'the lane broke'.
    With lazy creation, absence is data -- the same contract autofocus.log
    and serial.log already carry.
    """
    # pin-justified: handler construction; conftest mocks lvp_logger
    # wholesale, so the real handler never instantiates in pytest.
    src = _src('src/_lumascope/lvp_logger.py')
    idx = src.find('protocol_file_handler = _rotating_handler')
    assert idx != -1, 'lvp_logger must construct protocol_file_handler'
    window = src[idx : idx + 300]
    assert 'delay=True' in window, (
        'protocol_file_handler must set delay=True: a 0-byte protocol.log '
        'created at import makes a session with no stills protocol '
        'indistinguishable from a broken lane; absence is the signal'
    )
