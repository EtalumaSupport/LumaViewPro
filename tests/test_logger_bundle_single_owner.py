"""Regression: the bundle file handlers have a single owner, so a record is written to the bundle once.

Bug shape (pre-fix): the same RotatingFileHandler instances that write the
shared bundle (lvpmain.log / errors.log) were attached to the root logger AND
to the LVP loggers AND to `kivy`. A record that
propagated up to root was then written to the bundle a SECOND time through
root's copy of the same handler. This was masked only by disabling propagation
in non-debug, so a debug run -- exactly when the bundle matters for diagnosis --
doubled every LVP line.

Fix: one owner holds the bundle handlers, and every ordinary logger reaches
them by propagation. A LumaViewPro program's start hangs them on root; a
session's start in an integrator's process hangs them on LumaViewPro's one
logger root, `LVP`, which still propagates to the integrator's root. In both,
a warning from `LVP` itself, from a logger under it and (for a program) from
`kivy` lands once in the main log and once in the errors log.

Run in a child per shape: logging is process-wide, and the suite's process
has started its own, with no files (``tests/conftest.py``).
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]

_CHILD = """
import json, logging, sys
sys.path.insert(0, 'src')
from _lumascope import lvp_logger
getattr(lvp_logger, sys.argv[1])()
names = ['LVP.some_module', 'LVP'] + (['kivy'] if sys.argv[1] == 'start_program_logging' else [])
for name in names:
    logging.getLogger(name).warning(f'once from {name}')
for handler in (lvp_logger.file_handler, lvp_logger.error_file_handler):
    handler.flush()
print(json.dumps([lvp_logger.LOG_FILE, lvp_logger.ERRORS_LOG_FILE, names]))
"""


def _bundle_counts(tmp_path, start):
    folder = tmp_path / start
    folder.mkdir()
    for item in ('src', 'version.txt'):
        (folder / item).symlink_to(REPO / item)
    shutil.copytree(REPO / 'data', folder / 'data')
    done = subprocess.run(
        [sys.executable, '-c', _CHILD, start],
        cwd=str(folder),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    main, errors, names = json.loads(done.stdout.strip().splitlines()[-1])
    texts = [pathlib.Path(p).read_text(encoding='utf-8') for p in (main, errors)]
    # A whole line's end, not a substring: 'once from LVP' is inside 'once from LVP.some_module'.
    return {
        name: [
            sum(line.endswith(f' once from {name}') for line in text.splitlines()) for text in texts
        ]
        for name in names
    }


def test_each_shape_writes_a_propagated_record_to_the_bundle_once(tmp_path):
    for start in ('start_program_logging', 'start_session_logging'):
        counts = _bundle_counts(tmp_path, start)
        doubled = {name: c for name, c in counts.items() if c != [1, 1]}
        assert not doubled, f'{start}: (main, errors) counts other than once: {doubled}'
