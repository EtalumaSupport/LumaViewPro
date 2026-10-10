# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every LumaViewPro log rotates wherever its data folder lives.

A backup is named from the file name alone: ``lumaviewpro.log.1`` becomes
``lumaviewpro.1.log``. Editing the whole path instead renamed into a folder
that does not exist whenever a folder above the log had ``.log`` in its name
(a Windows user ``j.logan``), and from that rollover on every record was lost.

The suite's conftest stands a MagicMock in for ``lvp_logger``, so each test
runs the real module in a child process.
"""

import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]


def _run(code: str):
    done = subprocess.run(
        [sys.executable, '-c', code], capture_output=True, text=True, timeout=60, cwd=str(REPO)
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_a_log_under_a_folder_named_with_log_keeps_its_backups_and_its_newest_record(tmp_path):
    log_dir = tmp_path / 'j.logan' / 'logs' / 'LVP_Log'
    log_dir.mkdir(parents=True)
    code = (
        'import json, logging, pathlib, lvp_logger\n'
        'from logging.handlers import RotatingFileHandler\n'
        f'log_dir = pathlib.Path({str(log_dir)!r})\n'
        "handler = RotatingFileHandler(log_dir / 'lumaviewpro.log', maxBytes=2000, backupCount=5)\n"
        'handler.namer = lvp_logger.file_handler.namer\n'
        "log = logging.getLogger('rotation_probe')\n"
        'log.propagate = False\n'
        'log.addHandler(handler)\n'
        'for i in range(120):\n'
        "    log.warning('record %04d %s', i, 'x' * 40)\n"
        'handler.close()\n'
        "newest = (log_dir / 'lumaviewpro.log').read_text().splitlines()[-1]\n"
        'print(json.dumps([sorted(p.name for p in log_dir.iterdir()), newest]))\n'
    )

    names, newest = _run(code)

    assert names == [f'lumaviewpro.{n}.log' for n in range(1, 4)] + ['lumaviewpro.log']
    assert newest.startswith('record 0119')


def test_every_shipped_log_names_its_backups_by_the_one_namer():
    code = (
        'import json, lvp_logger\n'
        'from logging.handlers import RotatingFileHandler\n'
        'handlers = [v for v in vars(lvp_logger).values() if isinstance(v, RotatingFileHandler)]\n'
        "namer = getattr(lvp_logger, 'rotated_log_name', None)\n"
        'print(json.dumps([len(handlers), all(h.namer is namer for h in handlers)]))\n'
    )

    count, one_namer = _run(code)

    assert count == 10
    assert one_namer
