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
import shutil
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


def _run_with_data_root(data_root: pathlib.Path, code: str):
    """Run ``code`` with the real ``lvp_logger`` keeping its logs under ``data_root``.

    A frozen process with no installer marker keeps its data beside its
    executable, so pointing the executable into ``data_root`` moves the log
    folder there and leaves the checkout's own logs alone. The root carries
    the shipped ``settings.json``, as an installation does: the logger reads
    ``debug_mode`` from it.
    """
    (data_root / 'data').mkdir(exist_ok=True)
    shutil.copy(REPO / 'data' / 'settings.json', data_root / 'data' / 'settings.json')
    exe = data_root / 'LumaViewPro.exe'
    prefix = f'import sys\nsys.frozen = True\nsys.executable = {str(exe)!r}\n'
    return _run(prefix + code)


def test_every_log_rotates_at_its_row_of_the_one_cap_table(tmp_path):
    code = (
        'import json, os, lvp_logger\n'
        'from logging.handlers import RotatingFileHandler\n'
        'handlers = [v for v in vars(lvp_logger).values() if isinstance(v, RotatingFileHandler)]\n'
        'print(json.dumps([\n'
        '    {os.path.basename(h.baseFilename): [h.maxBytes, h.backupCount] for h in handlers},\n'
        '    {name: list(cap) for name, cap in lvp_logger.LOG_CAPS.items()},\n'
        ']))\n'
    )

    attached, table = _run_with_data_root(tmp_path, code)

    assert attached == table


def test_every_log_at_its_cap_fits_the_bundle_budget(tmp_path):
    code = (
        'import json, lvp_logger\n'
        'whole = sum(size * (1 + backups) for size, backups in lvp_logger.LOG_CAPS.values())\n'
        'print(json.dumps([whole, lvp_logger.LOG_BUDGET_BYTES]))\n'
    )

    whole, budget = _run_with_data_root(tmp_path, code)

    assert whole <= budget, f'the logs at every cap hold {whole} bytes, past the {budget} budget'


def test_a_backup_past_its_logs_count_is_removed_when_the_logger_loads(tmp_path):
    log_dir = tmp_path / 'logs' / 'LVP_Log'
    log_dir.mkdir(parents=True)
    planted = [f'{stem}.{n}.log' for stem in ('lumaviewpro', 'autofocus') for n in range(1, 6)]
    for name in [*planted, 'notes.9.log', 'lumaviewpro.old.log']:
        (log_dir / name).write_text('x')
    code = (
        'import json, os, lvp_logger\n'
        'from logging.handlers import RotatingFileHandler\n'
        'counts = {\n'
        '    os.path.basename(v.baseFilename): v.backupCount\n'
        '    for v in vars(lvp_logger).values() if isinstance(v, RotatingFileHandler)\n'
        '}\n'
        "print(json.dumps([counts['lumaviewpro.log'], counts['autofocus.log']]))\n"
    )

    main_backups, af_backups = _run_with_data_root(tmp_path, code)

    left = {p.name for p in log_dir.iterdir()}
    assert {f'lumaviewpro.{n}.log' for n in range(1, 6)} & left == {
        f'lumaviewpro.{n}.log' for n in range(1, main_backups + 1)
    }
    assert {f'autofocus.{n}.log' for n in range(1, 6)} & left == {
        f'autofocus.{n}.log' for n in range(1, af_backups + 1)
    }
    # Files that are no log's numbered backup are not the cap's to remove.
    assert {'notes.9.log', 'lumaviewpro.old.log'} <= left
