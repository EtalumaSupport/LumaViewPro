# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol's load and a run's preparation each say how many steps and how long.

Both walk every step -- a load on the GUI's thread at a Load and at startup,
a preparation on the worker pool at the GUI's Run -- so a large protocol
spends seconds in each. A field bundle's camera drops inside those windows
could be placed only by inference from GUI display events, because neither
pass logged its size or its duration. Each now says both, once, in the main log: the
log is the product here, so the line is what is asserted.
"""

from __future__ import annotations

import logging
import re

from modules.sequenced_capture_runner import SequencedCaptureRunMode
from tests.test_protocol_execution import (  # noqa: F401 -- pytest fixtures
    _make_autogain_settings,
    _make_image_capture_config,
    _make_multi_step_protocol,
    executor,
    executors,
    scope,
)

STEPS = 3


def _protocol():
    return _make_multi_step_protocol([{'name': f'step_{i}'} for i in range(STEPS)])


def _lines(caplog, pattern):
    return [r.getMessage() for r in caplog.records if re.search(pattern, r.getMessage())]


def test_a_load_says_its_steps_and_its_read_and_check_times(scope, tmp_path, caplog):
    path = tmp_path / 'sized.tsv'
    _protocol().to_file(file_path=path)
    caplog.set_level(logging.INFO)

    scope.protocols.load_protocol(file_path=path)

    lines = _lines(caplog, r'\[Protocol\] Loaded ')
    assert len(lines) == 1, lines
    assert re.fullmatch(
        rf'\[Protocol\] Loaded "sized\.tsv": {STEPS} steps, '
        r'read in \d+\.\d\d s, checked in \d+\.\d\d s',
        lines[0],
    ), lines[0]


def test_a_prepared_run_says_its_steps_and_how_long_the_checks_took(executor, tmp_path, caplog):
    caplog.set_level(logging.INFO)

    executor.prepare(
        protocol=_protocol(),
        run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
        sequence_name='sized',
        image_capture_config=_make_image_capture_config(),
        autogain_settings=_make_autogain_settings(),
        parent_dir=tmp_path / 'ProtocolData',
        max_scans=1,
    )

    lines = _lines(caplog, r'\[SeqCapExec\] Run prepared')
    assert len(lines) == 1, lines
    assert re.fullmatch(
        rf'\[SeqCapExec\] Run prepared: {STEPS} steps checked in \d+\.\d\d s', lines[0]
    ), lines[0]
