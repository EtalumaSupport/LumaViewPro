# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run starter takes its subject by position and everything else by name.

Each ``start_X`` and ``run_X`` of ``ProtocolRunner`` takes the thing it runs
-- the protocol, or the layer -- as its first argument, and every other
argument by keyword only, so a caller cannot hand a parent folder where a
sequence name was meant. A composite has no subject, so its starters take
every argument by keyword.
"""

import inspect

from modules.protocol_runner import ProtocolRunner

_SUBJECT = {
    'single_scan': 'protocol',
    'protocol': 'protocol',
    'autofocus': 'channel',
    'autofocus_all_steps': 'protocol',
    'zstack': 'channel',
    'composite': None,
}


def test_each_starter_takes_only_its_subject_by_position():
    wrong = {}
    for name in vars(ProtocolRunner):
        if not name.startswith(('start_', 'run_')):
            continue
        run = name.split('_', 1)[1]
        if run not in _SUBJECT:
            wrong[name] = 'not listed with its subject'
            continue
        parameters = list(inspect.signature(getattr(ProtocolRunner, name)).parameters.values())[1:]
        by_position = [p.name for p in parameters if p.kind is not inspect.Parameter.KEYWORD_ONLY]
        if by_position != ([_SUBJECT[run]] if _SUBJECT[run] else []):
            wrong[name] = by_position
    assert not wrong, wrong
