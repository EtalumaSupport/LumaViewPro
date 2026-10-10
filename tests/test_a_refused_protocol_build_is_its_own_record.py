# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol build refused for invalid steps writes no ERROR line per step.

``Protocol.from_config`` logged every validation error at ERROR before
raising the one refusal, so two refused New presses on a 96-step protocol
with exposure 0 wrote 192 lines to the error log beside the one refusal
shown. The refusal is the record: whoever catches it reports it once.
"""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock

import pytest

import modules.protocol as protocol_module
from modules.protocol import Protocol, ProtocolFormatError
from tests.scope_fakes import build_scope
from tests.test_a_zstack_with_no_range_is_refused import _standalone_config


_REPO_ROOT = pathlib.Path(__file__).parent.parent
_TILING = _REPO_ROOT / 'data' / 'tiling.json'


@pytest.fixture
def sim_scope():
    scope = build_scope(simulate=True, source_path=_REPO_ROOT)
    try:
        yield scope
    finally:
        scope.disconnect()


def test_invalid_steps_are_refused_without_a_log_line_each(sim_scope, monkeypatch):
    config = _standalone_config({'range': 20.0, 'step_size': 5.0}, use_zstacking=False)
    config['positions'] = [{'name': f'P{i}', 'x': 1.0 + i, 'y': 2.0, 'z': 3.0} for i in range(3)]
    config['layer_configs']['BF']['exposure_ms'] = 0.0

    # The suite replaces lvp_logger, so the module's logger is watched directly.
    logger = MagicMock()
    monkeypatch.setattr(protocol_module, 'logger', logger)

    with pytest.raises(ProtocolFormatError) as refusal:
        Protocol.from_config(
            input_config=config,
            tiling_configs_file_loc=_TILING,
            capabilities=sim_scope.capabilities,
            objective_helper=sim_scope.objective_helper,
            wellplate_loader=sim_scope.wellplate_loader,
        )

    assert 'validation error(s)' in str(refusal.value)
    assert logger.error.call_args_list == []
