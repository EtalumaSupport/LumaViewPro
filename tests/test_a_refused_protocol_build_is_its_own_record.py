# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol build refused for invalid steps writes no ERROR line per step.

``Protocol.from_config`` logged every validation error at ERROR before
raising the one refusal, so two refused New presses on a 96-step protocol
with exposure 0 wrote 192 lines to the error log beside the one refusal
shown. The refusal is the record: whoever catches it reports it once.
"""

from __future__ import annotations

import logging
import pathlib

import pytest

import modules.protocol as protocol_module
from modules.protocol import Protocol, ProtocolFormatError
from tests.scope_fakes import build_scope
from tests.test_a_zstack_with_no_range_is_refused import _standalone_config
from tests.installation_fixtures import shipped_grids
from tests.log_capture import records_of


_REPO_ROOT = pathlib.Path(__file__).parent.parent


@pytest.fixture
def sim_scope():
    scope = build_scope(simulate=True, source_path=_REPO_ROOT)
    try:
        yield scope
    finally:
        scope.disconnect()


def test_invalid_steps_are_refused_without_a_log_line_each(sim_scope, caplog):
    config = _standalone_config({'range': 20.0, 'step_size': 5.0}, use_zstacking=False)
    config['positions'] = [{'name': f'P{i}', 'x': 1.0 + i, 'y': 2.0, 'z': 3.0} for i in range(3)]
    config['layer_configs']['BF']['exposure_ms'] = 0.0

    with pytest.raises(ProtocolFormatError) as refusal:
        Protocol.from_config(
            input_config=config,
            tiling_grids=shipped_grids(),
            capabilities=sim_scope.capabilities,
            objective_helper=sim_scope.objective_helper,
            wellplate_loader=sim_scope.wellplate_loader,
        )

    assert 'validation error(s)' in str(refusal.value)
    errors = [
        r.getMessage() for r in records_of(caplog, protocol_module) if r.levelno == logging.ERROR
    ]
    assert errors == [], errors
