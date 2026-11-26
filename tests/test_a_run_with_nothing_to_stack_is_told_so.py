# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run whose every place holds one image is told there was nothing to stack.

Only the run that wrote a folder builds its hyperstacks, so a folder where
no well, tile or Add has more than one image is a complete answer: no stack,
and the Hyperstacks Saved notice says there was nothing to stack and the
images are saved, never a refused hyperstack. A place whose one
capture failed is counted apart, since it has no image to be saved. A
build a person points at a folder (Z-Projection here) still refuses a
folder with nothing to combine.

Pinned through ``StackBuilder.load_folder``, the build a run's
``PostProcessingAPI.run_hyperstack`` puts on its lane, over a run folder
written by the production protocol and run record, and the notice the
run's reporter shows for its answer.
"""

from __future__ import annotations

import pytest

import _lumascope.modules.config_helpers as config_helpers
from _lumascope.modules.exceptions import HyperstacksSavedNotice, PostProcessingRefusedError
from _lumascope.modules.zprojector import ZProjector
from tests.installation_fixtures import shipped_grids
from tests.test_a_stopped_run_stacks_its_complete_scans import (
    _build,
    _run_folder,
    _scans,
    _stack_shapes,
    _steps_of,
    _wells,
)
from tests.test_adding_a_step_is_an_api_capability import (  # noqa: F401 -- pytest fixture
    session,
    turret_in_a_known_slot,
)


def _bf_protocol(session, *, tiling='1x1'):
    """A New protocol taking BF alone at every place, no Z-stack."""
    for layer in config_helpers.get_channel_configs(session.settings):
        session.set_channel_settings(layer, acquire='image' if layer == 'BF' else None)
        session.set_channel_settings(layer, autofocus=False)
    return session.new_protocol(tiling=tiling, use_zstacking=False)


def _every_step(protocol):
    return list(range(len(protocol.steps())))


def test_a_one_scan_run_of_single_stills_is_told_nothing_to_stack(session, tmp_path):
    protocol = _bf_protocol(session)
    folder = _run_folder(tmp_path, protocol, _scans(_every_step(protocol), [0]))

    answer = _build(folder)

    assert answer['new_count'] == 0
    assert answer['artifact_paths'] == []
    words = str(HyperstacksSavedNotice(answer))
    assert words.startswith('Nothing to stack: no place in ')
    assert 'its images are saved' in words
    assert f'{len(protocol.steps())} single-image group(s) skipped.' in words


def test_a_one_scan_tiled_run_of_single_stills_is_told_nothing_to_stack(session, tmp_path):
    # Each tile is its own stack, so one still a tile has nothing to combine.
    protocol = _bf_protocol(session, tiling='2x2')
    folder = _run_folder(tmp_path, protocol, _scans(_every_step(protocol), [0]))

    answer = _build(folder)

    assert answer['new_count'] == 0
    assert str(HyperstacksSavedNotice(answer)).startswith('Nothing to stack: ')


def test_a_place_whose_one_capture_failed_is_counted_as_having_no_image(session, tmp_path):
    protocol = _bf_protocol(session)
    steps = _every_step(protocol)
    folder = _run_folder(tmp_path, protocol, _scans(steps[1:], [0]), failed=[(steps[0], 0)])

    answer = _build(folder)

    words = str(HyperstacksSavedNotice(answer))
    assert answer['new_count'] == 0
    assert f'{len(steps) - 1} single-image group(s) skipped.' in words
    # The place is named, never left to the log to name.
    assert '1 single-image group(s) have no image in the folder: ' in words, words
    assert 'log' not in words, words


def test_a_run_with_one_place_to_stack_stacks_it_and_counts_the_singles(session, tmp_path):
    protocol = _bf_protocol(session)
    steps = _every_step(protocol)
    first = _steps_of(protocol, _wells(protocol, 1)[0])
    folder = _run_folder(tmp_path, protocol, _scans(steps, [0]) + _scans(first, [1]))

    answer = _build(folder)

    assert answer['new_count'] == 1
    assert _stack_shapes(answer['artifact_paths'])[0][0] == 2
    words = str(HyperstacksSavedNotice(answer))
    assert words.startswith('1 hyperstack(s) saved to ')
    assert f'{len(steps) - len(first)} single-image group(s) skipped.' in words


def test_a_z_projection_of_single_stills_is_still_refused(session, tmp_path):
    protocol = _bf_protocol(session)
    folder = _run_folder(tmp_path, protocol, _scans(_every_step(protocol), [0]))

    with pytest.raises(PostProcessingRefusedError) as refused:
        ZProjector(has_turret=False).load_folder(path=folder, tiling_grids=shipped_grids())

    assert refused.value.reason == 'no_data'
