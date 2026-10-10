# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run's name is one plain name, never a path.

The run saves its protocol inside its folder under that name. ``prepare``
took it unchecked, so a name holding a separator or a drive -- from a
script, the SDK or a REST caller -- wrote the protocol file wherever the
name pointed, outside the run folder and the live folder. Such a name is
refused before anything is committed, told once, for every caller.
"""

from __future__ import annotations

import pytest

from modules.exceptions import ProtocolRunRefusedError
from modules.sequenced_capture_runner import SequencedCaptureRunMode
from tests.test_protocol_execution import (  # noqa: F401 -- pytest fixtures
    _make_autogain_settings,
    _make_image_capture_config,
    _make_single_step_protocol,
    executor,
    executors,
    scope,
)


def _prepare(executor, parent_dir, sequence_name):
    return executor.prepare(
        protocol=_make_single_step_protocol(),
        run_trigger_source='api_scan',
        run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
        sequence_name=sequence_name,
        image_capture_config=_make_image_capture_config(),
        autogain_settings=_make_autogain_settings(),
        parent_dir=parent_dir,
        max_scans=1,
    )


def _warnings(centre_posts, since):
    from modules.notification_center import Severity

    return [n for n in centre_posts[since:] if n.severity in (Severity.WARNING, Severity.ERROR)]


@pytest.mark.parametrize(
    'name',
    ['a/b', 'a\\b', 'C:x', 'C:\\x', '..', '/abs/evil', '../../../outside/evil'],
)
def test_a_name_that_is_a_path_is_refused_once_and_nothing_is_written(
    executor, tmp_path, centre_posts, name
):
    parent_dir = tmp_path / 'live' / 'ProtocolData'
    since = len(centre_posts)

    with pytest.raises(ProtocolRunRefusedError) as refusal:
        _prepare(executor, parent_dir, name)

    assert refusal.value.reason == 'sequence_name_invalid'
    assert repr(name) in refusal.value.message
    assert len(_warnings(centre_posts, since)) == 1
    assert sorted(p.name for p in tmp_path.rglob('*')) == []


@pytest.mark.parametrize('name', ['my run', 'exp..v2', 'plate.tsv', ''])
def test_a_plain_name_is_accepted(executor, tmp_path, name):
    plan = _prepare(executor, tmp_path / 'ProtocolData', name)

    assert plan.sequence_name == name
