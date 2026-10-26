# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A post-processing build's log names each group by what it holds.

Every build logged its groups with the first image's well, colour and tile
group, and counted the images as tiles: a BF plus LED 0 composite read
``color=BF ... tiles=2``. A group is named by every value it holds in each of
those columns and counts its images, whichever build made it.
"""

import pytest

import modules.protocol_post_processor as protocol_post_processor
from tests.test_adding_a_step_is_an_api_capability import (  # noqa: F401 -- pytest fixture
    turret_in_a_known_slot,
)
from tests.test_an_add_is_one_channel_group import _added, _composite, _run_folder
from tests.test_protocol_execution import scope  # noqa: F401 -- pytest fixture
from tests.stock_channels import LED0


@pytest.mark.usefixtures('turret_in_a_known_slot')
def test_a_two_channel_composite_group_is_logged_with_both_channels(scope, tmp_path, monkeypatch):
    told = []
    monkeypatch.setattr(
        protocol_post_processor.logger, 'info', lambda message, *a, **k: told.append(message)
    )

    _composite(_run_folder(tmp_path, _added(scope, adds=(('BF', LED0),))))

    (started,) = [line for line in told if 'group start' in line]
    assert f'color=BF+{LED0}' in started
    assert 'images=2' in started
    assert 'tiles=' not in started
