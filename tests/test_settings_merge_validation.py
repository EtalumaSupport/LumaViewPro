# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Keys the defaults merge supplies are validated after the merge, not before.

A settings file written by an older build legitimately lacks keys a newer
build introduced; the defaults merge is what supplies them. Validating those
keys at read time would reject every carried-forward file, so the check has to
run on the merged result -- the dict callers actually index.

The third test is the safety interlock on the other direction: registering a
key that data/settings.json does not seed would abort startup for every user,
since the merge could never supply it.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from modules.settings_init import _MERGE_SUPPLIED_KEYS, _validate_merged_settings


class _Logger:
    def __init__(self):
        self.messages = []

    def debug(self, msg):
        self.messages.append(msg)


def _complete_settings():
    return dict.fromkeys(_MERGE_SUPPLIED_KEYS, False)


def test_missing_merge_supplied_key_raises_and_names_it():
    settings = _complete_settings()
    settings.pop('preview_host_downscale')

    with pytest.raises(ValueError) as excinfo:
        _validate_merged_settings(settings, _Logger())

    # The operator has to know WHICH key and what to do about it.
    assert 'preview_host_downscale' in str(excinfo.value)
    assert 'settings.json' in str(excinfo.value)


def test_complete_settings_pass():
    _validate_merged_settings(_complete_settings(), _Logger())


def test_every_registered_key_is_seeded_in_shipped_defaults():
    defaults = json.loads(
        (pathlib.Path(__file__).resolve().parent.parent / 'data' / 'settings.json').read_text()
    )
    unseeded = _MERGE_SUPPLIED_KEYS - defaults.keys()

    assert not unseeded, (
        f'{sorted(unseeded)} are validated after the defaults merge but absent from '
        'data/settings.json, so the merge can never supply them -- every install would '
        'fail to start. Seed them in the defaults or drop them from the registry.'
    )
