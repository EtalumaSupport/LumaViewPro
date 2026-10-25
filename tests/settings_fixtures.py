# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Settings for a factory-built session, the shipped template under a test's overrides.

A session's factory completes the dict it is handed from the shipped
template, as the file's preparation does, so a test may hand it any
dict. ``complete_settings`` is the convenience for a test that wants the
template's values with a few of its own laid over, said by name, and the
fast simulator tier the suite runs on. Its live folder is a fresh
temporary one unless the test names its own: the template's is relative
to the installation, and the door would create it under the checkout.

The template is read through ``settings_init.load_template``, the one
read of it, so what the application sees of the template the fixtures
see too.
"""

import copy
import logging
import pathlib
import tempfile

from modules import settings_init, settings_paths
from modules.python_values import python_values

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_TEMPLATE = _REPO_ROOT / 'data' / 'settings.json'
_LOGGER = logging.getLogger('tests.settings_fixtures')

# Where a fixture's live folders go: the test session's temporary root,
# set by ``tests/conftest.py`` for the session, else the system's.
LIVE_FOLDER_ROOT: pathlib.Path | None = None


def shipped_template() -> dict:
    """The shipped template as the application reads it."""
    return settings_init.load_template(_LOGGER, str(_TEMPLATE))


def _fresh_live_folder() -> str:
    root = LIVE_FOLDER_ROOT or pathlib.Path(tempfile.mkdtemp(prefix='lvp_live_folders_'))
    return tempfile.mkdtemp(prefix='live_', dir=root)


def complete_settings(**overrides) -> dict:
    """The shipped template with ``overrides`` laid over it.

    Overrides win at every depth; keys the overrides lack come from the
    template. Turret slot keys are normalized to ints on both sides
    BEFORE the merge -- merged afterwards, the template's string keys
    would sit beside an int-keyed override and the later normalization
    would let the template's ``None`` win.
    """
    merged = copy.deepcopy(overrides)
    template = shipped_template()
    # The template ships the firmware tier; the suite runs the fast one
    # unless a test asks for the other by name. Moving a test class to the
    # firmware tier is a decision taken per class, not by the template.
    template['simulator_tier'] = 'fast'
    if 'live_folder' not in merged:
        merged['live_folder'] = _fresh_live_folder()
    settings_init._normalize_turret_slot_keys(merged)
    settings_init._normalize_turret_slot_keys(template)
    settings_init._deep_merge_defaults(merged, template)
    return merged


def settings_writer(settings: dict):
    """An ``update_settings(path, value)`` for a fake context holding ``settings``.

    A GUI unit test stands a ``SimpleNamespace`` in for the context; the
    widget under test now writes through ``ctx.update_settings``. This does
    to ``settings`` what ``ScopeSession.update_settings`` does to the live
    dict -- the same check against the shipped template, then the store --
    so a widget that writes a path the Session would refuse fails here too.
    """
    template = shipped_template()

    def update_settings(path: str, value: object) -> None:
        # As the Session's: a closed set's member is written as its plain string.
        value = python_values(value)
        settings_paths.check_write(template, path, value)
        *blocks, leaf = path.split('.')
        holder = settings
        for block in blocks:
            holder = holder[block]
        holder[leaf] = value

    return update_settings


def protocol_filepath_writer(settings: dict):
    """A ``set_protocol_filepath(file_path)`` for a fake context holding ``settings``."""

    def set_protocol_filepath(file_path: str) -> None:
        settings['protocol']['filepath'] = file_path

    return set_protocol_filepath
