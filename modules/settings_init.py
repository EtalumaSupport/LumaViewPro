# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
import copy
import os
import json
import logging
import pathlib
import time
from typing import ClassVar

from lib.lumascope_error import Fault, RefusalCause
from modules import labware_loader
from modules.exceptions import (
    ArgumentRefusedError,
    InstallationFileError,
    SettingsFileNotReplacedError,
    StoredSettingReplacedNotice,
)
from modules.path_utils import read_installation_file
from modules.python_values import python_values

# A setting's value: what one leaf of the settings file holds.
type SettingValue = str | int | float | bool | list | dict | None

settings = None

# The stored values the last preparation replaced, as notices not yet
# reported. The preparation runs before any host has a listener to show them
# to, so the Session reports them once it has one.
_stored_replacements: list[StoredSettingReplacedNotice] = []

debug_setting = None

# Which file load_debug_setting() actually read debug_mode from
# (current.json or settings.json basename), so the startup banner can
# state the source. Editing the wrong file is a common confusion -- the
# live value comes from current.json once it exists, not settings.json.
debug_setting_source = None

# Required top-level keys that must exist in a valid settings file.
# Missing keys cause hard-to-debug runtime errors downstream.
_REQUIRED_SETTINGS_KEYS = frozenset(
    {
        'microscope',
        'live_folder',
        'frame',
    }
)


class SettingsFileError(Fault, ValueError):
    """A settings file exists but could not be turned into a settings dict.

    Subclasses ValueError deliberately: `load_lvp_settings` routes a bad
    `current.json` to the shipped template by catching
    `(JSONDecodeError, ValueError)`, and a sibling of that family would
    escape the catch and turn every recoverable failure into a startup
    crash.
    """

    causes: ClassVar[dict[str, RefusalCause | None]] = {'settings_file_unreadable': None}
    title = 'Settings File Unreadable'


# Layer sub-keys whose names changed when the storage dict became the L2 API
# surface. Old name -> new name. `stim_config.illumination` lives one level
# down and is handled alongside; only the fluorescence channels carry a stim_config.
_RENAMED_LAYER_KEYS = {
    'ill_ma': 'illumination_ma',
    'exp_ms': 'exposure_ms',
}
_RENAMED_STIM_KEYS = {
    'illumination': 'illumination_ma',
}
# The recording's duration cap, named for its unit.
_RENAMED_VIDEO_KEYS = {
    'max_duration_seconds': 'max_duration_s',
}


def _migrate_renamed_keys(container: dict, mapping: dict) -> bool:
    """Move any old-named keys in one dict to their new names.

    ASSIGNS rather than setdefault, and that is the whole point. A build
    carrying this function never WRITES an old name, so finding one proves
    an older build wrote this file more recently than whatever new-named
    value sits beside it -- which happens when a user downgrades, changes a
    value, and upgrades again. Keeping the new-named value there would
    silently discard the edit they just made.
    """
    moved = False
    for old_name, new_name in mapping.items():
        if old_name not in container:
            continue
        container[new_name] = container.pop(old_name)
        moved = True
    return moved


def migrate_layer_key_names_dict(settings_dict: dict) -> bool:
    """Carry per-layer illumination and exposure keys to their unit-suffixed names.

    `settings[layer]['ill_ma'/'exp_ms']` -> `illumination_ma`/`exposure_ms`,
    and `stim_config['illumination']` -> `illumination_ma`. The storage dict
    is the L2 API surface, so its keys are the names callers write; these
    spellings had to match what `get_layer_configs` already emitted.

    Must run before the settings.json default-merge: the merge only ADDS
    missing keys, so without this fold an install carrying ill_ma = 150
    would get the shipped illumination_ma = 5.0 merged in beside it and
    come up on the default while the real value sat unread.

    Layers are found by SHAPE, not by importing get_layers: this runs during
    logger bootstrap, where importing modules.common_utils raises
    `cannot import name 'get_layers' from partially initialized module`.

    Returns:
        True when at least one key was moved.
    """
    moved = False
    for value in settings_dict.values():
        if not isinstance(value, dict):
            continue
        if _migrate_renamed_keys(value, _RENAMED_LAYER_KEYS):
            moved = True
        stim = value.get('stim_config')
        if isinstance(stim, dict) and _migrate_renamed_keys(stim, _RENAMED_STIM_KEYS):
            moved = True
    return moved


def read_settings_json(path: str, logger: logging.Logger | None = None) -> dict:
    """Open and parse one settings file. THE one place these files are read.

    Every reader of `current.json` / `settings.json` goes through here so
    that "what counts as an unusable settings file" is decided once. What
    each caller DOES about it still belongs to the caller -- they disagree
    for good reasons (the bootstrap readers try the next file, the report
    generator falls back to other directories, the GUI asks the user), so
    this classifies and they choose.

    No `encoding=` argument, deliberately. Every reader and the writer,
    `settings_saver.write_whole`, use the platform default, which is
    cp1252 on Windows. Reading as UTF-8 here would make a config that
    Windows wrote with any non-ASCII byte -- a live_folder under an
    accented user directory is the everyday case -- suddenly unparseable,
    and the caller would then offer to reset it. Changing this means
    migrating the read and the write together.

    `logger` is optional because `load_debug_setting` runs during logger
    bootstrap, before there is a logger to pass.

    Raises:
        FileNotFoundError: passed through untouched. Callers distinguish
            "no file" from "bad file" -- a missing file is normal on a
            fresh install, and at least one caller catches exactly this
            type to substitute an empty config.
        SettingsFileError: the file exists but did not yield a settings
            dict -- unparseable, undecodable, unreadable, or valid JSON
            that isn't an object (a top-level list used to raise
            AttributeError deep in validation and kill startup).
    """
    try:
        with open(path) as read_file:
            parsed = json.load(read_file)
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as e:
        raise SettingsFileError(f'{path}: not valid JSON ({e})') from e
    except (OSError, UnicodeDecodeError) as e:
        raise SettingsFileError(f'{path}: could not be read ({e})') from e

    if not isinstance(parsed, dict):
        raise SettingsFileError(f'{path}: expected a JSON object, got {type(parsed).__name__}')
    # Every reader lands here, which is why the key migration lives here and
    # not in load_settings: the GUI bootstrap, ScopeSession.load_user_settings
    # (which reads the file itself and never calls load_settings), the
    # support-report generator and app_config all go through this function.
    # Running before the caller's validation also means validation never
    # sees, or warns about, the old spellings.
    if migrate_layer_key_names_dict(parsed) and logger is not None:
        logger.info(f'[Settings ] {path}: carried layer keys to their unit-suffixed names')
    if logger is not None:
        logger.debug(f'[Settings ] read {path}')
    return parsed


def _missing_required_keys(settings: dict) -> list[str]:
    """The keys the app cannot start without that *settings* lacks, sorted."""
    return sorted(_REQUIRED_SETTINGS_KEYS - settings.keys())


def _validate_settings(settings: dict, filepath: str, logger) -> None:
    """Check that loaded settings contain all required keys and types.

    Raises on missing critical keys. Warns on missing optional keys or
    type mismatches -- allows the app to start with partial config.
    """
    missing = _missing_required_keys(settings)
    if missing:
        raise ValueError(
            f'[Settings ] {filepath} missing required keys: {missing}. '
            'App cannot start without these keys.'
        )

    # Type checks for critical nested structures
    if 'frame' in settings:
        frame = settings['frame']
        if not isinstance(frame, dict):
            logger.warning(
                f'[Settings ] {filepath}: "frame" should be a dict, got {type(frame).__name__}'
            )
        else:
            for field in ('width', 'height'):
                if field not in frame:
                    logger.warning(f'[Settings ] {filepath}: "frame" missing "{field}"')
                elif not isinstance(frame[field], int):
                    logger.warning(
                        f'[Settings ] {filepath}: "frame.{field}" should be int, got {type(frame[field]).__name__}'
                    )

    # Validate layer settings have expected structure
    from modules.common_utils import get_layers

    _REQUIRED_LAYER_FIELDS = {
        'illumination_ma': (int, float),
        'gain_db': (int, float),
        'exposure_ms': (int, float),
        'acquire': (str, type(None)),
        'autofocus': bool,
        'false_color': (bool, list),
        'focus': (int, float, type(None)),
    }
    for layer in get_layers():
        if layer not in settings:
            logger.warning(f'[Settings ] {filepath}: missing layer "{layer}"')
            continue
        layer_settings = settings[layer]
        if not isinstance(layer_settings, dict):
            logger.warning(f'[Settings ] {filepath}: "{layer}" should be dict')
            continue
        for field, _expected_type in _REQUIRED_LAYER_FIELDS.items():
            if field not in layer_settings:
                logger.warning(f'[Settings ] {filepath}: "{layer}" missing "{field}"')

    # Validate motion settings
    if 'motion' in settings:
        if not isinstance(settings['motion'], dict):
            logger.warning(f'[Settings ] {filepath}: "motion" should be dict')
        elif 'acceleration_max_pct' not in settings['motion']:
            logger.warning(f'[Settings ] {filepath}: "motion" missing "acceleration_max_pct"')


def _check_container_shape(current, template, path=''):
    """Compare a loaded config against the shipped one, shape only.

    settings.json already describes the structure the rest of the app
    assumes -- 149 places index into these dicts without checking -- so it
    serves as the schema and cannot drift from itself the way a
    hand-maintained one would.

    Only container KIND is compared: a dict where the template has a dict,
    a list where it has a list. Scalars are never inspected, because int
    and float are interchangeable across this file (four per-layer fields
    are declared as either) and null is a legitimate "unset". Rejecting on
    scalar type would throw away configurations that work today, and the
    cost of a false rejection is a user being offered a reset.

    Keys absent from the config are fine -- the default merge fills them,
    and an install predating a migration legitimately lacks whole blocks.
    Keys absent from the TEMPLATE are fine too: users and plugins may hold
    extra ones.

    Returns the mismatches, deepest key path first, each as (the dotted
    path, the kind the template holds, the kind found, the value found).
    """
    problems = []
    for key, template_value in template.items():
        if key not in current:
            continue
        value = current[key]
        where = f'{path}.{key}' if path else key
        for kind in (dict, list):
            if isinstance(template_value, kind) and not isinstance(value, kind):
                problems.append((where, kind.__name__, type(value).__name__, value))
                break
        else:
            if isinstance(template_value, dict):
                problems.extend(_check_container_shape(value, template_value, where))
    return problems


def _deep_merge_defaults(current: dict, defaults: dict, path: str = '', logger=None) -> list[str]:
    """Recursively merge missing keys from defaults into current.

    Only adds keys that are absent in current -- never overwrites existing
    values. Returns list of keys that were added (for logging).
    """
    added = []
    for key, default_value in defaults.items():
        full_key = f'{path}.{key}' if path else key
        if key not in current:
            current[key] = default_value
            added.append(full_key)
        elif isinstance(default_value, dict) and isinstance(current[key], dict):
            added.extend(_deep_merge_defaults(current[key], default_value, full_key, logger))
    return added


def migrate_video_settings_dict(settings_dict: dict) -> bool:
    """Carry a configured manual_video section to its new name, video.

    The rate/duration authority applies to every recording path, not just
    manual record, so the section renamed. Must run on a loaded dict
    before the settings.json default-merge: the merge only ADDS missing
    keys, so without this fold an install carrying manual_video.max_fps
    = 10 would get the shipped video.max_fps = 0 merged in and silently
    lose its configured cap.

    Returns:
        True when a manual_video section was found and folded.
    """
    old = settings_dict.pop('manual_video', None)
    if old is None:
        return False
    video = settings_dict.setdefault('video', {})
    for key, value in old.items():
        video.setdefault(key, value)
    return True


def migrate_video_key_names_dict(settings_dict: dict) -> bool:
    """Carry the video section's keys to their unit-suffixed names.

    Must run after `migrate_video_settings_dict`, so a key folded from
    manual_video is renamed with the rest, and before the settings.json
    default-merge, which would otherwise merge the shipped
    max_duration_s = 300 in beside a stored max_duration_seconds.

    Returns:
        True when at least one key was moved.
    """
    video = settings_dict.get('video')
    return isinstance(video, dict) and _migrate_renamed_keys(video, _RENAMED_VIDEO_KEYS)


# Set when current.json could not be used and the app came up on the shipped
# template instead. Holds the rejected file's path and why, until a human
# decides what to do about it.
#
# Read it as `settings_init.rejected_current_json`, never via
# `from modules.settings_init import rejected_current_json`. Several modules
# import `settings` that second way, which copies the value at import time --
# for a dict that is harmless, for a flag that changes during startup it would
# freeze the answer to whatever it was before the settings even loaded.
rejected_current_json = None


# Written into a layer whose video_config arrived absent or null. The shipped
# template carries the same pair, so an untouched install never reaches
# these; a rate or duration the recorder cannot use is the writer's range
# (``settings_paths._RANGES``), replaced at load and told once.
DEFAULT_VIDEO_DURATION_SEC = 5
DEFAULT_VIDEO_FPS = 30


def normalize_loaded_settings(settings_dict: dict) -> bool:
    """Repair loaded values the running version cannot use as written.

    Distinct from the default merge, which only ADDS absent keys. These
    keys are PRESENT and hold something the app would misread: a retired
    spinner label, a video config written as null. The merge cannot see any
    of them, because nothing is missing. A value outside a setting's range
    is not repaired here: the writer's own rule replaces it after the merge
    and reports it (``settings_paths.replace_refused_stored_values``).

    Returns:
        True when at least one value was repaired.
    """
    # Deferred to break an import cycle: lvp_logger imports load_debug_setting
    # from this module at its module top (before lvp_logger.logger is defined),
    # and both of these import lvp_logger.logger at their own top.
    from modules import image_mode
    from modules.common_utils import get_layers
    from modules.zstack_config import ZStackReference

    changed = False

    output_format = settings_dict.get('image_output_format')
    if isinstance(output_format, dict) and output_format.get('sequenced') == 'ImageJ Hyperstack':
        # The file written never changed -- it was always OME-TIFF. Only the
        # label did, because it named a reader instead of the format.
        output_format['sequenced'] = image_mode.OutputFormat.HYPERSTACK.value
        changed = True

    # The z-stack's reference was stored as the spinner's label. It is stored
    # as its ZStackReference value now, and the label is the GUI's own, which
    # may change; these are the labels earlier releases wrote, which do not.
    zstack = settings_dict.get('zstack')
    stored_label = zstack.get('position') if isinstance(zstack, dict) else None
    retired_labels = {
        'Current Position at Top': ZStackReference.TOP,
        'Current Position at Center': ZStackReference.CENTER,
        'Current Position at Bottom': ZStackReference.BOTTOM,
    }
    if isinstance(stored_label, str) and stored_label in retired_labels:
        zstack['position'] = retired_labels[stored_label]
        changed = True

    # Protocol accordions are permanently enabled; a stored preference for
    # the retired toggle would be read by nothing.
    if settings_dict.pop('disable_protocol_accordions', None) is not None:
        changed = True

    # Reusing brightfield's autofocus result for the fluorescence steps was
    # removed, never shown: it skipped their autofocus without moving to
    # that focus, and carried one well's result to the next.
    protocol = settings_dict.get('protocol')
    if isinstance(protocol, dict) and protocol.pop('bf_af_for_fluorescence', None) is not None:
        changed = True

    # The longest an auto-gain step waited to converge, unread since its
    # settle became a frame count: the camera's auto-gain refuses a key it
    # does not take, so a stored one would refuse every arm.
    autogain = protocol.get('autogain') if isinstance(protocol, dict) else None
    if isinstance(autogain, dict) and autogain.pop('max_duration_seconds', None) is not None:
        changed = True

    # Seven shipped settings that nothing read: stimulation profiling, whose
    # producer is gone, and the autofocus and stage blocks, which named
    # features not built. A stored copy was readable, unwritable and never
    # range-checked, so it goes with the template's.
    profiling = settings_dict.get('profiling')
    if isinstance(profiling, dict) and profiling.pop('stim_profiling', None) is not None:
        changed = True
    for block in ('autofocus', 'stage'):
        if settings_dict.pop(block, None) is not None:
            changed = True

    # The REST server's host, key and CORS origins, which nothing read: it
    # binds this computer only and has no credentials. A shipped null key is
    # dropped too, so the test is membership, not the value.
    rest_api = settings_dict.get('rest_api')
    if isinstance(rest_api, dict):
        for key in ('host', 'api_key', 'cors_origins'):
            if key in rest_api:
                del rest_api[key]
                changed = True

    # The slot a person last turned the turret to, once preferred when two
    # slots carry one objective. Each protocol step now records its own
    # slot, and nothing reads this one.
    if settings_dict.pop('turret_position', None) is not None:
        changed = True

    # Unconfirmed, as stored before the confirmation recorded its topology.
    if settings_dict.get('objective_confirmed') is False:
        settings_dict['objective_confirmed'] = None
        changed = True

    # The camera driver once chosen in the settings, which nothing read:
    # bring-up asks the camera registry, and a host that wants one driver
    # names it to Lumascope itself.
    if settings_dict.pop('camera_type', None) is not None:
        changed = True

    # A top-level binning factor that nothing writes and no shipped template
    # carries. It was read as the headless binning and always answered 1;
    # binning lives at settings['binning']['size'] as the selector's label.
    # Dropping it means a file that somehow carries one cannot be mistaken
    # for a second, disagreeing source of the same fact.
    if settings_dict.pop('binning_size', None) is not None:
        changed = True

    # A plate the catalogue has renamed since this file named it, folded to
    # the key so the one store every reader trusts never carries a spelling
    # the catalogue lacks. Whether the plate exists at all is answered where
    # it is selected, against the catalogue -- not here, where refusing
    # would mean discarding the whole file.
    protocol_settings = settings_dict.get('protocol')
    if isinstance(protocol_settings, dict):
        stored_plate = protocol_settings.get('labware')
        folded_plate = labware_loader.canonical_plate_name(stored_plate)
        if folded_plate != stored_plate:
            protocol_settings['labware'] = folded_plate
            changed = True

    for layer in get_layers():
        layer_settings = settings_dict.get(layer)
        if not isinstance(layer_settings, dict):
            continue

        # A filename prefix every layer once shipped, which nothing reads:
        # a capture's name is built by the writer that saves it.
        if 'file_root' in layer_settings:
            del layer_settings['file_root']
            changed = True

        video_config = layer_settings.get('video_config')
        if not isinstance(video_config, dict):
            video_config = {}
            layer_settings['video_config'] = video_config
            changed = True
        if 'duration' not in video_config:
            video_config['duration'] = DEFAULT_VIDEO_DURATION_SEC
            changed = True
        if 'fps' not in video_config:
            video_config['fps'] = DEFAULT_VIDEO_FPS
            changed = True

    return changed


# The focus every layer shipped with before the template stopped carrying
# one. It was merged into each current.json for every layer nobody saved, so
# a stored focus of exactly this value is a channel whose focus was never set.
# Every writer stores a measured stage Z, which does not land on it.
_RETIRED_SHIPPED_FOCUS_UM = 4950.0


def forget_shipped_focus(settings_dict: dict) -> list[str]:
    """Read a layer focus still holding the old shipped value as never saved.

    A layer with no saved focus takes the stage's current Z wherever a step
    is built for it; the shipped number made every unsaved layer look saved,
    so its steps went to that height instead.

    Returns:
        The layers whose focus was set to None, in layer order.
    """
    from modules.common_utils import get_layers

    forgotten = []
    for layer in get_layers():
        layer_settings = settings_dict.get(layer)
        if (
            isinstance(layer_settings, dict)
            and layer_settings.get('focus') == _RETIRED_SHIPPED_FOCUS_UM
        ):
            layer_settings['focus'] = None
            forgotten.append(layer)
    return forgotten


def take_stored_replacements() -> list[StoredSettingReplacedNotice]:
    """The replacements not yet reported, handed over once."""
    taken = list(_stored_replacements)
    _stored_replacements.clear()
    return taken


# The reserved top-level key the stock rename leaves in the dict it
# prepared: each old channel name it moved, and the stock key it moved to --
# names only. The session's factory takes it out before the dict is any
# session's store (``take_old_channel_blocks``), so it never reaches a file;
# the shape and range checks walk the template, which has no such key.
OLD_CHANNEL_BLOCKS = '_old_channel_blocks'


def take_old_channel_blocks(settings_dict: dict) -> dict[str, str]:
    """The old channel names the stock rename moved, each with its stock key,
    taken out of the prepared ``settings_dict``; empty when it moved none."""
    return settings_dict.pop(OLD_CHANNEL_BLOCKS, {})


def fold_old_channel_blocks(logger: logging.Logger, settings_dict: dict) -> None:
    """Rename each old channel block to its stock key, before any reader of a block.

    A file an earlier release wrote keys its fluorescence blocks by the old
    colour names. Every reader before the scope exists -- the validation,
    the focus retirement, the repairs, the merge, the range check -- reads
    the release's keys, so each old block becomes its stock key's here
    (``layer_record.release_old_names``) and is read there; the unit is not
    known yet, and its bring-up places them by its own LEDs
    (``ScopeSession._place_old_channel_blocks``), from the record this
    leaves under ``OLD_CHANNEL_BLOCKS``. Run again on its own result it
    changes nothing, and keeps the record.

    An old block whose stock key already holds one (a file written across a
    rollback, or by hand) is dropped, the stock key's kept, and the drop is
    told with the load's replacements: the release has no setting of that
    name.
    """
    from modules.layer_record import release_old_names

    moved = {}
    dropped = []
    for old in release_old_names():
        if old.name not in settings_dict:
            continue
        block = settings_dict.pop(old.name)
        if old.stock_key in settings_dict:
            dropped.append((old.name, block, None))
            continue
        settings_dict[old.stock_key] = block
        moved[old.name] = old.stock_key
    if moved:
        settings_dict.setdefault(OLD_CHANNEL_BLOCKS, {}).update(moved)
        logger.info(
            '[Settings ] Renamed the old channel settings to their stock keys: '
            + ', '.join(f'{old} -> {stock}' for old, stock in moved.items())
        )
    if dropped:
        _stored_replacements.append(StoredSettingReplacedNotice(dropped))


def _apply_load_migrations(logger, settings_dict: dict) -> None:
    """Every fold that must run on a loaded dict before the default merge.

    Order matters against the merge, not among themselves: the merge only
    ADDS missing keys, so a rename left unfolded here would get the shipped
    default merged in beside the user's value and the real one would sit
    unread.
    """
    from modules import image_mode

    fold_old_channel_blocks(logger, settings_dict)
    if image_mode.migrate_settings_dict(settings_dict):
        logger.info('[Settings ] Consolidated capture/save toggles into image_mode')
    if migrate_video_settings_dict(settings_dict):
        logger.info('[Settings ] Renamed manual_video settings section to video')
    if migrate_video_key_names_dict(settings_dict):
        logger.info('[Settings ] Renamed video.max_duration_seconds to video.max_duration_s')
    if normalize_loaded_settings(settings_dict):
        logger.info('[Settings ] Repaired stored values the running version cannot use')


def _retire_shipped_values(logger, settings_dict: dict) -> None:
    """Every stored value that is read as never set, on a FILE only.

    A retirement reads a value the template once shipped as the absence of
    a choice. It is the file's, never a handed dict's: a caller who hands
    that value asked for it.
    """
    forgotten = forget_shipped_focus(settings_dict)
    if forgotten:
        logger.info(
            f'[Settings ] No focus was ever saved for {", ".join(forgotten)}: '
            'their steps take the current Z'
        )


def _load_and_validate(logger, filepath: str) -> dict:
    """Read one settings file and check it carries the keys the app needs.

    Its old channel blocks are renamed first, so the check, and every
    reader after it, sees them under the release's keys.
    """
    loaded = read_settings_json(filepath, logger)
    fold_old_channel_blocks(logger, loaded)
    _validate_settings(loaded, filepath, logger)
    return loaded


def load_template(logger: logging.Logger, template_path: str) -> dict:
    """The shipped template, a block for every catalogue key, checked for the keys the app needs.

    Read as a file the installation ships, not as the user's: a template
    that is missing, will not parse, or cannot give every catalogue key a
    block is the installation's fault and raises ``InstallationFileError``,
    never a reason to skip the step that needed it.
    """
    template = read_installation_file(template_path)
    missing = _missing_required_keys(template)
    if missing:
        raise InstallationFileError(template_path, f'lacks the keys the app needs: {missing}')
    _give_every_layer_a_block(template, template_path)
    _give_the_shipped_step_order(template, template_path)
    _validate_settings(template, template_path, logger)
    return template


def _give_the_shipped_step_order(template: dict, template_path: str) -> None:
    """The order a new protocol's steps take, from the catalogue's ``step_order``.

    The catalogue's record is the one place a layer's step order is said, so
    a key added there takes its place without an edit here. A template that
    states its own order would be a second, unread source of it, and is
    refused. A stored order is not a person's either: no write reaches it
    (``settings_paths.DERIVED``), so one that differs is an earlier
    release's derivation, re-derived at load (``prepare_loaded``) and not
    told.

    Raises:
        InstallationFileError: the template states ``step_channel_order``.
    """
    # Deferred: layer_record imports lvp_logger, which imports this module.
    from modules.layer_record import release_layers

    if 'step_channel_order' in template:
        raise InstallationFileError(
            template_path,
            "states step_channel_order, which is each scopes.json LayerOrder record's step_order",
        )
    template['step_channel_order'] = [
        layer.key for layer in sorted(release_layers(), key=lambda layer: layer.step_order)
    ]


def _give_every_layer_a_block(template: dict, template_path: str) -> None:
    """Every catalogue key's block: its own, else a copy of its modality's default.

    The file names a block only for a key with values of its own; every
    other key takes ``channel_defaults``' block for its modality, so a key
    added to the catalogue has settings without an edit here. The defaults
    are consumed: no prepared dict, published template or saved file holds
    them, so the only blocks a caller sees are layers.

    Raises:
        InstallationFileError: ``channel_defaults`` is missing or is not a
            mapping of blocks, or has no block for the modality of a key
            the file names none for.
    """
    # Deferred: layer_record imports lvp_logger, which imports this module.
    from modules.layer_record import release_layers

    defaults = template.pop('channel_defaults', None)
    if not isinstance(defaults, dict) or not all(isinstance(b, dict) for b in defaults.values()):
        raise InstallationFileError(
            template_path, f'has no usable channel_defaults (found {defaults!r})'
        )
    unfilled = []
    for layer in release_layers():
        if layer.key in template:
            continue
        default = defaults.get(layer.modality)
        if default is None:
            unfilled.append(f'{layer.key} ({layer.modality})')
            continue
        template[layer.key] = copy.deepcopy(default)
    if unfilled:
        raise InstallationFileError(
            template_path, f'has no block and no channel_defaults block for {unfilled}'
        )


def _normalize_turret_slot_keys(settings: dict) -> None:
    """Turret slot keys become ints, because a turret position is a number.

    JSON object keys are strings whether the value is a number or not, so a
    position round-trips through the file as "1" and has to be converted back
    on the way in. Every consumer downstream works in ints -- the motion API
    subscripts the config with a live motor position, and its type hint says
    dict[int, str] -- so this is the single boundary where the storage type
    becomes the runtime type.

    It lives here rather than in the GUI's settings load because a headless or
    REST caller runs this pipeline and never runs that widget: with the
    conversion in the widget the two hosts disagreed about the key type, which
    put duplicate keys in the saved file and raised KeyError off the GUI.
    """
    slots = settings.get('turret_objectives')
    if not isinstance(slots, dict):
        return
    settings['turret_objectives'] = {int(k): v for k, v in slots.items()}


def bring_up_live_folder(live_folder: str, directory: str) -> tuple[str, OSError | None]:
    """The live folder as it is stored: absolute, and created.

    The one rule for the value wherever it enters -- the settings file at
    load, and ``ScopeSession.set_live_folder`` after. The shipped template
    holds a relative folder, which means the installation's; left relative,
    each writer would resolve it against the process's working directory,
    and an installed build's working directory is not writable.

    Returns:
        The folder as it is stored, and the error that kept it from being
        created, or None. What a folder that cannot be created means is the
        entry's: at load it stays the person's, at a write it is refused.

    Raises:
        ValueError: ``live_folder`` is a string no file system can name (a
            NUL byte).
    """
    folder = pathlib.Path(live_folder)
    if not folder.is_absolute():
        folder = (pathlib.Path(directory) / folder).resolve()
        live_folder = str(folder)
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return live_folder, e
    return live_folder, None


def prepare_settings(
    logger: logging.Logger, directory: str, *, fall_back_to_template: bool
) -> tuple:
    """Read the settings file and make it USABLE. Every host runs this.

    Reading the file is only the first step. A settings dict is not usable
    until its shape has been checked against the shipped template, its
    retired spellings folded forward, its unusable values repaired, and
    the keys added by newer releases merged in. A host that runs only the
    read gets a dict that parses and is silently missing everything the
    running version added since the file was written.

    ``fall_back_to_template`` answers the one question with no
    host-independent answer: what to do when current.json exists and
    cannot be used. The GUI comes up on the shipped defaults and asks the
    user. A headless caller has nobody to ask, and handing it a plausible
    configuration that is not the user's is worse than refusing, so it
    gets the exception.

    Returns:
        (settings dict, rejected) where rejected is None, or
        (path, reason) naming the user's file that was set aside.
    """
    current_path = current_json_path(directory)
    template_path = os.path.join(directory, 'data', 'settings.json')
    data_dir = os.path.join(directory, 'data')
    rejected = None

    if os.path.exists(current_path):
        # Read first, outside the except below: a template that will not
        # parse is the installation's fault (``InstallationFileError``) and
        # is never reported as "current.json could not be used", which
        # would send the user to delete the one file that was still good.
        template = load_template(logger, template_path)
        try:
            prepared = _load_and_validate(logger, current_path)
            _retire_shipped_values(logger, prepared)
            prepared = prepare_loaded(
                logger, prepared, template, directory, source=current_path, handed=False
            )
        except (json.JSONDecodeError, ValueError) as e:
            if not fall_back_to_template:
                raise
            # current.json is unusable. Come up on the shipped template so the
            # user gets a working app and a chance to decide -- but do NOT
            # touch their file. It is the only copy of their configuration,
            # and the running app is now holding template values that would
            # overwrite it on the next save.
            logger.error(
                f'[Settings ] {current_path} could not be used ({e}); '
                'starting from the shipped defaults. The file has NOT been '
                'modified and no settings will be saved until this is resolved.'
            )
            prepared = prepare_loaded(
                logger,
                copy.deepcopy(template),
                template,
                directory,
                source=template_path,
                handed=False,
            )
            rejected = (current_path, str(e))
        return prepared, rejected

    if os.path.exists(template_path):
        template = load_template(logger, template_path)
        prepared = prepare_loaded(
            logger,
            copy.deepcopy(template),
            template,
            directory,
            source=template_path,
            handed=False,
        )
        return prepared, None

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Couldn't find 'data' directory at {data_dir}")
    raise FileNotFoundError(f'No settings files found in {data_dir}')


def prepare_loaded(
    logger: logging.Logger,
    loaded: dict,
    template: dict,
    directory: str,
    *,
    source: str,
    handed: bool,
) -> dict:
    """The one preparation every settings dict gets: ``loaded`` made USABLE against ``template``.

    Reading a file, or taking a dict from a caller's hand, is only the
    first step. In order: the containers are held to the template's kinds
    (the merge would not repair a mismatch, it only recurses where both
    sides are dicts, so this comes first); the retired spellings and the
    values the running version cannot use are folded forward; a numpy
    scalar or a closed set's member, a caller's or a fold's, at any depth,
    is made the plain value it holds, so the store and the file written
    from it hold plain data; the turret
    slot keys become ints on BOTH sides before the merge, because merged
    afterwards the template's string keys would sit beside the stored int
    keys and the normalization that followed would let the template's
    ``None`` win over every stored assignment; the keys newer releases
    added are merged in; a stored derived value is made the release's
    derivation; the values are held to the writer's ranges; the
    live folder is brought up. Run again on its own result it changes
    nothing, so a dict the file's preparation already made usable goes
    through a session's door unchanged.

    ``source`` is what a refusal names: the file's path, or the name of
    the argument a caller handed the dict as. ``handed`` is where the
    ranges row's two contracts part: a stored value the writer refuses is
    replaced by the shipped one and queued to be told once, since nobody
    can fix a file at load; a handed value is a write of every key at once
    and is refused as a write is. A container of the wrong kind is the
    file's ``SettingsFileError``, naming the file, or the argument's
    ``ArgumentRefusedError('wrong_kind')``, naming the key. ``handed`` is
    also ``ScopeSession.create``'s door, its one caller that passes True,
    and every host's dict passes it, so a live folder that cannot be
    created is warned about there alone: the drive, not the dict, holds
    that condition, so a second preparation would say it again.

    ``template`` is ``load_template``'s answer, never read a second way;
    it is copied here, so the merge never aliases its blocks into the
    store.

    Returns:
        ``loaded``, prepared in place.

    Raises:
        SettingsFileError: a file whose containers are not the template's
            kinds.
        ArgumentRefusedError: ``'wrong_kind'``, a handed dict's container
            that is not the template's kind; ``argument`` is its dotted
            path under ``source``.
        SettingRefusedError: a handed dict's value the writer refuses, with
            the write's own reason (``'out_of_range'``, ``'wrong_kind'``,
            ``'not_a_number'``).
    """
    template = copy.deepcopy(template)
    _normalize_turret_slot_keys(template)
    problems = _check_container_shape(loaded, template)
    if problems:
        if handed:
            where, want, _got, value = problems[0]
            raise ArgumentRefusedError(
                'wrong_kind', argument=f'{source}.{where}', value=value, takes=f'a {want}'
            )
        raise SettingsFileError(
            f'{source}: structure does not match settings.json -- '
            + '; '.join(f'{where}: expected {want}, got {got}' for where, want, got, _ in problems)
        )
    _apply_load_migrations(logger, loaded)
    # After the migrations, so a value one writes (a migrated image mode) is
    # plain data too, as a value a caller handed is.
    python_values(loaded)
    _normalize_turret_slot_keys(loaded)
    added = _deep_merge_defaults(loaded, template, logger=logger)
    if added:
        logger.info(f'[Settings ] Merged {len(added)} missing keys from settings.json: {added}')
    # Imported here: settings_paths' own imports reach this module (lvp_logger).
    from modules.settings_paths import (
        DERIVED,
        refuse_handed_values,
        replace_refused_stored_values,
    )

    if handed:
        refuse_handed_values(loaded, template)
    else:
        # A stored derived value is the release's, never a person's: no write
        # reaches one, so one that differs is an earlier release's
        # derivation (a step order in its own keys), re-derived and not told.
        # A handed one is a write of it, refused as a write is.
        for path in DERIVED:
            *blocks, leaf = path.split('.')
            stored, shipped = loaded, template
            for block in blocks:
                stored, shipped = stored[block], shipped[block]
            stored[leaf] = copy.deepcopy(shipped[leaf])
        replaced = replace_refused_stored_values(loaded, template)
        if replaced is not None:
            _stored_replacements.append(replaced)
    # A folder that cannot be created stays the person's: replacing it would
    # send their captures somewhere they will not look and save the
    # replacement over their choice. Captures into it are refused by the
    # capture-location owner, naming it, until it is reachable.
    loaded['live_folder'], unreachable = bring_up_live_folder(loaded['live_folder'], directory)
    if unreachable is not None and handed:
        logger.warning(
            f'[Settings ] The live folder {loaded["live_folder"]} could not be created '
            f'({unreachable}); captures into it are refused until it is reachable.'
        )
    return loaded


def load_lvp_settings(logger: logging.Logger, lvp_appdata: str) -> None:
    """Prepare the settings and publish them as this process's module state.

    The preparation itself is prepare_settings, which every host shares.
    What is specific to the app is the publishing: the GUI and the modules
    it imports read `settings_init.settings` directly, so a load has to
    land there as well as be returned.
    """
    global settings, rejected_current_json

    # Reset per call: a second load (tests) must not inherit the first's
    # verdict, and a load that raises must not leave the previous dict in
    # place looking like a successful one.
    settings = None
    rejected_current_json = None

    settings, rejected_current_json = prepare_settings(
        logger, lvp_appdata, fall_back_to_template=True
    )


def fall_back_to_template(logger: logging.Logger, lvp_appdata: str, reason: str) -> None:
    """Republish the store from the shipped template after a LATE rejection.

    ``prepare_settings`` already runs this recovery for a current.json that
    will not parse or whose containers are the wrong kind. A value that
    parses, has the right container shape, and is STILL not usable -- a
    binning label naming no factor the arithmetic accepts -- cannot be
    caught there: it is only discovered later, when something tries to
    configure a scope from it. Same policy, later trigger.

    The store is mutated IN PLACE rather than rebound. The GUI and the
    modules it imports each hold their own name bound to this one dict, so
    rebinding here would leave every one of them on the rejected values
    while only this module saw the template -- the divergence the single
    store exists to prevent.

    Setting ``rejected_current_json`` is the half that keeps the promise:
    it makes the session provisional, so the session saves nothing and
    says so, instead of writing template values over the only copy of the
    user's configuration. The file itself is not touched.
    """
    global rejected_current_json

    current_path = current_json_path(lvp_appdata)
    template_path = os.path.join(lvp_appdata, 'data', 'settings.json')
    template = load_template(logger, template_path)

    logger.error(
        f'[Settings ] {current_path} could not be used ({reason}); '
        'starting from the shipped defaults. The file has NOT been '
        'modified and no settings will be saved until this is resolved.'
    )
    prepared = prepare_loaded(
        logger, copy.deepcopy(template), template, lvp_appdata, source=template_path, handed=False
    )

    settings.clear()
    settings.update(prepared)
    rejected_current_json = (current_path, reason)


def retire_rejected_current_json() -> pathlib.Path | None:
    """Move the unusable current.json aside so a fresh one can take its place.

    Renamed, never deleted: it is the user's only copy of their
    configuration, and support can often read what they had out of it even
    when the app could not. Returns the new path.

    Called only after a human has chosen to start over -- the rename is the
    point of no return for that file's role, and nothing should reach it by
    timeout, by a dismissed dialog, or by any other default.

    Raises:
        SettingsFileNotReplacedError: The rename failed. The settings stay
            provisional, so the question can be answered again.
    """
    global rejected_current_json
    if rejected_current_json is None:
        return None
    path, _reason = rejected_current_json
    stamp = time.strftime('%Y%m%d-%H%M%S')
    retired = pathlib.Path(f'{path}.rejected-{stamp}')
    try:
        os.replace(path, retired)
    except OSError as e:
        raise SettingsFileNotReplacedError(path, e) from e
    rejected_current_json = None
    logging.getLogger('lvp_logger').warning(
        f'[Settings ] settings reset by user choice; previous file kept at {retired}'
    )
    return retired


def settings_are_provisional() -> bool:
    """True while the app is running on defaults nobody has agreed to keep.

    Writing current.json in this state would replace a user's whole
    configuration with the template, so the session saves nothing while it
    holds.
    """
    return rejected_current_json is not None


def current_json_path(directory: str | os.PathLike) -> str:
    """The live user configuration of the installation at ``directory``.

    The one spelling of where it lives, for its reader here and for each
    host that names it to ``ScopeSession.create(settings_file=...)``.
    """
    return os.path.join(directory, 'data', 'current.json')


def _resolve_settings_path(directory):
    current_path = current_json_path(directory)
    settings_path = os.path.join(directory, 'data', 'settings.json')
    data_dir = os.path.join(directory, 'data')

    if os.path.exists(current_path):
        return current_path
    if os.path.exists(settings_path):
        return settings_path
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Couldn't find 'data' directory at {data_dir}")
    raise FileNotFoundError(f'No settings files found in {data_dir}')


def read_before_preparation(directory: str, key: str) -> tuple[object, str]:
    """One installation-only setting, read before the settings are prepared.

    For what a process needs before ``prepare_settings`` can run: the
    instance lock a second launch must meet before it touches the files,
    and the logger, profilers and tracer configured before anything logs.
    Read only, by preparation's own rule: ``current.json`` is live, and the
    shipped ``settings.json`` answers when ``current.json`` is missing,
    unusable (preparation reports that once it runs), or lacks the key (a
    key a later release added). Every other reader indexes the prepared
    store.

    Returns:
        The value, and the basename of the file it came from.

    Raises:
        FileNotFoundError: no shipped ``settings.json``.
        SettingsFileError: the shipped ``settings.json`` is unusable.
        KeyError: the shipped ``settings.json`` lacks ``key``.
    """
    data_dir = os.path.join(directory, 'data')
    try:
        live = read_settings_json(os.path.join(data_dir, 'current.json'))
    except (FileNotFoundError, SettingsFileError):
        live = {}
    if key in live:
        return live[key], 'current.json'
    return read_settings_json(os.path.join(data_dir, 'settings.json'))[key], 'settings.json'


def load_debug_setting(directory: str) -> bool:
    global debug_setting, debug_setting_source
    debug_setting, debug_setting_source = read_before_preparation(directory, 'debug_mode')
    return debug_setting


def load_profile_trace_setting(directory: str) -> dict:
    """profile_trace_enabled and profile_trace_output_dir, before preparation.

    Returns ``{"enabled": bool, "output_dir": str | None}``. Called from
    lib/profile_trace.py at import, as load_debug_setting is from the logger.
    """
    enabled, _ = read_before_preparation(directory, 'profile_trace_enabled')
    output_dir, _ = read_before_preparation(directory, 'profile_trace_output_dir')
    return {'enabled': bool(enabled), 'output_dir': output_dir or None}


def load_tracemalloc_setting(directory: str) -> bool:
    """tracemalloc_enabled, before preparation; read by modules/common_utils.py at import."""
    enabled, _ = read_before_preparation(directory, 'tracemalloc_enabled')
    return bool(enabled)


def load_memory_profile_setting(directory: str) -> dict:
    """memory_profile_enabled and memory_profile_interval_s, before preparation.

    Returns ``{"enabled": bool, "interval_s": float}``; read by
    lib/memory_profile.py.
    """
    enabled, _ = read_before_preparation(directory, 'memory_profile_enabled')
    interval_s, _ = read_before_preparation(directory, 'memory_profile_interval_s')
    return {'enabled': bool(enabled), 'interval_s': float(interval_s)}
