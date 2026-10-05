# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
import copy
import os
import json
import logging
import pathlib
import time

from modules import labware_loader
from modules.exceptions import SettingsFileNotReplacedError, StoredSettingReplacedNotice
from modules.path_utils import read_installation_file


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


class SettingsFileError(ValueError):
    """A settings file exists but could not be turned into a settings dict.

    Subclasses ValueError deliberately: `load_lvp_settings` routes a bad
    `current.json` to the shipped template by catching
    `(JSONDecodeError, ValueError)`, and a sibling of that family would
    escape the catch and turn every recoverable failure into a startup
    crash.
    """


# Layer sub-keys whose names changed when the storage dict became the L2 API
# surface. Old name -> new name. `stim_config.illumination` lives one level
# down and is handled alongside; only Blue/Green/Red carry a stim_config.
_RENAMED_LAYER_KEYS = {
    'ill_ma': 'illumination_ma',
    'exp_ms': 'exposure_ms',
}
_RENAMED_STIM_KEYS = {
    'illumination': 'illumination_ma',
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

    No `encoding=` argument, deliberately. Every reader and the writer in
    `microscope_settings.save_settings` use the platform default, which is
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


def _validate_settings(settings: dict, filepath: str, logger) -> None:
    """Check that loaded settings contain all required keys and types.

    Raises on missing critical keys. Warns on missing optional keys or
    type mismatches -- allows the app to start with partial config.
    """
    missing = _REQUIRED_SETTINGS_KEYS - settings.keys()
    if missing:
        raise ValueError(
            f'[Settings ] {filepath} missing required keys: {sorted(missing)}. '
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

    Returns the list of mismatches, deepest key path first.
    """
    problems = []
    for key, template_value in template.items():
        if key not in current:
            continue
        value = current[key]
        where = f'{path}.{key}' if path else key
        for kind in (dict, list):
            if isinstance(template_value, kind) and not isinstance(value, kind):
                problems.append(f'{where}: expected {kind.__name__}, got {type(value).__name__}')
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


# Written into a layer whose video_config arrived absent, null, or with a
# rate the recorder cannot use. The shipped template carries the same pair,
# so an untouched install never reaches these.
DEFAULT_VIDEO_DURATION_SEC = 5
DEFAULT_VIDEO_FPS = 30


def normalize_loaded_settings(settings_dict: dict) -> bool:
    """Repair loaded values the running version cannot use as written.

    Distinct from the default merge, which only ADDS absent keys. These
    keys are PRESENT and hold something the app would misread: a retired
    spinner label, a per-layer acquire mode that is neither of the two the
    capture code branches on, a video config written as null. The merge
    cannot see any of them, because nothing is missing.

    Returns:
        True when at least one value was repaired.
    """
    # Deferred to break an import cycle: lvp_logger imports load_debug_setting
    # from this module at its module top (before lvp_logger.logger is defined),
    # and both of these import lvp_logger.logger at their own top.
    from modules import image_mode
    from modules.common_utils import get_layers

    changed = False

    output_format = settings_dict.get('image_output_format')
    if isinstance(output_format, dict) and output_format.get('sequenced') == 'ImageJ Hyperstack':
        # The file written never changed -- it was always OME-TIFF. Only the
        # label did, because it named a reader instead of the format.
        output_format['sequenced'] = image_mode.OUTPUT_FORMAT_HYPERSTACK
        changed = True

    # Protocol accordions are permanently enabled; a stored preference for
    # the retired toggle would be read by nothing.
    if settings_dict.pop('disable_protocol_accordions', None) is not None:
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

        # The capture path branches on exactly 'image' and 'video'; anything
        # else has to mean "do not acquire", or it falls through both.
        if layer_settings.get('acquire') not in ('image', 'video', None):
            layer_settings['acquire'] = None
            changed = True

        video_config = layer_settings.get('video_config')
        if not isinstance(video_config, dict):
            video_config = {}
            layer_settings['video_config'] = video_config
            changed = True
        if 'duration' not in video_config:
            video_config['duration'] = DEFAULT_VIDEO_DURATION_SEC
            changed = True
        # A zero or negative rate would divide into the frame interval.
        if video_config.get('fps', 0) <= 0:
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


def _apply_load_migrations(logger, settings_dict: dict) -> None:
    """Every fold that must run on a loaded dict before the default merge.

    Order matters against the merge, not among themselves: the merge only
    ADDS missing keys, so a rename left unfolded here would get the shipped
    default merged in beside the user's value and the real one would sit
    unread.
    """
    from modules import image_mode

    if image_mode.migrate_settings_dict(settings_dict):
        logger.info('[Settings ] Consolidated capture/save toggles into image_mode')
    if migrate_video_settings_dict(settings_dict):
        logger.info('[Settings ] Renamed manual_video settings section to video')
    if normalize_loaded_settings(settings_dict):
        logger.info('[Settings ] Repaired stored values the running version cannot use')
    forgotten = forget_shipped_focus(settings_dict)
    if forgotten:
        logger.info(
            f'[Settings ] No focus was ever saved for {", ".join(forgotten)}: '
            'their steps take the current Z'
        )


def _load_and_validate(logger, filepath: str) -> dict:
    """Read one settings file and check it carries the keys the app needs."""
    loaded = read_settings_json(filepath, logger)
    _validate_settings(loaded, filepath, logger)
    return loaded


def _load_template(logger, template_path: str) -> dict:
    """The shipped template, checked for the keys the app needs.

    Read as a file the installation ships, not as the user's: a template
    that is missing or will not parse is the installation's fault and
    raises ``InstallationFileError``, never a reason to skip the step that
    needed it.
    """
    template = read_installation_file(template_path)
    _validate_settings(template, template_path, logger)
    return template


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


def bring_up_live_folder(logger: logging.Logger, live_folder: str, directory: str) -> str:
    """The live folder as it is stored: absolute, and created.

    The one rule for the value wherever it enters -- the settings file at
    load, and ``update_settings`` after. The shipped template holds a
    relative folder, which means the installation's; left relative, each
    writer would resolve it against the process's working directory, and an
    installed build's working directory is not writable. A folder that
    cannot be created stays the person's: replacing it would send their
    captures somewhere they will not look and save the replacement over
    their choice. Captures into it are refused by the capture-location
    owner, naming it, until it is reachable.
    """
    folder = pathlib.Path(live_folder)
    if not folder.is_absolute():
        folder = (pathlib.Path(directory) / folder).resolve()
        live_folder = str(folder)
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning(
            f'[Settings ] The live folder {folder} could not be created ({e}); '
            'captures into it are refused until it is reachable.'
        )
    return live_folder


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
    current_path = os.path.join(directory, 'data', 'current.json')
    template_path = os.path.join(directory, 'data', 'settings.json')
    data_dir = os.path.join(directory, 'data')
    rejected = None

    if os.path.exists(current_path):
        # Read once, before the user's file: the shape check, the fallback
        # and the merge all need it, and its refusal is the installation's.
        template = read_installation_file(template_path)
        try:
            prepared = _load_and_validate(logger, current_path)
            _reject_if_misshapen(prepared, template, template_path, current_path)
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
            _validate_settings(template, template_path, logger)
            prepared = copy.deepcopy(template)
            rejected = (current_path, str(e))

        _apply_load_migrations(logger, prepared)

        # Merge missing keys from settings.json defaults into current.json.
        # current.json drifts from settings.json as new features add keys.
        # This ensures new keys are available without losing user values.
        added = _deep_merge_defaults(prepared, template, logger=logger)
        if added:
            logger.info(f'[Settings ] Merged {len(added)} missing keys from settings.json: {added}')
        # Imported here: settings_paths imports this module.
        from modules.settings_paths import replace_refused_stored_values

        replaced = replace_refused_stored_values(prepared, template)
        if replaced is not None:
            _stored_replacements.append(replaced)

        _normalize_turret_slot_keys(prepared)
        prepared['live_folder'] = bring_up_live_folder(logger, prepared['live_folder'], directory)

        return prepared, rejected

    if os.path.exists(template_path):
        prepared = _load_template(logger, template_path)
        _apply_load_migrations(logger, prepared)
        _normalize_turret_slot_keys(prepared)
        prepared['live_folder'] = bring_up_live_folder(logger, prepared['live_folder'], directory)
        return prepared, None

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Couldn't find 'data' directory at {data_dir}")
    raise FileNotFoundError(f'No settings files found in {data_dir}')


def _reject_if_misshapen(loaded, template, template_path, current_path):
    """Refuse a config whose shape the app cannot survive.

    Runs before the migrations and the default merge, which is the only
    position that works: the caller's except routes the rejection to the
    shipped template, and that except covers nothing further down. The
    merge would not repair a mismatch anyway -- it only recurses where both
    sides are already dicts.

    A template that will not parse is NOT allowed to condemn a healthy
    config: the caller reads it first, outside its except, so its
    ``InstallationFileError`` is never reported as "current.json could not
    be used", which would send the user to delete the one file that was
    still good.
    """
    problems = _check_container_shape(loaded, template)
    if problems:
        raise SettingsFileError(
            f'{current_path}: structure does not match {os.path.basename(template_path)} '
            f'-- {"; ".join(problems)}'
        )


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
    it makes the session provisional, so every save raises loudly instead
    of writing template values over the only copy of the user's
    configuration. The file itself is not touched.
    """
    global rejected_current_json

    current_path = os.path.join(lvp_appdata, 'data', 'current.json')
    template_path = os.path.join(lvp_appdata, 'data', 'settings.json')
    prepared = _load_template(logger, template_path)

    logger.error(
        f'[Settings ] {current_path} could not be used ({reason}); '
        'starting from the shipped defaults. The file has NOT been '
        'modified and no settings will be saved until this is resolved.'
    )
    _apply_load_migrations(logger, prepared)
    _normalize_turret_slot_keys(prepared)
    prepared['live_folder'] = bring_up_live_folder(logger, prepared['live_folder'], lvp_appdata)

    settings.clear()
    settings.update(prepared)
    rejected_current_json = (current_path, reason)


def retire_rejected_current_json() -> str | None:
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
    retired = f'{path}.rejected-{stamp}'
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
    configuration with the template, so the writer refuses while it holds.
    """
    return rejected_current_json is not None


def targets_current_json(file: object) -> bool:
    """Is this save aimed at the live user configuration?

    Matched on the resolved basename rather than the literal argument: the
    writer normalises its path afterwards (appends .json, absolutizes
    against the source root), and a caller outside this repo may hand it an
    absolute path to the same file. Comparing the string it was given would
    let those through.

    Lives here rather than beside the writer because which file holds the
    user's configuration is a fact about settings, not about the GUI -- any
    future writer needs the same answer.
    """
    if not isinstance(file, (str, os.PathLike)):
        return False
    name = os.fspath(file)
    if name[-5:].lower() != '.json':
        name += '.json'
    return os.path.basename(name).lower() == 'current.json'


def _resolve_settings_path(directory):
    current_path = os.path.join(directory, 'data', 'current.json')
    settings_path = os.path.join(directory, 'data', 'settings.json')
    data_dir = os.path.join(directory, 'data')

    if os.path.exists(current_path):
        return current_path
    if os.path.exists(settings_path):
        return settings_path
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Couldn't find 'data' directory at {data_dir}")
    raise FileNotFoundError(f'No settings files found in {data_dir}')


def load_debug_setting(directory: str) -> bool:
    global debug_setting, debug_setting_source

    try:
        filename = _resolve_settings_path(directory)

        temp_settings = read_settings_json(filename)

        # Named as the source only after the read succeeds: a rejected
        # file must not appear in the banner as the settings in force.
        debug_setting_source = os.path.basename(filename)

        debug_setting = temp_settings.get('debug_mode', False)
        return debug_setting

    except Exception as e:
        raise e


def load_profile_trace_setting(directory: str) -> dict:
    """Read profile_trace.enabled + profile_trace.output_dir from settings.

    Returns a dict {"enabled": bool, "output_dir": str | None}. Missing
    or unreadable settings file resolves to {"enabled": False,
    "output_dir": None} so the caller never has to guard for absence;
    profile_trace defaults OFF in that case.

    Called from lib/profile_trace.py at module-import time, mirroring
    the timing of load_debug_setting() above. Replaces the prior
    LVP_PROFILE_TRACE environment-variable gate.
    """
    try:
        filename = _resolve_settings_path(directory)
        temp_settings = read_settings_json(filename)
    except Exception:
        return {'enabled': False, 'output_dir': None}

    return {
        'enabled': bool(temp_settings.get('profile_trace_enabled', False)),
        'output_dir': temp_settings.get('profile_trace_output_dir') or None,
    }


def load_tracemalloc_setting(directory: str) -> bool:
    """Read tracemalloc_enabled from settings.

    Returns bool. Missing or unreadable settings file resolves to False
    so the caller never has to guard for absence; tracemalloc defaults
    OFF in that case (10-30% process-memory overhead is the cost).

    Called from modules/common_utils.py at module-import time, mirroring
    the timing of load_profile_trace_setting() above. Replaces the prior
    LVP_TRACEMALLOC environment-variable gate.
    """
    try:
        filename = _resolve_settings_path(directory)
        temp_settings = read_settings_json(filename)
    except Exception:
        return False

    return bool(temp_settings.get('tracemalloc_enabled', False))


def load_memory_profile_setting(directory: str) -> dict:
    """Read memory_profile settings (gate + cadence).

    Returns ``{"enabled": bool, "interval_s": float}``. Missing or unreadable
    settings file resolves to ``{"enabled": False, "interval_s": 5.0}`` so the
    caller never has to guard for absence; the memory profiler defaults OFF
    (tracemalloc carries 10-30% process-memory overhead, the same cost as the
    tracemalloc gate). Enable via ``memory_profile_enabled: true`` in the live
    settings (current.json once it exists, settings.json default) -- the same
    merged-settings path as profile_trace / tracemalloc.

    Called from lib/memory_profile.py, mirroring load_profile_trace_setting /
    load_tracemalloc_setting above.
    """
    try:
        filename = _resolve_settings_path(directory)
        temp_settings = read_settings_json(filename)
    except Exception:
        return {'enabled': False, 'interval_s': 5.0}

    return {
        'enabled': bool(temp_settings.get('memory_profile_enabled', False)),
        'interval_s': float(temp_settings.get('memory_profile_interval_s', 5.0)),
    }
