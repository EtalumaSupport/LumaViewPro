# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
import os
import json
import time


settings = None

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


def read_settings_json(path, logger=None):
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
    # not in load_settings: the GUI bootstrap, ScopeSession.create_headless
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
        'focus': (int, float),
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


def _load_and_validate(logger, filepath: str) -> dict:
    """Read one settings file and check it carries the keys the app needs."""
    loaded = read_settings_json(filepath, logger)
    _validate_settings(loaded, filepath, logger)
    return loaded


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


def prepare_settings(logger, directory, *, fall_back_to_template: bool) -> tuple:
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
        try:
            prepared = _load_and_validate(logger, current_path)
            _reject_if_misshapen(logger, prepared, template_path, current_path)
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
            if not os.path.exists(template_path):
                raise FileNotFoundError(
                    f'current.json corrupt and no settings.json fallback in {data_dir}'
                ) from e
            prepared = _load_and_validate(logger, template_path)
            rejected = (current_path, str(e))

        _apply_load_migrations(logger, prepared)

        # Merge missing keys from settings.json defaults into current.json.
        # current.json drifts from settings.json as new features add keys.
        # This ensures new keys are available without losing user values.
        if os.path.exists(template_path):
            try:
                defaults = read_settings_json(template_path, logger)
                added = _deep_merge_defaults(prepared, defaults, logger=logger)
                if added:
                    logger.info(
                        f'[Settings ] Merged {len(added)} missing keys from settings.json: {added}'
                    )
            except Exception:
                logger.warning('[Settings ] Could not load settings.json for default merge')

        _normalize_turret_slot_keys(prepared)

        return prepared, rejected

    if os.path.exists(template_path):
        prepared = _load_and_validate(logger, template_path)
        _apply_load_migrations(logger, prepared)
        _normalize_turret_slot_keys(prepared)
        return prepared, None

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Couldn't find 'data' directory at {data_dir}")
    raise FileNotFoundError(f'No settings files found in {data_dir}')


def _reject_if_misshapen(logger, loaded, template_path, current_path):
    """Refuse a config whose shape the app cannot survive.

    Runs before the migrations and the default merge, which is the only
    position that works: the caller's except routes the rejection to the
    shipped template, and that except covers nothing further down. The
    merge would not repair a mismatch anyway -- it only recurses where both
    sides are already dicts.

    A template that will not parse is NOT allowed to condemn a healthy
    config. Without that, a settings.json truncated by a bad upgrade would
    raise here, be reported as "current.json could not be used", and send
    the user to delete the one file that was still good.
    """
    try:
        template = read_settings_json(template_path, logger)
    except (FileNotFoundError, SettingsFileError) as e:
        logger.warning(
            f'[Settings ] {template_path} unreadable ({e}); skipping the shape '
            f'check on {current_path}'
        )
        return

    problems = _check_container_shape(loaded, template)
    if problems:
        raise SettingsFileError(
            f'{current_path}: structure does not match {os.path.basename(template_path)} '
            f'-- {"; ".join(problems)}'
        )


def load_lvp_settings(logger, lvp_appdata):
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


def retire_rejected_current_json():
    """Move the unusable current.json aside so a fresh one can take its place.

    Renamed, never deleted: it is the user's only copy of their
    configuration, and support can often read what they had out of it even
    when the app could not. Returns the new path.

    Called only after a human has chosen to start over -- the rename is the
    point of no return for that file's role, and nothing should reach it by
    timeout, by a dismissed dialog, or by any other default.
    """
    global rejected_current_json
    if rejected_current_json is None:
        return None
    path, _reason = rejected_current_json
    stamp = time.strftime('%Y%m%d-%H%M%S')
    retired = f'{path}.rejected-{stamp}'
    os.replace(path, retired)
    rejected_current_json = None
    return retired


def settings_are_provisional():
    """True while the app is running on defaults nobody has agreed to keep.

    Writing current.json in this state would replace a user's whole
    configuration with the template, so the writer refuses while it holds.
    """
    return rejected_current_json is not None


def targets_current_json(file):
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


def load_profile_trace_setting(directory):
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


def load_tracemalloc_setting(directory):
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


def load_memory_profile_setting(directory):
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


def load_fx2_debug_wire_setting(directory):
    """Read fx2_debug_wire_enabled from settings.

    Returns bool. Missing or unreadable settings file resolves to False
    so the caller never has to guard for absence; the FX2 wire-protocol
    debug trace defaults OFF (it is an L4 diagnostic surface).

    Called from drivers/fx2driver.py, ui/layer_control.py, and
    modules/lumascope_api/illumination.py at module-import time.
    Replaces the prior LVP_FX2_DEBUG_WIRE environment-variable gate.
    """
    try:
        filename = _resolve_settings_path(directory)
        temp_settings = read_settings_json(filename)
    except Exception:
        return False

    return bool(temp_settings.get('fx2_debug_wire_enabled', False))
