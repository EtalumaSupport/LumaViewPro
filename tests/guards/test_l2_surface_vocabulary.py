# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Structural guard: the retired L2 spellings stay retired in production.

The L2 surface converged on one word per fact:

    LED drive current   mA        -> illumination_ma
    turret move         tmove     -> move_turret
    turret homed query  has_thomed -> position_is_known('T')
    camera reads        *_cached, get_gain_db, get_exposure_ms
                                  -> the held property or read_*
    camera writes       set_auto_exposure_time, apply_layer_camera_settings
                                  -> set_auto_exposure, apply

Drivers deliberately keep the old vocabulary -- `mA` mirrors the board
command format and `thome` / `has_thomed` are the motor board's own
names -- so `drivers/` is NOT swept. The translation happens at the
facade, whose call into the driver is positional. The camera's
get_binning_size, get_black_level, get_resulting_frame_rate and
set_binning_size are the drivers' own names, which the API calls, so
they are not retired here.

This is an AST sweep, not a grep, and the distinction is the whole
point. `mA` is also a legitimate UNIT: it is the value of the
`IlluminationUnit` tag written into every saved TIFF, it appears in the
regex that parses `I_SENS` firmware output, and it labels GUI sliders.
A text search cannot tell those from an identifier; the parser can, so
prose and string literals are invisible here by construction.

Precedent for the shape: the Wave-7 attribute rename claimed 259 sites
and still shipped live misses, each of which produced a user-visible
failure on real hardware. A completed rename needs a guard that fails
if the old name comes back, not a one-time sweep.
"""

import ast

from tests.ast_seams import Packages, iter_package_modules, parse_module

PROD_PACKAGES = Packages({'modules', 'ui'})

RETIRED = {
    'mA': 'illumination_ma',
    'tmove': 'move_turret',
    '_tmove_impl': '_move_turret_impl',
    'has_thomed': "position_is_known('T')",
    '_thome_impl': '_home_turret_impl',
    'active_cached': 'camera.active',
    'gain_db_cached': 'camera.gain_db',
    'exposure_ms_cached': 'camera.exposure_ms',
    'frame_size_cached': 'camera.frame_size',
    'pixel_format_cached': 'camera.pixel_format',
    'min_frame_size_cached': 'capabilities.camera_limits.frame_min',
    'min_gain_db_cached': 'capabilities.camera_limits.gain_db_min',
    'max_gain_db_cached': 'capabilities.camera_limits.gain_db_max',
    'min_exposure_ms_cached': 'capabilities.camera_limits.exposure_ms_min',
    'max_exposure_ms_cached': 'capabilities.camera_limits.exposure_ms_max',
    'get_gain_db': 'camera.read_gain_db',
    'get_exposure_ms': 'camera.read_exposure_ms',
    'get_live_camera_settings': 'camera.read_gain_db and its siblings',
    'set_auto_exposure_time': 'camera.set_auto_exposure',
    'apply_layer_camera_settings': 'camera.apply',
    'set_layer_acquire': 'set_channel_acquire',
    'layer_acquire': 'channel_acquire',
    'set_layer_auto_gain': 'set_channel_auto_gain',
    'layer_auto_gain': 'channel_auto_gain',
    'layer_auto_gain_lock': 'channel_auto_gain_lock',
    'save_layer_focus': 'save_channel_focus',
    'apply_focus_to_layer_steps': 'apply_focus_to_channel_steps',
    'apply_layer_settings': 'apply_channel_settings',
    'LayerIdentity': 'ChannelIdentity',
    'LayerRecord': 'ChannelRecord',
    'layer_identity': 'channel_identity',
    'WellPlate': 'Labware',
    'is_current_turret_position_objective_set': 'is_current_slot_objective_set',
    'saved_focus': 'saved_focus_z_um',
    'focus_z': 'focus_z_um',
    'jog_step': 'jog_step_um',
}


def _retired_identifier_hits(tree: ast.AST) -> list[tuple[int, str]]:
    """Every use of a retired name AS AN IDENTIFIER, with its line."""
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.arg in RETIRED:
            hits.append((node.lineno, node.arg))
        elif isinstance(node, ast.keyword) and node.arg in RETIRED:
            hits.append((node.value.lineno, node.arg))
        elif isinstance(node, ast.Name) and node.id in RETIRED:
            hits.append((node.lineno, node.id))
        elif isinstance(node, ast.Attribute) and node.attr in RETIRED:
            hits.append((node.lineno, node.attr))
        elif (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name in RETIRED
        ):
            hits.append((node.lineno, node.name))
        elif isinstance(node, ast.alias):
            for name in (node.name.rpartition('.')[2], node.asname):
                if name in RETIRED:
                    hits.append((node.lineno, name))
    return hits


def test_a_retired_type_is_caught_where_it_is_defined_or_imported():
    """A retired type name revived as a class, or imported under its old name, is a hit."""
    tree = ast.parse(
        'class LayerIdentity:\n'
        '    pass\n'
        'from modules.layer_record import LayerRecord\n'
        'from modules.layer_record import ChannelRecord as LayerRecord\n'
    )
    assert _retired_identifier_hits(tree) == [
        (1, 'LayerIdentity'),
        (3, 'LayerRecord'),
        (4, 'LayerRecord'),
    ]


def test_no_retired_l2_spelling_in_production():
    hits = [
        (rel_path, line, name)
        for rel_path, tree in iter_package_modules(PROD_PACKAGES)
        for line, name in _retired_identifier_hits(tree)
    ]
    assert not hits, '\n'.join(
        f'{rel_path}:{line} uses retired `{name}` -- the L2 surface spells this '
        f'`{RETIRED[name]}`. If this is driver vocabulary it belongs in drivers/, '
        f'not here.'
        for rel_path, line, name in hits
    )


def test_the_unit_string_is_untouched():
    """`mA` as a UNIT must survive -- it is written into every saved TIFF.

    This pins the distinction the guard above rests on: banning the
    identifier must never cost the unit. If a future sweep renames this,
    every image this application writes carries a wrong unit.

    Parsed, not read as text: a source-text read would add this file to
    the read-text ratchet for a claim the AST answers exactly.
    """
    tree = parse_module('modules/image_utils.py')
    # The tag is written either as a dict-literal pair or as a subscript
    # assignment (it became conditional when Illumination joined the
    # optional-fields contract); the unit fact is the same in both forms.
    units = [
        value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values, strict=True)
        if isinstance(key, ast.Constant)
        and key.value == 'IlluminationUnit'
        and isinstance(value, ast.Constant)
    ]
    units += [
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Subscript)
        and isinstance(node.targets[0].slice, ast.Constant)
        and node.targets[0].slice.value == 'IlluminationUnit'
        and isinstance(node.value, ast.Constant)
    ]
    assert units == ['mA'], (
        f'the TIFF IlluminationUnit tag must still read mA, got {units!r} -- '
        'a saved image records the unit, not the parameter name'
    )
