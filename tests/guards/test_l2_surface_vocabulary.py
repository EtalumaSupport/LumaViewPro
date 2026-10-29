# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Structural guard: the retired L2 spellings stay retired in production.

The L2 surface converged on one word per fact:

    LED drive current   mA        -> illumination_ma
    turret move         tmove     -> move_turret
    turret homed query  has_thomed -> axes_without_position()
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

from tests.ast_seams import Packages, iter_package_modules

PROD_PACKAGES = Packages({'modules', 'ui'})

RETIRED = {
    'mA': 'illumination_ma',
    'tmove': 'move_turret',
    '_tmove_impl': '_move_turret_impl',
    'has_thomed': 'axes_without_position()',
    '_thome_impl': '_home_turret_impl',
    'active_cached': 'camera.active',
    'gain_db_cached': 'camera.gain_db',
    'exposure_ms_cached': 'camera.exposure_ms',
    'frame_size_cached': 'camera.frame_size',
    'pixel_format_cached': 'camera.pixel_format',
    'min_frame_size_cached': 'capabilities.camera_limits.frame_min',
    'min_gain_db_cached': 'capabilities.camera_limits.gain_min_db',
    'max_gain_db_cached': 'capabilities.camera_limits.gain_max_db',
    'min_exposure_ms_cached': 'capabilities.camera_limits.exposure_min_ms',
    'max_exposure_ms_cached': 'capabilities.camera_limits.exposure_max_ms',
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
    'is_current_turret_position_objective_set': 'turret_status().slot and get_turret_config',
    'is_current_slot_objective_set': 'turret_status().slot and get_turret_config',
    'get_turret_slot': 'turret_status().slot',
    'is_moving': 'is_any_axis_moving',
    'saved_focus': 'saved_focus_z_um',
    'focus_z': 'focus_z_um',
    'jog_step': 'jog_step_um',
    'CameraChanged': 'GainChanged and ExposureChanged',
    'add_camera_listener': 'add_gain_listener and add_exposure_listener',
    'remove_camera_listener': 'remove_gain_listener and remove_exposure_listener',
    'AppliedCameraSetting': 'AppliedGain, AppliedExposure and AppliedAutoGain',
    'read_frame_rate': 'camera.read_frame_rate_fps',
    'read_exposure_ms_min': 'camera.read_exposure_min_ms',
    'gain_db_min': 'capabilities.camera_limits.gain_min_db',
    'gain_db_max': 'capabilities.camera_limits.gain_max_db',
    'analog_gain_db_max': 'capabilities.camera_limits.analog_gain_max_db',
    'exposure_ms_min': 'capabilities.camera_limits.exposure_min_ms',
    'exposure_ms_max': 'capabilities.camera_limits.exposure_max_ms',
    'inter_cycle_delay_ms': 'inter_cycle_delay_s, in seconds',
    'frames_per_sec': 'frame_rate_fps',
    'frames_per_s': 'frame_rate_fps',
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
    every image this application writes carries a wrong unit. Read from what
    the writer writes, whatever form its source takes.
    """
    import numpy as np

    from modules import image_utils
    from modules.layer_record import GREY

    plane = image_utils.generate_tiff_data(
        data=np.zeros((4, 4), dtype=np.uint8),
        metadata={
            'pixel_size_um': None,
            'channel': 'BF',
            'significant_bits': 8,
            'illumination_ma': 50.0,
            'illumination_by_channel_ma': {'BF': 50.0},
        },
        image_type=None,
        color=GREY,
        white_bits=None,
    )['metadata']['Plane']
    units = [plane['IlluminationUnit'], plane['IlluminationByChannelUnit']]
    assert units == ['mA', 'mA'], (
        f'the TIFF illumination unit tags must still read mA, got {units!r} -- '
        'a saved image records the unit, not the parameter name'
    )
