# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""The per-layer display surface.

Accordion titles come from the layer identity record -- display name
plus an excitation suffix whose 'Ex' marker disambiguates excitation
from the emission colour in the layer's name ('Green Ex 488 nm', never
'Green 488 nm'), with the suffix absent when the layer has no
excitation (broadband and LED-less layers). Layer visibility is wired
per layer from the record, not per category, so a filterset carrying
only some fluorescence channels shows exactly what the unit has.

The ui modules are unimportable under the conftest kivy mocks, so the
title formatter is carved out of the parsed module via ast_seams and
exec'd standalone (the real body runs, not a copy), and the wiring is
asserted on the AST rather than source text.
"""

import ast
from types import SimpleNamespace

from tests.ast_seams import assert_def, find_def, parse_module


def _load_layer_title():
    node = find_def('ui/image_settings.py', 'layer_title')
    assert node is not None, 'ui/image_settings.py: def layer_title(...) not found'
    namespace = {}
    exec(ast.unparse(node), namespace)
    return namespace['layer_title']


def _record(display_name, excitation_nm):
    return SimpleNamespace(display_name=display_name, excitation_nm=excitation_nm)


class TestLayerTitleFormat:
    def test_excitation_suffix_carries_the_ex_marker(self):
        layer_title = _load_layer_title()
        assert layer_title(_record('Green', 488.0)) == 'Green Ex 488 nm'

    def test_null_excitation_is_hidden_not_rendered(self):
        layer_title = _load_layer_title()
        assert layer_title(_record('BF-Phase', None)) == 'BF-Phase'

    def test_integral_float_renders_without_decimal_point(self):
        layer_title = _load_layer_title()
        assert layer_title(_record('Blue', 405.0)) == 'Blue Ex 405 nm'

    def test_fractional_excitation_survives(self):
        layer_title = _load_layer_title()
        assert layer_title(_record('Red', 589.5)) == 'Red Ex 589.5 nm'


class TestPerLayerWiring:
    def test_per_layer_setter_and_title_seams_exist(self):
        assert_def(
            'ui/image_settings.py',
            'set_fluorescence_layer_control_visibility',
            class_name='ImageSettings',
            params=['self', 'layer', 'visible'],
        )
        assert_def(
            'ui/image_settings.py',
            'apply_layer_titles',
            class_name='ImageSettings',
            params=['self', 'layers'],
        )

    def test_scope_apply_drives_per_layer_fluorescence_and_titles(self):
        node = find_def('ui/microscope_settings.py', 'set_ui_features_for_scope')
        assert node is not None
        called = {
            sub.func.attr
            for sub in ast.walk(node)
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
        }
        assert 'set_fluorescence_layer_control_visibility' in called
        assert 'apply_layer_titles' in called

    def test_category_granular_fluorescence_setter_is_gone(self):
        # The old trio setter showed/hid Blue+Green+Red as one block,
        # which cannot express a Green-only unit; nothing may call or
        # reintroduce it.
        for rel_path in ('ui/image_settings.py', 'ui/microscope_settings.py'):
            assert find_def(rel_path, 'set_fluoresence_layer_controls_visibility') is None
            names = {
                sub.attr
                for sub in ast.walk(parse_module(rel_path))
                if isinstance(sub, ast.Attribute)
            }
            assert 'set_fluoresence_layer_controls_visibility' not in names

    def test_no_second_display_order_is_authored(self):
        # Display order derives from the release layer catalogue; a
        # hardcoded layer-order tuple here would be a mirror needing
        # manual sync with the catalogue.
        tree = parse_module('ui/image_settings.py')
        assigned = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        assert '_LAYER_DISPLAY_ORDER' not in assigned
