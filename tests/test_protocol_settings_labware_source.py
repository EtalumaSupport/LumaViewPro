# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression test: ProtocolSettings does not parse labware.json itself.

Bug history
-----------
- Audit finding (2026-05-19): ProtocolSettings.__init__ used to
  open data/labware.json directly and json.load it into
  self.labware, while ScopeSession constructs a WellPlateLoader
  that parses the same file. Two parses, no consistency check.
- Followup DOA (2026-05-20, issue 670): the audit fix replaced the
  local parse with `self.labware = ctx.wellplate_loader.labware`,
  but ctx is None during Kivy KV widget construction (MainDisplay
  is built before app_context.ctx is published). Startup crashed
  with AttributeError. Cluster scan showed nothing reads
  ProtocolSettings.labware -- the assignment was dead. The fix
  deletes the line entirely; labware comes from ctx.wellplate_loader
  at the use sites that actually need it (select_labware via
  get_selected_labware, post-construction).

Invariant
---------
The duplicate-parse invariant from the audit still holds:
ProtocolSettings.__init__ must not call json.load and the module
must not import json. The earlier "must reference
wellplate_loader.labware" assertion over-specified the code shape
and is removed -- the canonical source is still owned by
WellPlateLoader, just not mirrored onto a dead attribute.
"""

from __future__ import annotations

import ast
import pathlib


REPO = pathlib.Path(__file__).resolve().parent.parent
PROTOCOL_SETTINGS_SRC = REPO / 'ui' / 'protocol_settings.py'


def _module_tree() -> ast.Module:
    return ast.parse(PROTOCOL_SETTINGS_SRC.read_text())


def _init_method() -> ast.FunctionDef:
    tree = _module_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'ProtocolSettings':
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == '__init__':
                    return child
    raise AssertionError('ProtocolSettings.__init__ not found')


def test_init_does_not_call_json_load():
    """ProtocolSettings.__init__ no longer parses labware.json itself."""
    init = _init_method()
    for node in ast.walk(init):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == 'load':
                if isinstance(func.value, ast.Name) and func.value.id == 'json':
                    raise AssertionError(
                        'ProtocolSettings.__init__ contains json.load(); '
                        'labware should come from ctx.wellplate_loader.labware'
                    )


def test_module_does_not_import_json():
    """Removing json.load also removes the only use of the json module."""
    tree = _module_tree()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != 'json', (
                    'ui/protocol_settings.py no longer needs the json module; remove the import'
                )
        elif isinstance(node, ast.ImportFrom):
            assert node.module != 'json', 'ui/protocol_settings.py no longer needs the json module'


def test_init_does_not_dereference_ctx_attributes():
    """ProtocolSettings.__init__ runs during KV widget tree construction,
    BEFORE app_context.ctx is published in LumaViewProApp.build(). Any
    `ctx.X` dereference at __init__ time (other than guarded `if ctx is
    not None`) hits NoneType and crashes startup -- the issue-670 DOA.

    Allowed shape: `ctx.X if ctx is not None else <fallback>`. Any other
    `ctx.X` (Attribute on Name `ctx`) at __init__ time is a regression.
    Post-construction methods (select_labware, _init_ui, etc.) can
    deref ctx freely because ctx is populated by then.
    """
    init = _init_method()
    offenders = []
    for node in ast.walk(init):
        if isinstance(node, ast.IfExp):
            # IfExp like `ctx.X if ctx is not None else Y` -- skip the
            # ctx.X load that is guarded by the test.
            continue
        if not isinstance(node, ast.Attribute):
            continue
        if not (isinstance(node.value, ast.Name) and node.value.id == 'ctx'):
            continue
        # Walk up: if this Attribute is inside an IfExp whose test
        # gates on `ctx is not None`, accept it.
        parent_guarded = False
        for ancestor in ast.walk(init):
            if isinstance(ancestor, ast.IfExp) and node in ast.walk(ancestor.body):
                test = ancestor.test
                if (
                    isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name)
                    and test.left.id == 'ctx'
                    and any(isinstance(op, (ast.IsNot, ast.Is)) for op in test.ops)
                ):
                    parent_guarded = True
                    break
        if not parent_guarded:
            offenders.append(f'line {node.lineno}: ctx.{node.attr}')
    assert not offenders, (
        'ProtocolSettings.__init__ dereferences ctx without a `ctx is '
        'not None` guard; ctx is None during KV widget construction. '
        f'Offenders: {offenders}'
    )
