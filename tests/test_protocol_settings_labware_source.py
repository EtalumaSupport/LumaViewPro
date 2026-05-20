# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression test: ProtocolSettings reads labware via ctx.wellplate_loader.

Bug
---
ProtocolSettings.__init__ used to open data/labware.json directly and
json.load it into self.labware, while ScopeSession constructs a
WellPlateLoader that also parses the same file into
ctx.wellplate_loader.labware. The two parses had no consistency check;
a future labware-validation rule added to WellPlateLoader would not
apply to the UI-loaded copy (Rule-35 semantic-duplicate audit
2026-05-19, finding 6).

Fix
---
ProtocolSettings.__init__ now reads self.labware from
ctx.wellplate_loader.labware (the canonical source) and no longer
imports json or opens labware.json directly.

Test approach
-------------
AST source scan -- ProtocolSettings.__init__ must not contain a
json.load() call, must reference wellplate_loader.labware, and the
module must not import json. Behavioral exec is impractical because
ProtocolSettings construction requires Kivy widgets, ctx, and a
populated WellPlateLoader; the structural test catches re-introduction
of the duplicate parse without that mocking overhead.
"""

from __future__ import annotations

import ast
import pathlib


REPO = pathlib.Path(__file__).resolve().parent.parent
PROTOCOL_SETTINGS_SRC = REPO / "ui" / "protocol_settings.py"


def _module_tree() -> ast.Module:
    return ast.parse(PROTOCOL_SETTINGS_SRC.read_text())


def _init_method() -> ast.FunctionDef:
    tree = _module_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ProtocolSettings":
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "__init__":
                    return child
    raise AssertionError("ProtocolSettings.__init__ not found")


def test_init_does_not_call_json_load():
    """ProtocolSettings.__init__ no longer parses labware.json itself."""
    init = _init_method()
    for node in ast.walk(init):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "load":
                if isinstance(func.value, ast.Name) and func.value.id == "json":
                    raise AssertionError(
                        "ProtocolSettings.__init__ contains json.load(); "
                        "labware should come from ctx.wellplate_loader.labware"
                    )


def test_init_references_wellplate_loader_labware():
    """The labware dict comes from ctx.wellplate_loader.labware."""
    init = _init_method()
    source = ast.unparse(init)
    assert "wellplate_loader.labware" in source, (
        "ProtocolSettings.__init__ no longer routes labware through "
        "ctx.wellplate_loader.labware (canonical source)"
    )


def test_module_does_not_import_json():
    """Removing json.load also removes the only use of the json module."""
    tree = _module_tree()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "json", (
                    "ui/protocol_settings.py no longer needs the json module; "
                    "remove the import"
                )
        elif isinstance(node, ast.ImportFrom):
            assert node.module != "json", (
                "ui/protocol_settings.py no longer needs the json module"
            )
