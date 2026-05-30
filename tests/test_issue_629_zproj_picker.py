"""Regression for #629: the Z-Projection folder picker over-corrected.

The picker used to descend into the first existing of
[Manual/Z-Stacks, ProtocolData], so once a manual Z-stack existed it ALWAYS
shadowed protocol-produced z-stacks -- the reporter saw it "always open at the
Manual data folder, even after a Protocol produces a data folder." The picker
now descends only when exactly one source exists; when both exist it opens at
live_folder so neither is hidden.

file_dialogs.py imports kivy at module top (stubbed in the test harness), so we
load the pure helper from source and exec it rather than importing the module --
this exercises the real function body, not a copy.
"""

import ast
import pathlib

_SRC = (pathlib.Path(__file__).resolve().parents[1] / 'ui' / 'file_dialogs.py').read_text()


def _load_helper():
    tree = ast.parse(_SRC)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_zprojection_picker_default_path':
            namespace = {'pathlib': pathlib}
            exec(ast.get_source_segment(_SRC, node), namespace)
            return namespace['_zprojection_picker_default_path']
    raise AssertionError('_zprojection_picker_default_path not found in file_dialogs.py')


_picker_path = _load_helper()


def test_only_manual_descends_into_manual(tmp_path):
    (tmp_path / 'Manual' / 'Z-Stacks').mkdir(parents=True)
    assert _picker_path(tmp_path) == str(tmp_path / 'Manual' / 'Z-Stacks')


def test_only_protocol_descends_into_protocol(tmp_path):
    (tmp_path / 'ProtocolData').mkdir(parents=True)
    assert _picker_path(tmp_path) == str(tmp_path / 'ProtocolData')


def test_both_present_opens_live_folder(tmp_path):
    # The #629 case: a manual z-stack must NOT shadow protocol z-stacks.
    (tmp_path / 'Manual' / 'Z-Stacks').mkdir(parents=True)
    (tmp_path / 'ProtocolData').mkdir(parents=True)
    assert _picker_path(tmp_path) == str(tmp_path)


def test_neither_present_opens_live_folder(tmp_path):
    assert _picker_path(tmp_path) == str(tmp_path)
