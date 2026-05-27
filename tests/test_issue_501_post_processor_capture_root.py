# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""#501 regression: composite / stitched / z-proj / video / stack outputs
must preserve the protocol's capture_root in their filenames.

Bug
---
Per-image protocol saves use protocol.capture_root() as a filename
prefix (protocol_image_writer.py:266-273). The five post-processors
(composite_generation, stitcher, zprojector, video_builder,
stack_builder) all read a different field: a per-step DataFrame
column called "Custom Root" that is never written anywhere in the
codebase. Every read defaults to empty, so composite + stitched +
z-proj + video output filenames collapse to step-name only, dropping
the experiment's Root prefix. stack_builder didn't even reference
Custom Root -- it just used row0['Name'] directly with no prefix.

Fix
---
ProtocolPostProcessor.load_folder now extracts protocol.capture_root()
from the loaded Protocol and threads it into kwargs as 'capture_root'.
Each subclass's _generate_filename reads kwargs['capture_root'] and
uses it as the filename prefix instead of the dormant Custom Root
column.

Test approach
-------------
1. Base load_folder must extract protocol + populate kwargs with
   'capture_root' before _generate_filename is called.
2. Each of the five subclasses must:
   - Read 'capture_root' from kwargs in _generate_filename
   - NOT read the dormant 'Custom Root' DataFrame column
   - Use the read value as the prefix when non-empty

AST-based structural locks (UI dependencies prevent direct exec).
"""

from __future__ import annotations

import ast
import pathlib


REPO = pathlib.Path(__file__).resolve().parent.parent

POST_PROCESSORS = {
    'CompositeGeneration': REPO / 'modules' / 'composite_generation.py',
    'Stitcher': REPO / 'modules' / 'stitcher.py',
    'ZProjector': REPO / 'modules' / 'zprojector.py',
    'VideoBuilder': REPO / 'modules' / 'video_builder.py',
    'StackBuilder': REPO / 'modules' / 'stack_builder.py',
}

BASE_PATH = REPO / 'modules' / 'protocol_post_processor.py'


def _method_node(path: pathlib.Path, class_name: str, method_name: str) -> ast.FunctionDef:
    source = path.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f'{class_name}.{method_name} not found in {path}')


def test_base_load_folder_threads_capture_root_into_kwargs():
    """ProtocolPostProcessor.load_folder must read protocol.capture_root()
    and put it into kwargs before _generate_filename runs."""
    method = _method_node(BASE_PATH, 'ProtocolPostProcessor', 'load_folder')
    src = ast.unparse(method)
    assert 'capture_root' in src, (
        'ProtocolPostProcessor.load_folder must thread capture_root '
        'into kwargs. (#501)'
    )
    assert '.capture_root()' in src, (
        'ProtocolPostProcessor.load_folder must call '
        'protocol.capture_root() to source the prefix. (#501)'
    )
    # Sanity: kwargs is populated before _generate_filename is called.
    # Find statement indices for the assignment and the call.
    assign_idx = -1
    call_idx = -1
    for i, stmt in enumerate(method.body):
        unparsed = ast.unparse(stmt)
        if assign_idx == -1 and 'capture_root' in unparsed and (
            'kwargs' in unparsed or 'setdefault' in unparsed
        ):
            assign_idx = i
        if 'self._generate_filename' in unparsed:
            call_idx = i
            break
    assert assign_idx >= 0, (
        'capture_root must be assigned into kwargs in load_folder. (#501)'
    )
    assert call_idx >= 0, (
        '_generate_filename call not found in load_folder; test needs '
        'updating for the new shape.'
    )
    assert assign_idx < call_idx, (
        f'capture_root must be threaded into kwargs at statement '
        f'{assign_idx} BEFORE the _generate_filename call site (which '
        f'leads to the per-group loop at statement {call_idx}). (#501)'
    )


def _assert_subclass_uses_capture_root_kwarg(class_name: str, path: pathlib.Path):
    method = _method_node(path, class_name, '_generate_filename')
    src = ast.unparse(method)
    assert 'capture_root' in src, (
        f'{class_name}._generate_filename must read capture_root from '
        f'kwargs to prefix the output filename. (#501)'
    )
    assert "kwargs.get('capture_root'" in src or 'kwargs["capture_root"' in src, (
        f'{class_name}._generate_filename must read capture_root from '
        f'kwargs (not from a DataFrame column). (#501)'
    )
    assert "'Custom Root'" not in src and '"Custom Root"' not in src, (
        f'{class_name}._generate_filename must NOT read the dormant '
        f'"Custom Root" DataFrame column -- that field is never '
        f'written; use kwargs["capture_root"] from load_folder. (#501)'
    )


def test_composite_generation_uses_capture_root():
    _assert_subclass_uses_capture_root_kwarg(
        'CompositeGeneration', POST_PROCESSORS['CompositeGeneration']
    )


def test_stitcher_uses_capture_root():
    _assert_subclass_uses_capture_root_kwarg(
        'Stitcher', POST_PROCESSORS['Stitcher']
    )


def test_zprojector_uses_capture_root():
    _assert_subclass_uses_capture_root_kwarg(
        'ZProjector', POST_PROCESSORS['ZProjector']
    )


def test_video_builder_uses_capture_root():
    _assert_subclass_uses_capture_root_kwarg(
        'VideoBuilder', POST_PROCESSORS['VideoBuilder']
    )


def test_stack_builder_uses_capture_root():
    _assert_subclass_uses_capture_root_kwarg(
        'StackBuilder', POST_PROCESSORS['StackBuilder']
    )
