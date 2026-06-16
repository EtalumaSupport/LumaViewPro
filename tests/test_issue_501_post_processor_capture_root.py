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

import modules.common_utils as common_utils
from modules.composite_generation import _strip_channel_token


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
    # pin-justified: reaching the kwarg-threading behaviorally needs
    # load_folder driven all the way to _generate_filename with a real
    # protocol tree; the AST order-check pins the invariant directly.
    method = _method_node(BASE_PATH, 'ProtocolPostProcessor', 'load_folder')
    src = ast.unparse(method)
    assert 'capture_root' in src, (
        'ProtocolPostProcessor.load_folder must thread capture_root into kwargs. (#501)'
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
        if (
            assign_idx == -1
            and 'capture_root' in unparsed
            and ('kwargs' in unparsed or 'setdefault' in unparsed)
        ):
            assign_idx = i
        if 'self._generate_filename' in unparsed:
            call_idx = i
            break
    assert assign_idx >= 0, 'capture_root must be assigned into kwargs in load_folder. (#501)'
    assert call_idx >= 0, (
        '_generate_filename call not found in load_folder; test needs updating for the new shape.'
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
    _assert_subclass_uses_capture_root_kwarg('Stitcher', POST_PROCESSORS['Stitcher'])


def test_zprojector_uses_capture_root():
    _assert_subclass_uses_capture_root_kwarg('ZProjector', POST_PROCESSORS['ZProjector'])


def test_video_builder_uses_capture_root():
    _assert_subclass_uses_capture_root_kwarg('VideoBuilder', POST_PROCESSORS['VideoBuilder'])


def test_stack_builder_uses_capture_root():
    _assert_subclass_uses_capture_root_kwarg('StackBuilder', POST_PROCESSORS['StackBuilder'])


# ---------------------------------------------------------------------------
# #501 follow-up: a composite must not be named after one arbitrary channel.
# The composite group spans every channel, but the filename took the first
# row's step Name (e.g. 'A1_Green'), leaking that channel into the output.
# ---------------------------------------------------------------------------


def test_strip_channel_token_removes_channel():
    assert _strip_channel_token('A1_Green', 'Green') == 'A1'
    assert _strip_channel_token('A1_Green_2xOly_Z0_0000', 'Green') == 'A1_2xOly_Z0_0000'
    assert _strip_channel_token('A1_BF', 'BF') == 'A1'


def test_strip_channel_token_row_order_stable():
    # Whichever channel happens to be the group's first row, the composite
    # base name is identical -- so the output filename is row-order stable.
    bases = {_strip_channel_token(f'A1_{c}_2xOly', c) for c in ('Green', 'Red', 'Blue', 'BF')}
    assert bases == {'A1_2xOly'}


def test_strip_channel_token_noop_when_absent_or_empty():
    assert _strip_channel_token('A1_2xOly', '') == 'A1_2xOly'
    assert _strip_channel_token('A1_2xOly', 'Green') == 'A1_2xOly'


def test_composite_generation_strips_channel_token():
    """_generate_filename must drop the per-channel token from the step name
    before building the composite filename."""
    method = _method_node(
        POST_PROCESSORS['CompositeGeneration'], 'CompositeGeneration', '_generate_filename'
    )
    src = ast.unparse(method)
    assert '_strip_channel_token' in src, (
        'CompositeGeneration._generate_filename must strip the channel token '
        'from the step name so the composite is not named after one channel.'
    )


# ---------------------------------------------------------------------------
# #501 follow-up 2: a stitched output must not leak the per-tile token (a
# stitch spans all tiles), and a composite-stitch must not leak the channel
# token either (it spans all channels; its stored Color is 'Composite', so
# the leaked channel cannot be matched by Color). A hyperstack collapses all
# channels, so it must not leak the channel token. Custom name text is kept.
# ---------------------------------------------------------------------------


def test_strip_tile_token_removes_tile():
    assert common_utils.strip_tile_token('A1_BF_TA1', 'A1') == 'A1_BF'
    assert common_utils.strip_tile_token('A1_Green_TB2_4xOly_0000', 'B2') == 'A1_Green_4xOly_0000'


def test_strip_tile_token_segment_safe_and_noop():
    # The tile value coincides with the well label; only the 'T<tile>' segment
    # is removed, never the bare well token.
    assert common_utils.strip_tile_token('A1_BF', 'A1') == 'A1_BF'
    assert common_utils.strip_tile_token('myExperiment', 'A1') == 'myExperiment'
    assert common_utils.strip_tile_token('A1_BF_TA1', '') == 'A1_BF_TA1'


def test_strip_any_channel_token_removes_first_layer():
    assert common_utils.strip_any_channel_token('A1_BF_TA1') == 'A1_TA1'
    assert common_utils.strip_any_channel_token('A1_Green') == 'A1'
    # No layer token present -> unchanged (custom name preserved).
    assert common_utils.strip_any_channel_token('A1_4xOly') == 'A1_4xOly'
    assert common_utils.strip_any_channel_token('myExperiment') == 'myExperiment'


def test_stitch_filename_outcomes_match_agreed_rule():
    # Single-channel stitch: drop the tile token, KEEP the channel.
    base = common_utils.strip_tile_token('A1_BF_TA1', 'A1')
    plain = common_utils.generate_default_step_name(
        custom_name_prefix=base,
        well_label='A1',
        color='BF',
        objective_short_name='4xOly',
        scan_count=0,
        tile_label=None,
        stitched=True,
    )
    assert plain == 'A1_BF_4xOly_0000_stitched'

    # Composite-stitch: drop the tile token AND the channel token.
    base = common_utils.strip_tile_token('A1_BF_TA1', 'A1')
    base = common_utils.strip_any_channel_token(base)
    composite = common_utils.generate_default_step_name(
        custom_name_prefix=base,
        well_label='A1',
        color='Composite',
        objective_short_name='4xOly',
        scan_count=0,
        tile_label=None,
        stitched=True,
    )
    assert composite == 'A1_Composite_4xOly_0000_stitched'
    assert '_TA1' not in composite and '_BF' not in composite


def test_stack_filename_drops_channel_keeps_tile():
    base = common_utils.strip_any_channel_token('A1_BF_TA1')
    name = common_utils.generate_default_step_name(
        custom_name_prefix=base,
        well_label='A1',
        color=None,
        objective_short_name='4xOly',
        hyperstack=True,
    )
    assert name == 'A1_TA1_4xOly_hyperstack'
    assert '_BF' not in name


def test_stitcher_drops_tile_and_composite_channel():
    method = _method_node(POST_PROCESSORS['Stitcher'], 'Stitcher', '_generate_filename')
    src = ast.unparse(method)
    assert 'strip_tile_token' in src, (
        'Stitcher._generate_filename must drop the per-tile token -- a stitch '
        'spans all tiles. (#501)'
    )
    assert 'strip_any_channel_token' in src, (
        'Stitcher._generate_filename must drop the channel token for a '
        'composite-stitch (Color == "Composite"). (#501)'
    )


def test_stack_builder_drops_channel_token():
    method = _method_node(POST_PROCESSORS['StackBuilder'], 'StackBuilder', '_generate_filename')
    src = ast.unparse(method)
    assert 'strip_any_channel_token' in src, (
        'StackBuilder._generate_filename must drop the channel token -- a '
        'hyperstack collapses all channels. (#501)'
    )
