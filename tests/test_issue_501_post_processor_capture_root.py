# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""#501 regression: post-processor output filenames.

Two guarantees, both faces of the same bug:

1. CAPTURE_ROOT -- composite / stitch / z-proj / video / stack outputs must
   carry the protocol's capture_root prefix (the per-image saves do). The
   prefix is threaded into kwargs by ProtocolPostProcessor.load_folder (the
   only caller of _generate_filename); each subclass reads it from kwargs,
   never from the dormant, never-written "Custom Root" DataFrame column.

2. IDENTITY BY CONSTRUCTION -- the five post-processors build their output name
   from the canonical builder (build_step_name + step_components), reading the
   authoritative columns. A stitch omits the per-tile token by setting
   tile=None (not by stripping it back out of a name -- the old strip helper
   keyed on an empty Tile column and no-op'd, so the token survived). A
   composite/hyperstack collapses the channel by forcing channel='Composite' /
   channel=None. The three string-stripping helpers (strip_tile_token,
   strip_any_channel_token, composite_generation._strip_channel_token) are
   DELETED; the migration guard below pins that they stay deleted so a future
   merge cannot reintroduce the strip-based path.

The filename-outcome assertions exercise the same component path the
post-processors use; the structural properties (idempotence under channel
change, round-trip) live in test_step_name_builder.py.
"""

from __future__ import annotations

import ast
import pathlib

from modules.common_utils import build_step_name, step_components


REPO = pathlib.Path(__file__).resolve().parent.parent

_LAYERS = ['Blue', 'Green', 'Red', 'BF', 'PC', 'DF', 'Lumi']

POST_PROCESSORS = {
    'CompositeGeneration': REPO / 'modules' / 'composite_generation.py',
    'Stitcher': REPO / 'modules' / 'stitcher.py',
    'ZProjector': REPO / 'modules' / 'zprojector.py',
    'VideoBuilder': REPO / 'modules' / 'video_builder.py',
    'StackBuilder': REPO / 'modules' / 'stack_builder.py',
}

BASE_PATH = REPO / 'modules' / 'protocol_post_processor.py'
COMMON_UTILS_PATH = REPO / 'modules' / 'common_utils.py'


def _method_node(path: pathlib.Path, class_name: str, method_name: str) -> ast.FunctionDef:
    source = path.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f'{class_name}.{method_name} not found in {path}')


# ---------------------------------------------------------------------------
# 1. capture_root threading (behavioral, pinned via AST: UI deps block exec)
# ---------------------------------------------------------------------------


def test_base_load_folder_threads_capture_root_into_kwargs():
    """ProtocolPostProcessor.load_folder must read protocol.capture_root()
    and put it into kwargs before _generate_filename runs."""
    method = _method_node(BASE_PATH, 'ProtocolPostProcessor', 'load_folder')
    src = ast.unparse(method)
    assert 'capture_root' in src, (
        'ProtocolPostProcessor.load_folder must thread capture_root into kwargs.'
    )
    assert '.capture_root()' in src, (
        'ProtocolPostProcessor.load_folder must call protocol.capture_root() to source the prefix.'
    )
    # Sanity: kwargs is populated before _generate_filename is called.
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
    assert assign_idx >= 0, 'capture_root must be assigned into kwargs in load_folder.'
    assert call_idx >= 0, (
        '_generate_filename call not found in load_folder; test needs updating for the new shape.'
    )
    assert assign_idx < call_idx, (
        f'capture_root must be threaded into kwargs at statement '
        f'{assign_idx} BEFORE the _generate_filename call site (which '
        f'leads to the per-group loop at statement {call_idx}).'
    )


def _assert_subclass_reads_capture_root_kwarg(class_name: str, path: pathlib.Path):
    method = _method_node(path, class_name, '_generate_filename')
    src = ast.unparse(method)
    assert 'capture_root' in src, (
        f'{class_name}._generate_filename must read capture_root from '
        f'kwargs to prefix the output filename.'
    )
    # Required access kwargs['capture_root'] (load_folder always threads it) is
    # the antifragile form; a defensive kwargs.get(...) default still satisfies
    # the read contract.
    reads_kwarg = (
        "kwargs['capture_root']" in src
        or 'kwargs["capture_root"]' in src
        or "kwargs.get('capture_root'" in src
        or 'kwargs.get("capture_root"' in src
    )
    assert reads_kwarg, (
        f'{class_name}._generate_filename must read capture_root from '
        f'kwargs (not from a DataFrame column).'
    )
    assert "'Custom Root'" not in src and '"Custom Root"' not in src, (
        f'{class_name}._generate_filename must NOT read the dormant '
        f'"Custom Root" DataFrame column -- that field is never written.'
    )


def test_all_post_processors_read_capture_root_from_kwargs():
    for class_name, path in POST_PROCESSORS.items():
        _assert_subclass_reads_capture_root_kwarg(class_name, path)


# ---------------------------------------------------------------------------
# 2. Filename outcomes via the canonical component path (fail-before/pass-after)
# ---------------------------------------------------------------------------


def _build(row: dict, **overrides) -> str:
    return build_step_name(step_components(row, known_layers=_LAYERS, **overrides))


def test_single_channel_stitch_drops_tile_keeps_channel():
    # Stitch spans all tiles: drop the tile token, keep the channel.
    row = {'Well': 'A1', 'Color': 'BF', 'Tile': 'A1', 'Z-Slice': '', 'Name': 'A1_BF_TA1'}
    name = _build(row, tile=None, scan_count=0, objective='4xOly', post=('stitched',))
    assert name == 'A1_BF_4xOly_0000_stitched'
    assert '_TA1' not in name


def test_composite_stitch_drops_tile_and_channel():
    # A composite-stitch spans all channels: its Color column is 'Composite',
    # so the channel is 'Composite' by construction -- the stale per-channel
    # token baked into the Name string is never consulted (Well is set).
    row = {'Well': 'A1', 'Color': 'Composite', 'Tile': 'A1', 'Z-Slice': '', 'Name': 'A1_BF_TA1'}
    name = _build(row, tile=None, scan_count=0, objective='4xOly', post=('stitched',))
    assert name == 'A1_Composite_4xOly_0000_stitched'
    assert '_TA1' not in name and '_BF' not in name


def test_stitch_drops_tile_even_when_tile_column_empty():
    # The exact #501 face: the post-record Tile column is empty while the Name
    # still carries 'TA1'. The old strip helper keyed on the empty column and
    # no-op'd, leaking the token. Setting tile=None omits it by construction,
    # and the stale Name token is never re-parsed (Well is set).
    row = {'Well': 'A1', 'Color': 'BF', 'Tile': '', 'Z-Slice': '', 'Name': 'A1_BF_TA1'}
    name = _build(row, tile=None, scan_count=0, post=('stitched',))
    assert name == 'A1_BF_0000_stitched'
    assert '_TA1' not in name


def test_hyperstack_drops_channel_and_z_keeps_tile():
    # A hyperstack collapses all channels (channel=None) AND all z-slices
    # (z_index=None) but keeps the tile -- a single slice index would mislabel
    # the whole stack.
    row = {'Well': 'A1', 'Color': 'BF', 'Tile': 'A1', 'Z-Slice': 3, 'Name': 'A1_BF_TA1_Z3'}
    name = _build(row, channel=None, z_index=None, objective='4xOly', post=('hyperstack',))
    assert name == 'A1_TA1_4xOly_hyperstack'
    assert '_BF' not in name and '_Z3' not in name


def test_zprojection_drops_z_keeps_channel_and_tile():
    # A z-projection collapses every z-slice (z_index=None) but keeps channel
    # and tile (per-channel, per-tile output). The post chain records both the
    # source's stitched state and the projection.
    row = {'Well': 'A1', 'Color': 'BF', 'Tile': 'A1', 'Z-Slice': 3, 'Name': 'A1_BF_TA1_Z3'}
    name = _build(row, z_index=None, scan_count=0, post=('stitched', 'zproj_median'))
    assert name == 'A1_BF_TA1_0000_stitched_zproj_median'
    assert '_Z3' not in name


def test_chained_post_outputs_carry_both_suffixes():
    # A video of an already-stitched output carries the ordered chain
    # ('stitched', 'video') -- the single-str post field could only hold one,
    # dropping the other. Video keeps z (one slice per video).
    row = {'Well': 'A1', 'Color': 'BF', 'Tile': 'A1', 'Z-Slice': 3, 'Name': 'A1_BF_TA1_Z3'}
    name = _build(row, post=('stitched', 'video'))
    assert name == 'A1_BF_TA1_Z3_stitched_video'


# ---------------------------------------------------------------------------
# 3. Migration guard: the strip helpers are deleted and stay deleted; every
#    post-processor builds through the canonical builder. Replaces the prior
#    AST locks that asserted the strip helpers were CALLED.
# ---------------------------------------------------------------------------


def test_strip_helpers_are_deleted():
    common_src = COMMON_UTILS_PATH.read_text()
    assert 'def strip_tile_token' not in common_src, (
        'strip_tile_token must stay deleted -- the canonical builder omits a '
        'token by construction, no string-stripping helper needed.'
    )
    assert 'def strip_any_channel_token' not in common_src, (
        'strip_any_channel_token must stay deleted.'
    )
    composite_src = POST_PROCESSORS['CompositeGeneration'].read_text()
    assert 'def _strip_channel_token' not in composite_src, (
        'composite_generation._strip_channel_token must stay deleted.'
    )


def test_post_processors_build_through_canonical_builder():
    for class_name, path in POST_PROCESSORS.items():
        method = _method_node(path, class_name, '_generate_filename')
        src = ast.unparse(method)
        assert 'build_step_name' in src, (
            f'{class_name}._generate_filename must build its name through '
            f'build_step_name, not the legacy append-if-absent builder.'
        )
        assert 'strip_tile_token' not in src, (
            f'{class_name}._generate_filename must not strip a tile token; '
            f'omit it via tile=None instead.'
        )
        assert 'strip_any_channel_token' not in src, (
            f'{class_name}._generate_filename must not strip a channel token; '
            f'set channel via the component instead.'
        )
        assert '_strip_channel_token' not in src, (
            f'{class_name}._generate_filename must not strip a channel token.'
        )
