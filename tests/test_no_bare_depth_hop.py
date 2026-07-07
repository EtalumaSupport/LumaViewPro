# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Guard the pixel-depth+color hop SHAPE, not the bare name `color`.

A function that receives BOTH the payload depth (`significant_bits`) AND a
color/encoding signal as loose scalars is a "hop": every such site is a place a
later edit can drop or diverge one value from the other (the original post-proc
depth crash grew from exactly this -- a depth threaded as a parallel scalar).
Terminal serializers and renderers legitimately need both together to write the
file; those are allowlisted by exact symbol. Any NEW co-occurrence fails this
test, forcing the author to either couple the pair in a typed carrier or
consciously add a genuine sink to the allowlist.

Keying on the co-occurring SHAPE (not the name `color`) is deliberate: ~18 legit
`color=` params exist across the codebase, so banning the name is useless; it is
the pairing with a loose depth scalar that marks the droppable seam.
"""

from __future__ import annotations

import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
SCAN_DIRS = ('modules', 'drivers', 'ui')

DEPTH_PARAMS = frozenset({'significant_bits'})
COLOR_PARAMS = frozenset({'color', 'is_color', 'save_encoding', 'true_color', 'use_color'})

# Terminal serializer / renderer sinks where the depth and the color/encoding
# signal genuinely arrive together to write or draw the frame. Keyed by
# (repo-relative posix path, function name) so a rename or move prunes the entry
# instead of silently protecting an unrelated function.
ALLOWLISTED_SINKS = frozenset(
    {
        ('modules/image_save.py', 'prepare_image_for_saving'),
        ('modules/image_save.py', 'save_image'),
        ('modules/image_utils.py', 'encode_display_jpg'),
        ('modules/image_utils.py', 'write_tiff'),
        ('modules/image_utils.py', 'add_scale_bar'),
        ('modules/image_utils.py', '_compute_scale_bar_overlay'),
    }
)


def _param_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """All declared parameter names: positional, positional-only, keyword-only."""
    args = node.args
    return {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}


def _find_depth_color_hops(tree: ast.Module, rel_path: str):
    """Yield (rel_path, func_name, lineno) for defs taking depth AND color."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        names = _param_names(node)
        if names & DEPTH_PARAMS and names & COLOR_PARAMS:
            yield (rel_path, node.name, node.lineno)


def _scan_production():
    """Every depth+color co-occurring def across the scanned source dirs."""
    hops = []
    for d in SCAN_DIRS:
        for path in sorted((REPO / d).rglob('*.py')):
            tree = ast.parse(path.read_text())
            rel = path.relative_to(REPO).as_posix()
            hops.extend(_find_depth_color_hops(tree, rel))
    return hops


def test_no_unallowlisted_depth_color_hop():
    """No production function pairs a loose depth scalar with a loose color
    scalar unless it is an allowlisted terminal serializer/renderer sink."""
    offenders = [
        (rel, name, lineno)
        for (rel, name, lineno) in _scan_production()
        if (rel, name) not in ALLOWLISTED_SINKS
    ]
    assert offenders == [], (
        'New pixel-depth+color hop(s) found -- a loose `significant_bits` scalar '
        'travelling alongside a loose color/encoding scalar is a place the two '
        'can be dropped or diverged:\n'
        + '\n'.join(f'  {rel}:{lineno}  {name}(...)' for rel, name, lineno in offenders)
        + '\n\nCouple depth+color in a typed carrier so the pair cannot separate, '
        'or -- if this is a genuine terminal serializer/renderer where both must '
        'arrive together -- add (path, name) to ALLOWLISTED_SINKS in this file.'
    )


def test_allowlist_has_no_stale_entries():
    """Every allowlisted sink still exists as a depth+color def. A rename or
    removal must prune its entry, not leave a stale rule protecting nothing."""
    present = {(rel, name) for (rel, name, _lineno) in _scan_production()}
    stale = sorted(ALLOWLISTED_SINKS - present)
    assert stale == [], (
        'These ALLOWLISTED_SINKS entries no longer name a depth+color function '
        '(renamed, moved, or no longer takes both params) -- remove them:\n'
        + '\n'.join(f'  {rel}  {name}' for rel, name in stale)
    )


def test_guard_fires_on_a_synthetic_hop():
    """The detector is not vacuous: a non-allowlisted depth+color def is flagged."""
    src = 'def hand_off(image, significant_bits, color):\n    return image\n'
    hops = list(_find_depth_color_hops(ast.parse(src), 'fake/module.py'))
    assert hops == [('fake/module.py', 'hand_off', 1)]
