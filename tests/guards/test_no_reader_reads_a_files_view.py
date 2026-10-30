# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Only the writer touches a saved file's readable view; no LVP code reads it.

Every file LVP saves carries its record twice: typed, for LVP's one reader
(``read_file_record``), and as plain ``key = value`` text for a person in
Fiji -- an OME file's OriginalMetadata annotations, an ImageJ file's
``Info``, a plain file's first ImageDescription. The text is a rendering,
written by the one writer from the one record in the same call. A reader
that took a fact from it would make it a second store, parsed back from
text that has lost its types. This walk over every production module
refuses that at commit: the view's namespace, the ``Info`` key (in any
case, since tifffile takes ``info`` as the same key), a first page's
``description`` and the view's builders are named only by the writers.
"""

from __future__ import annotations

import ast

from tests.ast_seams import (
    MemberNames,
    iter_package_modules,
    parse_module,
    production_modules,
    walk_defs,
)

_WRITER_MODULE = 'modules/image_utils.py'
_WRITERS = MemberNames(
    (_WRITER_MODULE,),
    {'write_tiff', 'write_hyperstack_tiff', 'generate_tiff_data', '_splice_ome_additions'},
)
# The view's builders, and the namespace constant its annotations carry.
_VIEW_NAMES = MemberNames(
    (_WRITER_MODULE,), {'_readable_view', '_view_lines', '_ORIGINAL_METADATA_NAMESPACE'}
)
_DESCRIPTION_TAG = ('ImageDescription', 270)


def _all_production_modules():
    yield from production_modules()
    yield from iter_package_modules(('drivers', 'tools', 'rest'))


def _docstrings(tree: ast.Module) -> set[int]:
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                found.add(id(body[0].value))
    return found


def _is_info(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.lower() == 'info'
    )


def _keyed(node: ast.AST) -> ast.AST | None:
    """The key a subscript, a ``.get`` or an ``in`` test reads, else None."""
    if isinstance(node, ast.Subscript):
        return node.slice
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'get'
        and node.args
    ):
        return node.args[0]
    if isinstance(node, ast.Compare) and any(isinstance(op, ast.In) for op in node.ops):
        return node.left
    return None


def _touch(node: ast.AST, docstrings: set[int]) -> str | None:
    """What of the view ``node`` names, else None."""
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and 'OriginalMetadata' in node.value
        and id(node) not in docstrings
    ):
        return "the view's namespace"
    # A store is the name's definition, not a use of it.
    if (
        isinstance(node, ast.Name)
        and node.id in _VIEW_NAMES
        and not isinstance(node.ctx, ast.Store)
    ):
        return node.id
    if isinstance(node, ast.Attribute) and node.attr in _VIEW_NAMES:
        return node.attr
    key = _keyed(node)
    if key is not None and _is_info(key):
        return 'the Info key'
    if key is not None and isinstance(key, ast.Constant) and key.value in _DESCRIPTION_TAG:
        return 'the ImageDescription tag'
    if (
        isinstance(node, ast.Attribute)
        and node.attr == 'description'
        and isinstance(node.value, ast.Subscript)
        and isinstance(node.value.value, ast.Attribute)
        and node.value.value.attr == 'pages'
    ):
        return "a page's description"
    return None


def _touches(tree: ast.Module):
    """Yield (the outermost def or '<module>', lineno, what) for every touch of the view."""
    docstrings = _docstrings(tree)
    inside = set()
    for qualname, node in walk_defs(tree.body):
        for inner in ast.walk(node):
            what = _touch(inner, docstrings)
            if what is not None and id(inner) not in inside:
                inside.add(id(inner))
                yield qualname.split('.')[0], inner.lineno, what
    for inner in ast.walk(tree):
        what = _touch(inner, docstrings)
        if what is not None and id(inner) not in inside:
            yield '<module>', inner.lineno, what


def _owned(rel_path: str, owner: str, what: str) -> bool:
    if rel_path != _WRITER_MODULE:
        return False
    if owner in _WRITERS:
        return True
    # The view's own definitions: the builders and the namespace constant.
    return owner in _VIEW_NAMES or (owner == '<module>' and what == "the view's namespace")


def _offenders(rel_path: str, tree: ast.Module) -> list[str]:
    return [
        f'{rel_path}:{lineno} {owner} names {what}'
        for owner, lineno, what in _touches(tree)
        if not _owned(rel_path, owner, what)
    ]


def test_only_the_writer_names_a_files_view():
    found = [
        offender
        for rel_path, tree in _all_production_modules()
        for offender in _offenders(rel_path, tree)
    ]
    assert found == [], (
        "a saved file's readable view is written by the writer and read by no "
        'LVP code; take a fact from read_file_record instead: '
        f'{found}'
    )


def test_the_guard_sees_every_shape_of_a_read():
    planted = ast.parse(
        '\n'.join(
            (
                'def reader(tif, root):',
                '    tif.imagej_metadata["Info"]',
                '    tif.imagej_metadata.get("info")',
                '    "INFO" in tif.imagej_metadata',
                '    tif.pages[0].description',
                '    tif.pages[0].tags["ImageDescription"]',
                '    tif.pages[0].tags.get(270)',
                '    root.iter("openmicroscopy.org/OriginalMetadata")',
                '    image_utils._readable_view({})',
            )
        )
    )
    assert _offenders('modules/reader.py', planted) == [
        'modules/reader.py:2 reader names the Info key',
        'modules/reader.py:3 reader names the Info key',
        'modules/reader.py:4 reader names the Info key',
        "modules/reader.py:5 reader names a page's description",
        'modules/reader.py:6 reader names the ImageDescription tag',
        'modules/reader.py:7 reader names the ImageDescription tag',
        "modules/reader.py:8 reader names the view's namespace",
        'modules/reader.py:9 reader names _readable_view',
    ]


def test_the_guard_sees_the_writers_it_exempts():
    # The exemption must name the real writers, or a rename would leave the
    # guard exempting nothing while every other check still passed.
    owners = {owner for owner, _lineno, _what in _touches(parse_module(_WRITER_MODULE))}
    assert owners >= _WRITERS
