# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Ratchet on tests that read production SOURCE TEXT to make assertions.

A test that asserts on source text passes or fails on formatting. `'def
x(' in src` breaks when the signature is wrapped, reformatted, or gains a
parameter, and it keeps passing when the function is gutted. The suite
carries a lot of it, and the count was drifting upward.

`tests/ast_seams.py` is the alternative: it asserts the SEAM (the name,
its parameters, its return annotation) and survives any
behavior-preserving refactor. Note that `ast_seams` itself calls
`read_text` -- reading the file to PARSE it is the fix, not the problem,
which is why this guard ratchets a total rather than banning the call.

Measured at introduction: 335 `read_text` calls across 115 test files.
Classified (see `classify_read_text_sites`):

    148  body-pin           a literal .py path -- the migration target
    139  single computed    path from a variable; needs a human read,
                            and includes the ast_seams infrastructure
     27  hygiene-scan       inside a file loop; legitimately reads many
                            files and never migrates
     15  data/doc read      .md/.json/.txt -- not source at all
      6  kv-pin             .kv has no AST seam, so it is justified

The real migration denominator is therefore NOT 335. It is at most the
148 body-pins plus whatever share of the 139 computed-path sites turn out
to be single-module pins -- the point of publishing the split rather than
one scary number. Migration itself is deferred work, sequenced after the
API shrink so pins on surface that is about to move are handled once.

Policy for a NEW pin, in preference order: use `ast_seams.assert_def`;
if the thing asserted genuinely has no seam (a `.kv` file, a doc example,
a comment's wording), add a `pin-justified:` comment saying why, and
raise the budget below in the same commit.
"""

from __future__ import annotations

import ast

from tests.ast_seams import iter_package_modules

# Total `read_text` call sites in tests/, measured at introduction.
# Ratchet: may fall freely, may not rise without a deliberate bump.
# pin-justified: raised 335 -> 360 when three branches merged onto the
# beta line at once. The pin was taken against beta, so it never saw the
# tests that arrived with them; the growth is those tests, not new
# source-text assertions on the pinned line's own code.
_READ_TEXT_SITE_BUDGET = 360

# Files containing at least one, recorded for the same reason.
# pin-justified: raised 115 -> 122 by the same merge.
_READ_TEXT_FILE_BUDGET = 122


def _nodes_inside_iteration(tree):
    """ids of nodes lexically inside a for-loop or comprehension."""
    inside = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor, ast.ListComp, ast.GeneratorExp, ast.SetComp)):
            for sub in ast.walk(node):
                inside.add(id(sub))
    return inside


def classify_read_text_sites():
    """Group every `read_text` call in tests/ by what it is doing.

    Returns a dict of category -> list of 'file:line'. Categories carry
    different dispositions, which is the whole reason to separate them:
    a hygiene scan should never migrate, a `.kv` pin cannot, and a
    body-pin should.
    """
    found: dict[str, list[str]] = {
        'body-pin': [],
        'single-computed': [],
        'hygiene-scan': [],
        'data-doc-read': [],
        'kv-pin': [],
    }
    for rel_path, tree in iter_package_modules(('tests',)):
        in_loop = _nodes_inside_iteration(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != 'read_text':
                continue
            expr = ast.unparse(node)
            if id(node) in in_loop:
                category = 'hygiene-scan'
            elif '.kv' in expr:
                category = 'kv-pin'
            elif '.py' in expr:
                category = 'body-pin'
            elif any(ext in expr for ext in ('.md', '.json', '.txt')):
                category = 'data-doc-read'
            else:
                category = 'single-computed'
            found[category].append(f'{rel_path}:{node.lineno}')
    return found


def test_source_pin_count_does_not_grow():
    """New tests should assert seams, not source text."""
    sites = classify_read_text_sites()
    total = sum(len(v) for v in sites.values())
    assert total <= _READ_TEXT_SITE_BUDGET, (
        f'{total} read_text call sites in tests/, over the recorded '
        f'{_READ_TEXT_SITE_BUDGET}.\n\n'
        f'Prefer tests.ast_seams.assert_def, which asserts the seam and '
        f'survives reformatting. If the thing being asserted genuinely has '
        f'no seam (a .kv file, a doc example, a comment), add a '
        f'"pin-justified:" comment saying why and raise '
        f'_READ_TEXT_SITE_BUDGET in this commit.\n\n'
        f'Current split: ' + ', '.join(f'{k}={len(v)}' for k, v in sorted(sites.items()))
    )


def test_source_pin_file_count_does_not_grow():
    """Held separately: 20 new pins in one file is different from 20 files."""
    sites = classify_read_text_sites()
    files = {entry.rsplit(':', 1)[0] for entries in sites.values() for entry in entries}
    assert len(files) <= _READ_TEXT_FILE_BUDGET, (
        f'{len(files)} test files now read source text, over the recorded '
        f'{_READ_TEXT_FILE_BUDGET}: '
        f'{sorted(files)[:10]}{" ..." if len(files) > 10 else ""}'
    )


def test_classification_covers_every_site():
    """The published split must account for every site.

    Without this the docstring's denominator could drift from reality
    while both ratchets above still pass -- a classification nobody can
    trust is worse than no classification, because the migration is
    scoped from it.
    """
    sites = classify_read_text_sites()
    total = sum(len(v) for v in sites.values())
    assert total > 0, 'the scan found nothing, so it is no longer scanning'
    for category, entries in sites.items():
        assert all(':' in entry for entry in entries), f'{category} has a malformed entry'
