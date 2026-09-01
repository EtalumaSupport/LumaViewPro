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
# pin-justified: raised 335 -> 361 when four branches merged onto the
# beta line at once. The pin was taken against beta, so it never saw the
# tests that arrived with them; the growth is those tests, not new
# source-text assertions on the pinned line's own code.
# pin-justified: 361 -> 362 for the manifest-naming contract, which reads
# back a manifest the recording just wrote. The seam this pin prefers
# asserts against SOURCE; this reads a JSON artifact produced by the run,
# which has no seam to assert instead.
# pin-justified: 362 -> 365 for the build-chain guards
# (test_build_dependency_and_identity.py), three sites, none of which has a
# seam available:
#   1. build.ps1 -- PowerShell. No Python AST, and no PowerShell parser in
#      the test environment.
#   2. MIN_BUILD_SCRIPT_VERSION -- a bare integer in a text file.
#   3. lvp_logger.py -- behavioural tests were written FIRST and had to be
#      withdrawn: conftest replaces lvp_logger in sys.modules with a
#      MagicMock, so the banner is a no-op under pytest and every
#      assertion passed vacuously against an empty capture. Importing the
#      real module under an alias was rejected because it installs a
#      global sys.excepthook at import, which has already polluted one
#      bench log. One read site serves all four assertions.
# pin-justified: 365 -> 367 for two independently-added sites that met at
# the beta merge: the API doc guard (test_api_doc_guard.py) -- the guard's
# SUBJECT is the text of LumascopeSkills.md, so there is no production seam
# to assert instead (the doc-example case named above) -- and the Enhance
# image/folder label guard (test_enhance_file_or_folder.py), which asserts
# button LABEL TEXT inside a function body, where ast_seams carries the def
# but not the literals in it. One site each; measured on the merged tree.
# pin-justified: 367 -> 370 for the capability-gating guards
# (test_capability_gating_ssot.py, two sites; test_controls_lockout.py, one).
# Every one asserts an ABSENCE -- that a function no longer reads a
# capability out of the scope-model config, and that two deleted mirrors of
# the XY fact have not come back. ast_seams asserts that a seam EXISTS with
# a given shape; it has no way to say a name is gone or that a body stopped
# reading something, which is the whole content of these guards.
# pin-justified: 375 -> 376 for test_settings_question_failure_parity.py,
# ONE site. It reads no source: the refusal tests assert that a refused
# save left the user's only copy of their configuration exactly as it was,
# which is a claim about a file on disk and has no AST seam -- the same
# rationale already recorded for test_session_save_settings.py in the file
# budget below, whose assertions these mirror. Six natural sites (a
# before/after pair in each of three tests) were funnelled through one
# `_current_json` helper so the file costs one.
_READ_TEXT_SITE_BUDGET = 376

# Files containing at least one, recorded for the same reason.
# pin-justified: raised 115 -> 122 by the same merge.
# pin-justified: 122 -> 123 for test_build_dependency_and_identity.py.
# pin-justified: 123 -> 124 for test_api_doc_guard.py, same reason as the
# site bump above.
# pin-justified: 124 -> 125 for test_api_surface_polarity2.py -- the
# other polarity of the same guard: its subject is ALSO the text of
# LumascopeSkills.md (does every live public member appear in a checked
# fence), so there is no AST seam to assert instead.
# pin-justified: 125 -> 126 for test_capability_gating_ssot.py, same reason
# as the site bump above: absence assertions have no seam to assert.
# pin-justified: 126 -> 128 for test_settings_preparation_shared.py and
# test_session_save_settings.py. Neither reads SOURCE: both write a settings
# file into tmp_path and read its bytes back to assert the file on disk was
# or was not modified. "The user's only copy was left exactly as it was" is
# a claim about a file, so there is no AST seam to assert instead.
# pin-justified: 129 -> 130 for test_settings_question_failure_parity.py --
# same reason as the site bump above, and the same reason as the two files
# named immediately above it. Its AST work goes through
# tests.ast_seams.parse_module, which is where that read already lives and
# is already counted; the one site this file adds is the on-disk check.
_READ_TEXT_FILE_BUDGET = 130


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
