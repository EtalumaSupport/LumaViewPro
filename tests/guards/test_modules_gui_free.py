# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""modules/ imports no GUI framework and no ui/ code.

The layering rule is that modules/ sits below ui/, so a headless or REST
caller can import any of it without a display. That held only by habit: the
previous guard read source LINE BY LINE, skipped indented lines as "deferred
imports", guessed at triple-quoted strings, and listed files with a
non-recursive glob -- so it saw 84 of the 99 files under modules/ and none of
modules/lumascope_api/, which is the L2 surface the rule most exists to
protect. Both real violations in the tree were function-local, so it reported
a clean sweep while both stood.

An AST walk sees every import node at any nesting depth, in every file the
package actually contains. A deferred import is still an import: it fails at
call time on a headless host, which is worse than failing at import time
because it fails in front of a user.
"""

from __future__ import annotations

import ast

from tests.ast_seams import iter_package_modules

# modules/ui_listener_bridge.py exists to forward engine state to whatever
# listener the host registered, and its one ui/ import is the GUI host's
# own layer object, resolved lazily so headless never reaches it. Retiring
# the import means giving the bridge a registration seam instead; that is
# tracked as its own piece of work, and the exemption is named here so the
# guard stays green without becoming blind.
_UI_IMPORT_EXEMPT = frozenset({'modules/ui_listener_bridge.py'})

# No allowlist. A GUI-framework import in modules/ has no legitimate form:
# the work either belongs in ui/, or it needs a scheduler injected by the
# host. Enumerating exceptions here would re-admit exactly what the rule
# forbids.
_KIVY_IMPORT_EXEMPT: frozenset[str] = frozenset()


def _imported_roots(tree: ast.Module):
    """Yield (lineno, dotted_name) for every import anywhere in the module."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            # A relative import has no module root to police.
            if node.level == 0 and node.module:
                yield node.lineno, node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name


def _violations(prefix: str, exempt: frozenset[str]) -> list[str]:
    found = []
    for rel_path, tree in iter_package_modules(('modules',)):
        if rel_path in exempt:
            continue
        for lineno, name in _imported_roots(tree):
            if name == prefix or name.startswith(f'{prefix}.'):
                found.append(f'{rel_path}:{lineno} imports {name}')
    return found


def test_modules_imports_no_kivy():
    violations = _violations('kivy', _KIVY_IMPORT_EXEMPT)
    assert not violations, (
        'modules/ must import no GUI framework -- a headless or REST caller '
        'imports these:\n  ' + '\n  '.join(violations)
    )


def test_modules_imports_no_ui():
    violations = _violations('ui', _UI_IMPORT_EXEMPT)
    assert not violations, 'modules/ sits below ui/ and must not import it:\n  ' + '\n  '.join(
        violations
    )


def test_the_sweep_reaches_the_whole_package():
    """The gap that let both real violations hide.

    The retired guard globbed modules/*.py non-recursively, so the L2 API
    package was never examined. Pin the walker's reach directly, or a future
    narrowing goes unnoticed exactly the way the last one did.
    """
    swept = {rel for rel, _ in iter_package_modules(('modules',))}
    assert 'modules/lumascope_api/__init__.py' in swept or any(
        rel.startswith('modules/lumascope_api/') for rel in swept
    ), 'the sweep must reach modules/lumascope_api/'
    assert any(rel.startswith('modules/plugins/') for rel in swept), (
        'the sweep must reach modules/plugins/'
    )
    assert len(swept) > 90, f'expected the whole package, swept only {len(swept)}'


def test_the_exemption_is_real():
    """An exemption for a file with nothing to exempt is a stale allowlist."""
    for rel_path, tree in iter_package_modules(('modules',)):
        if rel_path not in _UI_IMPORT_EXEMPT:
            continue
        assert any(name == 'ui' or name.startswith('ui.') for _, name in _imported_roots(tree)), (
            f'{rel_path} is exempt from the ui/ ban but imports no ui/ module'
        )
