# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``wire_encoding.PROJECT_PACKAGES`` is every top-level package and module the application imports.

The API's table (``wire_encoding.project_classes``) takes in a class an
annotation reaches when the class is defined in one of these names. It is
read from the module's name, not its file, because a frozen build puts
every module's file in one folder; so the names are a list, and this guard
holds the list to the source tree: a package added beside ``modules/`` and
left out would let a class the API hands out from it pass every rule unseen.
"""

from __future__ import annotations

from modules.wire_encoding import PROJECT_PACKAGES
from tests.ast_seams import tracked_python_files

# Top-level directories holding Python the application never imports, each
# with its reason.
NOT_IMPORTED = {
    'tests': 'the suite',
    'docs': "the shipped examples, a client's own code",
    'scripts': "the release build's and the installers' own",
}


def _top_level_python() -> set[str]:
    """The top-level name each tracked ``.py`` is imported under: its first folder, or its own stem."""
    return {path.split('/', 1)[0].removesuffix('.py') for path in tracked_python_files()}


def test_the_project_packages_are_every_imported_top_level_name() -> None:
    imported = _top_level_python() - NOT_IMPORTED.keys()
    assert 'drivers' in imported, f'the walk did not reach drivers/: {sorted(imported)}'

    assert imported == PROJECT_PACKAGES, (
        'wire_encoding.PROJECT_PACKAGES is not the repository: '
        f'missing {sorted(imported - PROJECT_PACKAGES)}, not in the tree '
        f'{sorted(PROJECT_PACKAGES - imported)}. A package the application imports is '
        'added there; one it never imports, to NOT_IMPORTED with the reason.'
    )
