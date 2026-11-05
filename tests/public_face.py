# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``src/lumascope/__init__.py``: the public face, generated from the marks.

Every name the reference has a section for (``tests.api_reference.published``),
the root ``LumascopeError``, and every constant a module declares
(``api_surface.constants_of``), each imported from the module that defines
it. Type checkers read the same import lines, so the one list is the
module's ``__all__``. The file is regenerated, never edited, and
``tests/guards/test_the_public_face_is_generated.py`` holds it equal to a
fresh render.
"""

from __future__ import annotations

import collections
import sys
import types

from _lumascope.modules.api_surface import constants_of
from tests.ast_seams import REPO_ROOT

FACE = REPO_ROOT / 'src' / 'lumascope' / '__init__.py'
# Set, the guard writes the render to FACE instead of comparing it.
WRITE = 'LVP_WRITE_PUBLIC_FACE'
REGENERATE = (
    f'{WRITE}=1 python3 -m pytest tests/guards/test_the_public_face_is_generated.py -k current'
)

_DOC = '''\
"""The Lumascope API: every published name, imported from this one module.

    from lumascope import ScopeSession, RunStatus, PositionOutOfRangeError

A name is here when the API publishes it; ``docs/api-reference.md`` is each
one's reference. Everything under ``_lumascope`` is internal and can change
in any release. Importing this module loads the whole API.
"""'''


def declared_constants(modules: tuple[types.ModuleType, ...]) -> dict[str, str]:
    """Each constant a module in *modules* publishes, by name, with its module's dotted path.

    Raises:
        ValueError: A declared name is not one of its module's attributes, or
            two modules declare one name.
    """
    found: dict[str, str] = {}
    for module in modules:
        for name in constants_of(module):
            if name not in vars(module):
                raise ValueError(f'{module.__name__} declares {name}, which it does not define')
            if found.setdefault(name, module.__name__) != module.__name__:
                raise ValueError(f'{name} is declared by {found[name]} and {module.__name__}')
    return found


def homes(names: dict[str, object], constants: dict[str, str]) -> dict[str, str]:
    """Each published name with the dotted path of the module it is imported from.

    An object is imported from the module that defines it, where it must be
    held under its published name; a constant from the module that declares it.

    Raises:
        ValueError: A name is a constant and an object at once, or an object's
            module does not hold it under that name.
    """
    found = dict(constants)
    for name, value in names.items():
        if name in found:
            raise ValueError(f'{name} is both a declared constant and a published object')
        module = value.__module__
        if getattr(sys.modules.get(module), name, None) is not value:
            raise ValueError(f'{module} does not hold {name} at its top level')
        found[name] = module
    return found


def _all_order(name: str) -> tuple[int, str]:
    """``__all__``'s order, as ruff's RUF022 sorts it: constants, then classes, then the rest."""
    return (0 if name.isupper() else 1 if name[:1].isupper() else 2, name)


def face_text(found: dict[str, str]) -> str:
    """The public module's text: one import per defining module, then ``__all__``."""
    by_module: dict[str, list[str]] = collections.defaultdict(list)
    for name, module in found.items():
        by_module[module].append(name)
    lines = [
        '# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.',
        f'# Generated from the marks by tests/public_face.py; do not edit. Regenerate: {REGENERATE}',
        _DOC,
        '',
    ]
    for module in sorted(by_module):
        lines.append(f'from {module} import (')
        lines += [f'    {name},' for name in sorted(by_module[module])]
        lines.append(')')
    lines += ['', '__all__ = [']
    lines += [f"    '{name}'," for name in sorted(found, key=_all_order)]
    lines.append(']')
    return '\n'.join(lines) + '\n'
