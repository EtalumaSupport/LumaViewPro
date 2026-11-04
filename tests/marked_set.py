# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The declared API: everything that carries a mark, wherever it is defined.

A class's marked member or published field, keyed ``Class.member``, and a
module's marked top-level function, keyed by its dotted path
(``rest.app.build_app``), each with its owner -- the class, or the module --
carried beside the key rather than split from it. The guards read this one
set; the wire and the doors read a member's own mark, so nothing in
production enumerates marks across packages.

The guards run under ``tests/conftest.py``, whose stand-ins answer any
attribute, a mark among them. So a mark is read only as ``API`` or
``IN_PROCESS``, only on a real function or class defined in the module that
holds it, and a module that does not import raises, naming it, never
skipped.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import pkgutil
import types
from functools import cache

from modules import wire_encoding
from modules.api_surface import API, IN_PROCESS, fields_of, mark_of

MARKS = frozenset({API, IN_PROCESS})


@dataclasses.dataclass(frozen=True)
class Marked:
    """One thing the declared API holds.

    Attributes:
        owner: The class that defines it, or the module of a top-level function.
        owner_name: The class's name in ``project_classes()``, or the module's dotted path.
        member: Its own name in its owner.
        tag: ``API`` or ``IN_PROCESS``; a published field is ``API``.
        field: Whether it is a published field rather than a marked member.
    """

    owner: type | types.ModuleType
    owner_name: str
    member: str
    tag: str
    field: bool = False

    @property
    def value(self) -> object:
        """What its owner holds under its name: the member or function; for a field, whatever the class holds."""
        return vars(self.owner).get(self.member)

    @property
    def at_top_level(self) -> bool:
        """Whether it is a module's function rather than a class's member or field."""
        return isinstance(self.owner, types.ModuleType)


@cache
def project_modules() -> tuple[types.ModuleType, ...]:
    """Every module of every ``wire_encoding.PROJECT_PACKAGES`` package, imported.

    A package's modules are what ``pkgutil`` lists, as the wire's own walk
    of ``modules/`` lists them. A stand-in the conftest installs for a whole
    package (``lvp_logger``) is no module and holds no mark.

    Raises:
        ImportError: A module does not import; the error names it.
    """
    names = []
    for package in sorted(wire_encoding.PROJECT_PACKAGES):
        top = importlib.import_module(package)
        names.append(package)
        if hasattr(top, '__path__'):
            names += [info.name for info in pkgutil.walk_packages(top.__path__, f'{package}.')]
    found = []
    for name in names:
        try:
            module = importlib.import_module(name)
        except Exception as error:
            raise ImportError(f'{name} does not import, so its marks cannot be read') from error
        if isinstance(module, types.ModuleType):
            found.append(module)
    return tuple(found)


def marked_functions(modules: tuple[types.ModuleType, ...] | None = None) -> dict[str, Marked]:
    """Every top-level function marked ``API`` or ``IN_PROCESS`` in *modules*, the project's by default."""
    found = {}
    for module in project_modules() if modules is None else modules:
        for name, value in vars(module).items():
            if not inspect.isfunction(value) or value.__module__ != module.__name__:
                continue
            tag = mark_of(value)
            if tag in MARKS:
                found[f'{module.__name__}.{name}'] = Marked(module, module.__name__, name, tag)
    return found


def marked_members(universe: dict[str, type]) -> dict[str, Marked]:
    """Every marked member and published field of the classes in *universe*, by ``Class.member``."""
    found = {}
    for cname, cls in universe.items():
        for member, value in vars(cls).items():
            tag = mark_of(value)
            if tag in MARKS:
                found[f'{cname}.{member}'] = Marked(cls, cname, member, tag)
        for name in fields_of(cls):
            found[f'{cname}.{name}'] = Marked(cls, cname, name, API, field=True)
    return found


def marked_set(universe: dict[str, type] | None = None) -> dict[str, Marked]:
    """The declared API: *universe*'s marked members and fields, the project's by default, and every marked function."""
    universe = wire_encoding.project_classes() if universe is None else universe
    return marked_members(universe) | marked_functions()


def marked_classes_outside(universe: dict[str, type]) -> list[str]:
    """Each class defined in a project module that carries a mark and is not, by identity, in *universe*."""
    held = {id(cls) for cls in universe.values()}
    outside = []
    for module in project_modules():
        for value in vars(module).values():
            if not inspect.isclass(value) or value.__module__ != module.__name__:
                continue
            if id(value) in held:
                continue
            if fields_of(value) or any(mark_of(v) in MARKS for v in vars(value).values()):
                outside.append(f'{module.__name__}.{value.__qualname__}')
    return sorted(outside)
