# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Structural guard: every public L2 member is documented or declares it is not.

The companion guard beside this one checks doc -> code: every call form in
LumascopeSkills.md resolves and binds against a live object. Nothing asked the
inverse, so a member could ship with no mention in the reference forever -- and
two separate backlogs of undocumented public members accumulated exactly that
way, the second one found by a census that had to be written from scratch
because the first one's list had gone stale.

The rule here is the inverse direction: a public member on the L2 surface
either appears in the reference, or says in its own docstring that it is not
part of that surface. A member that is neither is the illegal state, and it is
what this test makes unconstructible.

Why the declaration lives in the docstring and not in a list here
----------------------------------------------------------------
An exclusion list in a test file is a hand-maintained roster, and a roster is
the thing that rotted twice. It also puts the knowledge in the wrong place: the
person adding an internal helper is editing the helper, not this file, and will
never think to come here. The declaring sentence was already in the tree before
this guard existed -- `create_diagnostic` and `move_to_simulated_sample_plane`
both carry it -- so this adopts a convention the code had already invented
rather than inventing one.

Privatizing those members instead (a leading underscore) is the more honest
signal and remains available, but renaming a public member is an
external-schema change that needs consumer notification; the declaration costs
nothing and can be done today.

What this proves, and what it does not
--------------------------------------
ABSENCE is reliable: a name appearing nowhere in the reference is genuinely
undocumented, and that is the defect being guarded. PRESENCE is WEAK -- the
check is the bare name matched anywhere in the file, so a changelog line counts
as a mention. This guard therefore catches "nobody documented it at all"; it
does not grade the quality of an entry. A stronger check was attempted and
abandoned: distinguishing a fenced mention from prose needs a fence parser, and
the naive version counted every fenced example as unmentioned. Upgrading this
is worth doing the day someone has a reason; it is not worth a roster today.

The surface is DERIVED from a live scope, never listed: a new sub-API is
covered on the day it is wired up. The first two attempts at this census
guessed sub-API names, missed `scope.protocols` entirely and dropped
`scope.capabilities` because its type lives outside the api package.
"""

import inspect
import re
import warnings

import pytest

from tests.ast_seams import REPO_ROOT

DOC = REPO_ROOT / 'docs' / 'LumascopeSkills.md'

#: The sentence a member uses to declare itself off the L2 surface.
NON_L2_DECLARATION = 'not part of the L2 API surface'

_SKIP_TYPE_MODULES = ('builtins', 'threading', 'logging', 'queue')


@pytest.fixture(scope='module')
def live_scope():
    from modules.lumascope_api import Lumascope

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Lumascope(simulate=True)


def _public_members(obj):
    cls = type(obj)
    out = []
    for name in dir(cls):
        if name.startswith('_'):
            continue
        attr = inspect.getattr_static(cls, name, None)
        if inspect.isfunction(attr) or isinstance(attr, (property, staticmethod, classmethod)):
            out.append((name, attr))
    return sorted(out)


def _l2_surface(scope):
    """Every object an L2 caller reaches from the scope, derived not listed."""
    surface, seen_types = [('scope', scope)], {type(scope)}
    for name in sorted(dir(scope)):
        if name.startswith('_'):
            continue
        try:
            value = getattr(scope, name)
        except Exception:
            continue
        if isinstance(value, (str, int, float, bool, type(None), list, dict, tuple, set)):
            continue
        if inspect.isroutine(value) or type(value) in seen_types:
            continue
        if type(value).__module__.startswith(_SKIP_TYPE_MODULES):
            continue
        seen_types.add(type(value))
        if _public_members(value):
            surface.append((f'scope.{name}', value))
    return surface


def _declares_non_l2(attr):
    # Unwrap to the underlying function before reading the docstring. A
    # property hides it behind fget, and classmethod / staticmethod behind
    # __func__ -- miss either and a declared member is reported undeclared,
    target = attr.fget if isinstance(attr, property) else attr
    target = getattr(target, '__func__', target)
    # Collapse whitespace before matching: the declaration is prose and wraps
    # across lines wherever it happens to fall, so a flat substring search
    # misses a perfectly good declaration purely because of where the line
    # broke -- which is exactly how this guard first accused
    # `create_diagnostic`, whose docstring reads "not\npart of the L2 API
    # surface".
    flattened = re.sub(r'\s+', ' ', inspect.getdoc(target) or '')
    return NON_L2_DECLARATION in flattened


def test_every_public_member_is_documented_or_declared(live_scope):
    doc_text = DOC.read_text()
    undeclared = []
    for label, obj in _l2_surface(live_scope):
        for name, attr in _public_members(obj):
            if re.search(rf'\b{re.escape(name)}\b', doc_text):
                continue
            if _declares_non_l2(attr):
                continue
            undeclared.append(f'  {label}.{name}')

    assert not undeclared, (
        f'{len(undeclared)} public members are absent from LumascopeSkills.md and do '
        'not declare themselves off the L2 surface. Either document the member in the '
        f'reference, or state "{NON_L2_DECLARATION}" in its docstring and say what it '
        'is for instead.\n' + '\n'.join(undeclared)
    )


def test_the_surface_derivation_still_finds_the_sub_apis(live_scope):
    """A derivation that silently stops finding sub-APIs would report the whole
    surface documented while checking almost none of it -- the failure the
    roster this guard replaces was prone to."""
    found = {label for label, _ in _l2_surface(live_scope)}
    for required in ('scope.motion', 'scope.illumination', 'scope.imaging', 'scope.protocols'):
        assert required in found, f'{required} vanished from the derived surface: {sorted(found)}'
