# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Proof that the spec'd scope double bites, plus the MagicMock ratchet.

A fixture meant to reject wrong-world access is worthless unless
something proves it rejects. These tests are that proof: they assert the
double raises on the exact accesses a bare `MagicMock()` would have
accepted, including the two names from the confirmed wrong-world
families.
"""

from __future__ import annotations

import ast

import pytest

from tests.ast_seams import iter_package_modules
from tests.scope_fakes import spec_scope

# Test files that still build a bare MagicMock/Mock scope, counted at
# introduction. A RATCHET: the number may fall freely and may not rise.
# Bulk migration of these is deferred to the suite-quality batch; this
# only holds the line so the population cannot grow while that waits.
_MAGICMOCK_SCOPE_FILE_BUDGET = 19


class TestSpecScopeRejectsTheWrongWorld:
    def test_unknown_attribute_raises(self):
        """The whole point: a name the real scope lacks is not invented."""
        scope = spec_scope()
        with pytest.raises(AttributeError):
            _ = scope.no_such_capability

    def test_the_known_wrong_world_names_raise(self):
        """`led_on_fast` and `camera` are the confirmed dead probes.

        A bare MagicMock answers True for both, which is how the
        production branches guarding them stayed green. This double must
        not.
        """
        scope = spec_scope()
        with pytest.raises(AttributeError):
            _ = scope.led_on_fast
        with pytest.raises(AttributeError):
            _ = scope.camera

    def test_hasattr_is_false_for_a_name_the_real_scope_lacks(self):
        """`hasattr` must answer honestly, since production probes with it."""
        scope = spec_scope()
        assert not hasattr(scope, 'led_on_fast')
        assert not hasattr(scope, 'camera')

    def test_real_sub_api_access_is_allowed(self):
        """The inverse failure: rejecting legitimate production access.

        Every sub-API is assigned in `__init__`, so a CLASS autospec
        would have none of them and this test would fail -- which is
        exactly the inversion the instance autospec avoids.
        """
        scope = spec_scope()
        for sub_api in ('illumination', 'imaging', 'motion', 'diagnostics', 'io'):
            assert hasattr(scope, sub_api), f'{sub_api} missing from the double'
        scope.illumination.led_on(channel=0, mA=100)
        scope.illumination.led_on.assert_called_once_with(channel=0, mA=100)

    def test_wrong_signature_raises(self):
        """A specced method rejects a call the real one would reject."""
        scope = spec_scope()
        with pytest.raises(TypeError):
            scope.illumination.led_on(nonexistent_kwarg=1)

    def test_setting_an_unknown_attribute_raises(self):
        """A typo in a test's setup fails loudly instead of passing."""
        with pytest.raises(AttributeError):
            spec_scope(camera_conected=True)  # deliberate typo

    def test_setting_a_real_attribute_works(self):
        scope = spec_scope(camera_connected=True)
        assert scope.camera_connected is True


def _is_bare_mock_call(node) -> bool:
    """True for `MagicMock(...)` / `Mock(...)`, however imported."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    call_name = func.attr if isinstance(func, ast.Attribute) else getattr(func, 'id', '')
    return call_name in ('MagicMock', 'Mock')


def _binds_a_scope_name(name: str) -> bool:
    return 'scope' in name.lower()


def _files_building_magicmock_scopes():
    """Test files that hand a bare Mock/MagicMock to something scope-ish.

    AST-based rather than a regex so reformatting cannot change the count
    and a MagicMock named in a comment or docstring is not counted.

    Two shapes, both real and both found in the suite:

        scope = MagicMock()             an assignment (incl. self.scope)
        Thing(scope=MagicMock())        a keyword argument

    Worth recording how the number was arrived at, because two of the
    three attempts were wrong: a hand grep found 16, a first AST scan
    covering only assignments found 14 (a DIFFERENT 14 -- each method
    missed files the other caught), and covering both shapes finds 19.
    A ratchet blind to a shape is one that new code can adopt that shape
    to evade, so the budget is set from the broadest correct scan, not
    from either earlier undercount.
    """
    hits = set()
    for rel_path, tree in iter_package_modules(('tests',)):
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and _is_bare_mock_call(node.value):
                for target in node.targets:
                    name = (
                        target.attr
                        if isinstance(target, ast.Attribute)
                        else getattr(target, 'id', '')
                    )
                    if _binds_a_scope_name(name):
                        hits.add(rel_path)
            elif isinstance(node, ast.AnnAssign) and _is_bare_mock_call(node.value):
                target = node.target
                name = (
                    target.attr if isinstance(target, ast.Attribute) else getattr(target, 'id', '')
                )
                if _binds_a_scope_name(name):
                    hits.add(rel_path)
            elif isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg and _binds_a_scope_name(kw.arg) and _is_bare_mock_call(kw.value):
                        hits.add(rel_path)
    return hits


def test_magicmock_scope_population_does_not_grow():
    """Ratchet: no NEW test file may build a bare MagicMock scope.

    `spec_scope()` exists so new tests do not have to, and `sim_scope`
    is better still. Migrating the existing files is deferred work, but
    the population must not grow while that waits -- an unbounded
    "migrate opportunistically" is how the two wrong-world families
    survived long enough to reach production.
    """
    files = _files_building_magicmock_scopes()
    assert len(files) <= _MAGICMOCK_SCOPE_FILE_BUDGET, (
        f'{len(files)} test files now build a bare MagicMock scope, over the '
        f'recorded {_MAGICMOCK_SCOPE_FILE_BUDGET}:\n  '
        + '\n  '.join(sorted(files))
        + '\n\nUse the sim_scope fixture, or tests.scope_fakes.spec_scope(). '
        'If a bare MagicMock is genuinely required, raise the budget in this '
        'commit and say why.'
    )
