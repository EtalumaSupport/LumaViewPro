# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Proof that the spec'd scope double bites.

A fixture meant to reject wrong-world access is worthless unless
something proves it rejects. These tests are that proof: they assert the
double raises on the exact accesses a bare `MagicMock()` would have
accepted, including the two names from the confirmed wrong-world
families.
"""

from __future__ import annotations

import pytest

from tests.scope_fakes import spec_scope


class TestSpecScopeRejectsTheWrongWorld:
    def test_unknown_attribute_raises(self):
        """The whole point: a name the real scope lacks is not invented."""
        # a stand-in by design: the double is the subject
        scope = spec_scope()
        with pytest.raises(AttributeError):
            _ = scope.no_such_capability

    def test_the_known_wrong_world_names_raise(self):
        """`led_on_fast` and `camera` are the confirmed dead probes.

        A bare MagicMock answers True for both, which is how the
        production branches guarding them stayed green. This double must
        not.
        """
        # a stand-in by design: the double is the subject
        scope = spec_scope()
        with pytest.raises(AttributeError):
            _ = scope.led_on_fast
        with pytest.raises(AttributeError):
            _ = scope.camera

    def test_hasattr_is_false_for_a_name_the_real_scope_lacks(self):
        """`hasattr` must answer honestly, since production probes with it."""
        # a stand-in by design: the double is the subject
        scope = spec_scope()
        assert not hasattr(scope, 'led_on_fast')
        assert not hasattr(scope, 'camera')

    def test_real_sub_api_access_is_allowed(self):
        """The inverse failure: rejecting legitimate production access.

        Every sub-API is assigned in `__init__`, so a CLASS autospec
        would have none of them and this test would fail -- which is
        exactly the inversion the instance autospec avoids.
        """
        # a stand-in by design: the double is the subject
        scope = spec_scope()
        for sub_api in ('illumination', 'imaging', 'motion', 'diagnostics'):
            assert hasattr(scope, sub_api), f'{sub_api} missing from the double'
        scope.illumination.led_on(channel=0, illumination_ma=100)
        scope.illumination.led_on.assert_called_once_with(channel=0, illumination_ma=100)

    def test_wrong_signature_raises(self):
        """A specced method rejects a call the real one would reject."""
        # a stand-in by design: the double is the subject
        scope = spec_scope()
        with pytest.raises(TypeError):
            scope.illumination.led_on(nonexistent_kwarg=1)

    def test_setting_an_unknown_attribute_raises(self):
        """A typo in a test's setup fails loudly instead of passing."""
        with pytest.raises(AttributeError):
            # a stand-in by design: the double is the subject
            spec_scope(camera_conected=True)  # deliberate typo

    def test_setting_a_real_attribute_works(self):
        # a stand-in by design: the double is the subject
        scope = spec_scope(camera_connected=True)
        assert scope.camera_connected is True
