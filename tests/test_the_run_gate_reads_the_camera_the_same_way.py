# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""One camera predicate, so the run gate and the per-board gates agree.

``are_all_connected()`` -- what a run's preparation asks before it starts
-- tested ``is_connected()`` alone, while ``camera_connected`` tested
``driver.active and is_connected()``. Two spellings of one question, and
either could be changed without the other.

**Today the difference is invisible, and that is a fact about the DRIVERS,
not about these two gates.** All three cameras in the tree make
``is_connected()`` return False whenever ``active`` is unset -- the
simulator and the IDS driver test ``active in (False, None)`` first, the
Pylon driver tests ``active is None`` -- so the term the run gate was
missing is one its callee applies anyway. A driver that answered the two
independently would be enough to split them, which is what the stub below
is, and that is the state this pins out of existence rather than a bug a
user has met.

The property keeps its swallow: its callers are display and metrics paths
asking per frame, with no answer available but False. The run gate gets
the raise, because "the USB tree went away mid-question" is a different
refusal from "the camera is not connected" and the user is owed which.
"""

from __future__ import annotations

import pytest

from modules.lumascope_api import Lumascope


class _CameraAnsweringBothSeparately:
    """A driver whose is_connected() does NOT imply active.

    The shipped drivers couple the two inside is_connected(); nothing
    requires them to, and the API must not be the place that assumes it.
    """

    def __init__(self, *, active, connected):
        self.active = active
        self._connected = connected

    def is_connected(self):
        return self._connected


@pytest.fixture
def scope():
    scope = Lumascope(simulate=True)
    yield scope
    scope.disconnect()


class TestTheTwoGatesReadOnePredicate:
    def test_a_camera_that_is_not_active_fails_the_run_gate_too(self, scope):
        assert scope.are_all_connected() is True, 'precondition: the sim starts connected'

        scope._camera_driver = _CameraAnsweringBothSeparately(active=False, connected=True)

        assert scope.camera_connected is False, 'precondition: the per-board gate refuses it'
        assert scope.are_all_connected() is False, (
            'a camera the composite gate refuses must not pass the run gate'
        )

    def test_a_connected_active_camera_passes_both(self, scope):
        scope._camera_driver = _CameraAnsweringBothSeparately(active=True, connected=True)

        assert scope.camera_connected is True
        assert scope.are_all_connected() is True

    def test_the_answer_is_a_bool_even_when_active_is_none(self, scope):
        """A Pylon camera holds None in ``active`` once the device is gone.

        A predicate written as one and-chain returns that None, and the
        display thread logs the value it is given.
        """
        scope._camera_driver = _CameraAnsweringBothSeparately(active=None, connected=True)

        assert scope.camera_connected is False
        assert scope.are_all_connected() is False

    def test_the_shipped_driver_couples_the_two_inside_is_connected(self, scope):
        """Why the split above has never been reachable in the field."""
        scope._camera_driver.active = False

        assert scope._camera_driver.is_connected() is False


class TestADriverThatThrows:
    def test_the_run_gate_lets_the_throw_out(self, scope):
        """So prepare() can refuse "state unknown" rather than "disconnected"."""

        def _explode():
            raise RuntimeError('usb tree gone')

        scope._camera_driver.is_connected = _explode

        with pytest.raises(RuntimeError):
            scope.are_all_connected()

    def test_the_display_property_still_answers_false(self, scope):
        """Its callers run per frame and cannot handle a raise."""

        def _explode():
            raise RuntimeError('usb tree gone')

        scope._camera_driver.is_connected = _explode

        assert scope.camera_connected is False
