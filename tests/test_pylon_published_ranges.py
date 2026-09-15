"""Published Basler ranges are values the camera will actually accept.

A GenICam float node reports its range continuously but accepts only
steps of its own increment, and the reported maximum can sit a float
epsilon past the last real step (48.00000004350822 on a 48 dB gain
node). That number became the slider cap and was persisted to the
settings store, so the advertised maximum read back as a long decimal
and the camera rejected the value it had advertised.

Gain and exposure are the same node shape and are tested together --
fixing one and leaving the other is how this kind of defect comes back.
"""

import math

import pytest

from drivers.pyloncamera import _accepted_range


class _Node:
    """A GenICam float node: a range, an increment, and an increment mode."""

    def __init__(self, low, high, increment=None, mode=None, inc_raises=False):
        self._low = low
        self._high = high
        self._increment = increment
        self._mode = mode
        self._inc_raises = inc_raises

    def GetMin(self):
        return self._low

    def GetMax(self):
        return self._high

    def GetIncMode(self):
        if self._mode is None:
            raise AttributeError('node has no increment mode')
        return self._mode

    def GetInc(self):
        if self._inc_raises:
            raise RuntimeError('GetInc is not valid for this increment mode')
        return self._increment


@pytest.fixture
def modes():
    from pypylon import genicam

    return genicam


def test_the_reported_gain_cap_loses_its_float_tail(modes):
    """The bench symptom: a typed 48 dB read back as 48.00000004350822."""
    node = _Node(0.0, 48.00000004350822, 0.01, modes.fixedIncrement)

    low, high = _accepted_range(node, 'Gain')

    assert high == 48.0
    assert low == 0.0


def test_the_cap_is_floored_to_the_grid_not_rounded(modes):
    """Rounding to nearest can land above the ceiling the camera enforces."""
    node = _Node(0.0, 47.999, 0.01, modes.fixedIncrement)

    _, high = _accepted_range(node, 'Gain')

    assert high == 47.99


def test_exposure_is_quantised_too(modes):
    """The sibling node, same shape -- and the one the first draft missed."""
    node = _Node(20.0, 10000.000000731, 1.0, modes.fixedIncrement)

    low, high = _accepted_range(node, 'ExposureTime')

    assert high == 10000.0
    assert low == 20.0


@pytest.mark.parametrize('mode_name', ['listIncrement', 'noIncrement'])
def test_a_node_without_a_fixed_step_is_published_as_reported(modes, mode_name):
    """There is no single step to snap to, and inventing one would narrow a
    range the camera did not narrow."""
    mode = getattr(modes, mode_name)
    node = _Node(0.0, 48.00000004350822, None, mode, inc_raises=True)

    low, high = _accepted_range(node, 'Gain')

    assert high == 48.00000004350822
    assert low == 0.0


def test_a_node_that_cannot_answer_its_mode_is_published_as_reported():
    """An IInteger node has no increment mode; it must not take the snap path."""
    node = _Node(0.0, 42.1, None, None)

    low, high = _accepted_range(node, 'Gain')

    assert (low, high) == (0.0, 42.1)


def test_an_already_exact_range_is_returned_unchanged(modes):
    node = _Node(0.0, 48.0, 0.01, modes.fixedIncrement)

    assert _accepted_range(node, 'Gain') == (0.0, 48.0)


@pytest.mark.parametrize(
    'low,high,increment',
    [(0.0, 48.0, 0.01), (20.0, 10000.0, 1.0), (0.5, 15.5, 0.25), (1.0, 2.0, 0.1)],
)
def test_the_published_max_is_never_above_the_reported_max(modes, low, high, increment):
    """Narrowing inward is the invariant; a bound above the real ceiling is
    the failure this exists to prevent."""
    node = _Node(low, high, increment, modes.fixedIncrement)

    got_low, got_high = _accepted_range(node, 'n')

    assert got_high <= high
    assert got_low >= low
    assert got_low <= got_high


def test_the_published_max_lands_on_the_increment_grid(modes):
    """The point of the exercise: the cap must be a value the node accepts."""
    node = _Node(0.0, 48.00000004350822, 0.01, modes.fixedIncrement)

    _, high = _accepted_range(node, 'Gain')

    steps = (high - 0.0) / 0.01
    assert abs(steps - round(steps)) < 1e-9


def test_an_inverted_range_is_not_made_worse(modes):
    """A node reporting max below min is already broken; the snap must not
    turn that into a bound above the reported max."""
    node = _Node(10.0, 5.0, 0.01, modes.fixedIncrement)

    low, high = _accepted_range(node, 'Gain')

    assert (low, high) == (10.0, 5.0)


def test_a_zero_increment_does_not_divide_by_zero(modes):
    node = _Node(0.0, 48.0, 0.0, modes.fixedIncrement)

    assert _accepted_range(node, 'Gain') == (0.0, 48.0)


def test_the_helper_matches_a_hand_computed_grid(modes):
    """An independent derivation of the same answer, so a sign or
    off-by-one in the production arithmetic shows up here."""
    low, high, increment = 0.5, 15.567, 0.25
    node = _Node(low, high, increment, modes.fixedIncrement)

    _, got = _accepted_range(node, 'n')

    expected = low + math.floor((high - low) / increment) * increment
    assert got == pytest.approx(expected)
    assert got <= high
