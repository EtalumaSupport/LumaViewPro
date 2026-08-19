# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Liveness checks for the retirement tripwire.

The tripwire exists to catch a caller the static census could not see, during
the window before a member is made private or deleted. A tripwire that does
not actually fire is worse than none: it converts "we looked" into "we
checked", and the member gets removed on the strength of a silence that was
never listening.

So these do not test that the code is present -- they test that it trips, that
it stays quiet for the callers it is supposed to ignore, and that wrapping a
member does not change what the member does.
"""

from __future__ import annotations

import logging
import os

import pytest

from modules.lumascope_api import _retiring
from modules.lumascope_api._retiring import (
    PRIVATE,
    REMOVED,
    retirement_fires,
    retiring,
)


@pytest.fixture(autouse=True)
def _clean_record():
    _retiring.clear_retirement_fires()
    yield
    _retiring.clear_retirement_fires()


def _call_from(filename: str, fn):
    """Invoke fn from a frame whose file is `filename`.

    Compiling with an explicit filename is what lets a test stand in for a
    caller inside the API package without adding a fixture module to it.
    """
    code = compile('result = fn()', filename, 'exec')
    scope = {'fn': fn}
    exec(code, scope)
    return scope['result']


INSIDE = os.path.join(os.path.dirname(os.path.abspath(_retiring.__file__)), 'pretend_caller.py')
OUTSIDE = '/somewhere/else/consumer.py'


def test_external_caller_trips_the_wire_and_warns(caplog):
    @retiring('imaging.doomed', becoming=REMOVED)
    def doomed():
        return 'value'

    with caplog.at_level(logging.WARNING, logger='LVP.api'):
        assert _call_from(OUTSIDE, doomed) == 'value'

    fires = retirement_fires()
    assert 'imaging.doomed' in fires, (
        'an external call did not register -- the tripwire is not listening'
    )
    assert any(OUTSIDE in site for site in fires['imaging.doomed'])
    assert any('imaging.doomed' in r.getMessage() for r in caplog.records), (
        'external call produced no warning'
    )


def test_internal_caller_is_counted_but_does_not_warn(caplog):
    @retiring('imaging.doomed', becoming=PRIVATE)
    def doomed():
        return 1

    with caplog.at_level(logging.WARNING, logger='LVP.api'):
        _call_from(INSIDE, doomed)

    assert not caplog.records, (
        'a call from inside the API package warned; that noise is exactly what '
        'buries the external call the wire is watching for'
    )
    assert retirement_fires(external_only=True) == {}
    assert 'imaging.doomed' in retirement_fires(external_only=False)


def test_repeat_calls_warn_once_per_site_but_keep_counting(caplog):
    @retiring('motion.doomed', becoming=PRIVATE)
    def doomed():
        return None

    other = '/somewhere/else/second_consumer.py'
    with caplog.at_level(logging.WARNING, logger='LVP.api'):
        _call_from(OUTSIDE, doomed)
        _call_from(OUTSIDE, doomed)
        _call_from(OUTSIDE, doomed)
        _call_from(other, doomed)

    assert len(caplog.records) == 2, (
        f'expected one warning per distinct call site, got {len(caplog.records)}'
    )
    sites = retirement_fires()['motion.doomed']
    assert sum(sites.values()) == 4, 'every call must be counted even when not logged'
    assert len(sites) == 2


def test_wrapping_does_not_change_behaviour():
    @retiring('imaging.passthrough', becoming=PRIVATE)
    def passthrough(a, b, *, c=3):
        """Docstring preserved."""
        return a + b + c

    assert passthrough(1, 2) == 6
    assert passthrough(1, 2, c=10) == 13
    assert passthrough.__name__ == 'passthrough'
    assert passthrough.__doc__ == 'Docstring preserved.'
    assert passthrough.__retiring__ == ('imaging.passthrough', PRIVATE)


def test_exceptions_propagate_unchanged():
    @retiring('imaging.raiser', becoming=REMOVED)
    def raiser():
        raise KeyError('original')

    with pytest.raises(KeyError, match='original'):
        raiser()

    assert 'imaging.raiser' in retirement_fires(external_only=False), (
        'a call that raised must still be recorded -- it was still a call'
    )


def test_property_getter_can_be_wrapped(caplog):
    class Thing:
        @property
        @retiring('motion.is_doomed', becoming=PRIVATE)
        def is_doomed(self):
            return True

    thing = Thing()
    with caplog.at_level(logging.WARNING, logger='LVP.api'):
        assert _call_from(OUTSIDE, lambda: thing.is_doomed) is True

    assert 'motion.is_doomed' in retirement_fires(external_only=False), (
        'property access did not register; the decorator order in the module docstring is wrong'
    )


def test_unknown_disposition_fails_at_decoration_time():
    with pytest.raises(ValueError, match='becoming must be one of'):

        @retiring('imaging.typo', becoming='deprecated')
        def _typo():
            pass
