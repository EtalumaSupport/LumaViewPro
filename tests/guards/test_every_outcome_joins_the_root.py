# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every exception the API raises is a LumascopeError, and says which kind it is.

A caller catching what the scope raises, in Python or behind a REST server,
needs one base to catch and one answer to "what is this": a refusal (declined;
nothing broke), a notice (information), a quiet outcome (recorded, never
shown) or a fault (something failed). The answer is the class's, read by the
one reporter (``notification_center.outcome_of``), so a class with no kind
reads as a fault by default and a class with two reads as whichever the
reporter asks about first.

The umbrellas (``ConfigError``, ``ProtocolError``, ``CaptureError``,
``AutofocusAborted``) name a subject, not a kind: each child states its own.

Discovered, not listed: every class in ``modules/``, ``drivers/`` and
``rest/`` whose bases reach a built-in exception. The simulator walker's two
(``ui/``) never reach an API caller and are not read.
"""

from __future__ import annotations

import ast
import builtins
import importlib

import pytest

from lib.lumascope_error import Fault, LumascopeError
from modules.exceptions import (
    AutofocusAborted,
    CaptureError,
    ConfigError,
    InstallationFileError,
    Notice,
    ProtocolError,
    Quiet,
    Refusal,
    _Umbrella,
)
from tests.ast_seams import REPO_ROOT, module_name

KINDS = (Refusal, Notice, Quiet, Fault)
UMBRELLAS = (ConfigError, ProtocolError, CaptureError, AutofocusAborted)
_BUILTIN_EXCEPTIONS = {
    name
    for name, value in vars(builtins).items()
    if isinstance(value, type) and issubclass(value, BaseException)
}


def _exception_classes():
    """(module name, class name) for every exception class the API's layers define."""
    found = []
    for layer in ('modules', 'drivers', 'rest'):
        for path in sorted((REPO_ROOT / layer).rglob('*.py')):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            classes = {
                node.name: [ast.unparse(base).split('.')[-1] for base in node.bases]
                for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef)
            }
            module = module_name(path.relative_to(REPO_ROOT).as_posix())
            found.extend((module, name) for name, bases in classes.items() if bases)
    return found


def _resolved():
    classes = []
    for module, name in _exception_classes():
        cls = getattr(importlib.import_module(module), name, None)
        # The kinds and the umbrella mixin sit on the root; none is an outcome.
        if cls in KINDS or cls is _Umbrella:
            continue
        if isinstance(cls, type) and issubclass(cls, BaseException):
            classes.append(cls)
    return classes


EXCEPTION_CLASSES = _resolved()


def test_the_discovery_finds_the_api_s_exceptions():
    assert len(EXCEPTION_CLASSES) >= 100, 'the AST walk drifted and found too few classes'
    names = {cls.__name__ for cls in EXCEPTION_CLASSES}
    assert {'HardwareError', 'ProtocolRunRefusedError', 'ServerRefusedError'} <= names


@pytest.mark.parametrize('cls', EXCEPTION_CLASSES, ids=lambda cls: cls.__qualname__)
def test_every_exception_is_a_lumascope_error(cls):
    assert issubclass(cls, LumascopeError), (
        f'{cls.__module__}.{cls.__qualname__} is not a LumascopeError: a caller catching '
        'what the scope raises misses it'
    )


@pytest.mark.parametrize('cls', EXCEPTION_CLASSES, ids=lambda cls: cls.__qualname__)
def test_every_exception_has_exactly_one_kind_or_is_an_umbrella(cls):
    kinds = [kind.__name__ for kind in KINDS if issubclass(cls, kind)]
    if cls in UMBRELLAS:
        assert kinds == [], f'{cls.__name__} is an umbrella and states no kind; got {kinds}'
        return
    assert len(kinds) == 1, (
        f'{cls.__module__}.{cls.__qualname__} must be exactly one of Refusal, Notice, Quiet '
        f'or Fault; it is {kinds or "none"}'
    )


@pytest.mark.parametrize('cls', EXCEPTION_CLASSES, ids=lambda cls: cls.__qualname__)
def test_except_exception_still_catches_every_one(cls):
    """Joining the root moves no class out of an existing ``except Exception``."""
    assert issubclass(cls, Exception), (
        f'{cls.__qualname__} left Exception: every except Exception would stop catching it'
    )


@pytest.mark.parametrize('umbrella', UMBRELLAS, ids=lambda cls: cls.__name__)
def test_an_umbrella_cannot_be_raised(umbrella):
    """A raise of the subject alone has no kind, no reason and no title to answer with."""
    with pytest.raises(TypeError, match='never raised'):
        umbrella('a message', 'a_reason') if umbrella is CaptureError else umbrella('a message')


def test_an_umbrella_s_children_are_still_caught_by_it():
    from modules.exceptions import CaptureFailedError, ConfigurationDefectError

    assert isinstance(ConfigurationDefectError('settings_not_prepared', 'x'), ConfigError)
    assert isinstance(CaptureFailedError('x', 'no_frame_returned'), CaptureError)


def test_a_configuration_defect_outside_its_vocabulary_fails_where_it_is_raised():
    from modules.exceptions import ConfigurationDefectError

    with pytest.raises(TypeError, match='nope'):
        ConfigurationDefectError('nope', 'x')


def test_a_capture_failure_outside_its_vocabulary_fails_where_it_is_raised():
    from modules.exceptions import CaptureFailedError

    with pytest.raises(TypeError, match='nope'):
        CaptureFailedError('x', 'nope')


def test_a_defect_s_words_are_shown_only_where_they_are_written_for_the_person():
    from modules.exceptions import CaptureFailedError, ConfigurationDefectError
    from modules.notification_center import outcome_of

    assert outcome_of(ConfigurationDefectError('settings_not_prepared', 'x')).for_person is False
    assert outcome_of(ConfigurationDefectError('layers_unresolved', 'x')).for_person is True
    assert outcome_of(CaptureFailedError('x', 'no_frame_returned')).for_person is True
    assert outcome_of(CaptureFailedError('x', 'unknown_save_encoding')).for_person is False


def test_no_kind_is_itself_an_exception():
    """A kind is a mixin on the root, so naming one never widens what ``except Exception`` catches."""
    for kind in (*KINDS, LumascopeError):
        assert not issubclass(kind, Exception), kind.__name__
        assert kind.__name__ not in _BUILTIN_EXCEPTIONS


def test_a_fault_s_words_reach_the_person_only_when_its_type_says_so():
    """Whose words a fault's are is its class's, not a list kept beside the reporter."""
    from drivers.exceptions import HardwareError
    from modules.notification_center import outcome_of

    # Written for the person, and now shown as written: the file and what is
    # wrong with it, where a generic sentence stood before.
    installation = outcome_of(InstallationFileError('/x/objectives.json', 'is missing'))
    assert installation.kind == 'fault'
    assert installation.for_person is True

    assert outcome_of(HardwareError('no response from motor board')).for_person is True
    # A Python error's words are a developer's: the person reads the generic sentence.
    assert outcome_of(RuntimeError('list index out of range')).for_person is False
