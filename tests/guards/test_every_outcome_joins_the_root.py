# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every exception the API raises is a LumascopeError, and says which kind it is.

A caller catching what the scope raises, in Python or behind a REST server,
needs one base to catch and one answer to "what is this": a refusal (declined;
nothing broke), a notice (information), a quiet outcome (recorded, never
shown) or a fault (something failed). The answer is the class's, read by the
one reporter (``notification_center.outcome_of``), so a class with no kind
reads as a fault by default and a class with two reads as whichever the
reporter asks about first.

The umbrellas (each class that names ``_Umbrella`` among its bases, such as
``ConfigError``) name a subject, not a kind: each child states its own.

Discovered, not listed: every class in ``modules/``, ``drivers/`` and
``rest/`` whose bases reach a built-in exception. The simulator walker's two
(``ui/``) never reach an API caller and are not read.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import inspect
import textwrap

import pytest

from lib.lumascope_error import Fault, LumascopeError
from modules.exceptions import (
    CaptureError,
    ConfigError,
    InstallationFileError,
    Notice,
    Quiet,
    Refusal,
    _Umbrella,
)
from tests.ast_seams import REPO_ROOT, module_name

KINDS = (Refusal, Notice, Quiet, Fault)
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
UMBRELLAS = tuple(cls for cls in EXCEPTION_CLASSES if _Umbrella in cls.__bases__)


def test_the_discovery_finds_the_api_s_exceptions():
    assert len(EXCEPTION_CLASSES) >= 100, 'the AST walk drifted and found too few classes'
    names = {cls.__name__ for cls in EXCEPTION_CLASSES}
    assert {'HardwareError', 'ProtocolRunRefusedError', 'ServerRefusedError'} <= names


def _named(classes):
    return [f'{cls.__module__}.{cls.__qualname__}' for cls in classes]


def test_every_exception_is_a_lumascope_error():
    outside = [cls for cls in EXCEPTION_CLASSES if not issubclass(cls, LumascopeError)]
    assert outside == [], (
        f'not LumascopeErrors, so a caller catching what the scope raises misses them: '
        f'{_named(outside)}'
    )


def test_every_exception_has_exactly_one_kind_or_is_an_umbrella():
    wrong = []
    for cls in EXCEPTION_CLASSES:
        kinds = [kind.__name__ for kind in KINDS if issubclass(cls, kind)]
        expected = 0 if cls in UMBRELLAS else 1
        if len(kinds) != expected:
            wrong.append(f'{cls.__module__}.{cls.__qualname__}: {kinds or "none"}')
    assert wrong == [], (
        'each must be exactly one of Refusal, Notice, Quiet or Fault, and an umbrella '
        f'none: {wrong}'
    )


def test_except_exception_still_catches_every_one():
    """Joining the root moves no class out of an existing ``except Exception``."""
    left = [cls for cls in EXCEPTION_CLASSES if not issubclass(cls, Exception)]
    assert left == [], f'every except Exception would stop catching these: {_named(left)}'


def test_no_exception_defines_new():
    """A ``__new__`` anywhere in a ``TimeoutError`` child's chain hands its
    arguments to ``OSError``, which reads two of them as an errno and its
    words: ``str()`` became ``[Errno Z] still_moving``. So kinds, umbrellas
    and outcomes construct in ``__init__`` only."""
    defining = [
        f'{cls.__module__}.{cls.__qualname__}'
        for cls in (*EXCEPTION_CLASSES, *KINDS, _Umbrella)
        if '__new__' in vars(cls)
    ]
    assert defining == [], f'construct in __init__, not __new__: {defining}'


def test_no_umbrella_can_be_raised():
    """A raise of the subject alone has no kind, no reason and no title to answer with."""
    assert {
        'ConfigError',
        'ProtocolError',
        'CaptureError',
        'AutofocusAborted',
        'MoveNotCompletedError',
        'HomingFailedError',
        'HardwareError',
    } <= {cls.__name__ for cls in UMBRELLAS}, 'the umbrella discovery drifted'
    constructed = []
    for umbrella in UMBRELLAS:
        # Refused before the umbrella reads its arguments, so any of the
        # right number will do.
        required = [
            name
            for name, param in inspect.signature(umbrella.__init__).parameters.items()
            if name != 'self'
            and param.default is param.empty
            and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
        ]
        try:
            umbrella(*('x' for _ in required))
        except TypeError as refused:
            if 'never raised' in str(refused):
                continue
        constructed.append(umbrella.__name__)
    assert constructed == [], f'these umbrellas can be raised themselves: {constructed}'


def _constructions(names):
    """``path:line`` of every call in production code to a class in ``names``, by its name."""
    paths = [REPO_ROOT / 'lumaviewpro.py']
    for layer in ('modules', 'drivers', 'rest', 'ui', 'lib', 'tools'):
        paths.extend(sorted((REPO_ROOT / layer).rglob('*.py')))
    found = []
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.Call) and ast.unparse(node.func).split('.')[-1] in names:
                found.append(f'{path.relative_to(REPO_ROOT)}:{node.lineno}')
    return found


def test_no_production_code_builds_an_umbrella():
    """An umbrella refuses construction only on the path that runs it: a
    raise of ``HardwareError`` in a driver's failure branch would reach its
    caller as a ``TypeError``, past every ``except HardwareError`` -- the
    capability probe at bring-up among them. So the build fails on the line."""
    assert _constructions({cls.__name__ for cls in UMBRELLAS}) == []


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
    from drivers.exceptions import CameraReadFailedError, LedFirmwareWedgedError
    from modules.notification_center import outcome_of

    # Written for the person, and now shown as written: the file and what is
    # wrong with it, where a generic sentence stood before.
    installation = outcome_of(InstallationFileError('/x/objectives.json', 'is missing'))
    assert installation.kind == 'fault'
    assert installation.for_person is True

    wedged = LedFirmwareWedgedError('LED firmware wedged. Power-cycle the unit.')
    assert outcome_of(wedged).for_person is True
    # A driver fault carrying the SDK's text: the person reads its title.
    read = CameraReadFailedError('BlackLevel read failed: RuntimeException: node not readable')
    assert outcome_of(read).for_person is False
    # A Python error's words are a developer's: the person reads the generic sentence.
    assert outcome_of(RuntimeError('list index out of range')).for_person is False


# ---------------------------------------------------------------------------
# A title on the type, and a reason, on every one (W0 (d1), R1)
# ---------------------------------------------------------------------------

RAISABLE = tuple(
    cls for cls in EXCEPTION_CLASSES if issubclass(cls, LumascopeError) and cls not in UMBRELLAS
)
_TABLES = ('_TITLES', 'causes', '_REASONS', '_WORDS')


def _ours(cls):
    """The classes in ``cls``'s chain that this codebase defines."""
    return [k for k in cls.__mro__ if k.__module__.split('.')[0] in ('modules', 'drivers', 'rest')]


def _init_sets_reason(cls):
    for k in _ours(cls):
        init = vars(k).get('__init__')
        if init is None:
            continue
        tree = ast.parse(textwrap.dedent(inspect.getsource(init)))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(
                    isinstance(t, ast.Attribute)
                    and t.attr == 'reason'
                    and ast.unparse(t.value) == 'self'
                    for t in targets
                ):
                    return True
            if (
                isinstance(node, ast.Call)
                and ast.unparse(node.func).endswith('__init__')
                and (any(kw.arg == 'reason' for kw in node.keywords) or len(node.args) >= 2)
            ):
                return True
    return False


def test_every_raisable_outcome_has_a_title_on_its_type():
    """A title is the class's, or its table's by reason: never chosen where it is raised."""
    untitled = [
        cls
        for cls in RAISABLE
        if not isinstance(getattr(cls, 'title', None), str)
        # A _WORDS table holds each reason's title beside its words.
        and not any(hasattr(cls, table) for table in ('_TITLES', '_WORDS'))
    ]
    assert untitled == [], (
        f'state a title in the class body, or a _TITLES table keyed by reason: {_named(untitled)}'
    )


def test_every_raisable_outcome_declares_its_reason():
    """A caller branches on the reason, so every class states one."""
    reasonless = [
        cls
        for cls in RAISABLE
        if not isinstance(getattr(cls, 'reason', None), str)
        and not any(hasattr(cls, table) for table in _TABLES)
        and not _init_sets_reason(cls)
    ]
    assert reasonless == [], (
        'state a reason in the class body, a reason-keyed table, or set it in __init__: '
        f'{_named(reasonless)}'
    )


def _owner(cls, attr):
    """The class along ``cls``'s MRO whose own body declares ``attr``, or None."""
    return next((k for k in cls.__mro__ if attr in vars(k)), None)


def test_every_title_and_reason_is_the_class_s_own():
    """A title or reason a class only inherits passes the two guards above on
    its parent's value, so a class that lost its own declaration answered with
    its parent's heading and code and every guard stayed green. Each is declared
    in the class's own body, set per instance (no class declares it), or read
    from the class's own reason-keyed table; a class that shares its parent's
    says so in one line (``title = Parent.title``)."""
    inherited = [
        f'{cls.__qualname__}.{attr} <- {_owner(cls, attr).__qualname__}'
        for cls in RAISABLE
        for attr in ('reason', 'title')
        if _owner(cls, attr) not in (None, cls)
        and not any(table in vars(cls) for table in ('_TITLES', '_WORDS'))
    ]
    assert inherited == [], f'declare it in the class, or read it from its own table: {inherited}'


def test_no_outcome_sets_a_title_per_instance():
    """``self.title`` is assigned only from a table, so a title cannot be chosen per raise."""
    from_a_table = ('self._TITLES[', 'self._WORDS[', 'ENDING_TITLES[', 'type(self)._TITLES[')
    per_instance = []

    def names_self_title(target):
        if isinstance(target, ast.Tuple):
            return any(names_self_title(element) for element in target.elts)
        return (
            isinstance(target, ast.Attribute)
            and target.attr == 'title'
            and ast.unparse(target.value) == 'self'
        )

    for cls in (*RAISABLE, *UMBRELLAS):
        init = vars(cls).get('__init__')
        if init is None:
            continue
        tree = ast.parse(textwrap.dedent(inspect.getsource(init)))
        for node in ast.walk(tree):
            # A tuple target too: ``self.title, message = self._WORDS[reason]``.
            sets_title = isinstance(node, ast.Assign) and any(
                names_self_title(t) for t in node.targets
            )
            if sets_title and not ast.unparse(node.value).startswith(from_a_table):
                per_instance.append(f'{cls.__qualname__}: {ast.unparse(node)}')
    assert per_instance == [], f'a title set per instance, not read from a table: {per_instance}'


def test_no_production_call_passes_a_title_to_an_outcome():
    names = {cls.__name__ for cls in (*RAISABLE, *UMBRELLAS)}
    passing = []
    for layer in ('modules', 'drivers', 'rest', 'ui'):
        for path in sorted((REPO_ROOT / layer).rglob('*.py')):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if (
                    isinstance(node, ast.Call)
                    and ast.unparse(node.func).split('.')[-1] in names
                    and any(kw.arg == 'title' for kw in node.keywords)
                ):
                    passing.append(f'{path.relative_to(REPO_ROOT)}:{node.lineno}')
    assert passing == [], f'a title passed where an outcome is built: {passing}'


_REASON_TABLES = ('causes', '_TITLES', '_WORDS', '_STATUSES', '_SENTENCES')


def test_every_reason_keyed_table_on_a_class_names_one_vocabulary():
    """Two reason-keyed tables on one class name one vocabulary: a reason one
    table has and another lacks fails where it is raised, a refusal turned crash
    (a reason with a cause and no status is a KeyError at the raise)."""
    apart = []
    for cls in RAISABLE:
        tables = {name: set(vars(cls)[name]) for name in _REASON_TABLES if name in vars(cls)}
        keys = list(tables.values())
        if any(table != keys[0] for table in keys[1:]):
            union = set().union(*keys)
            missing = {
                name: sorted(union - table) for name, table in tables.items() if union - table
            }
            apart.append(f'{cls.__qualname__}: {missing}')
    assert apart == [], f'each table lacks the reasons named: {apart}'
