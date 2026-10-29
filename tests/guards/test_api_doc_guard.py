"""Structural guard: LumascopeSkills.md's code is the API's, and it runs.

Every call form a reader could copy out of the guide must resolve on a real
object (polarity 1) AND accept the arguments the guide passes it (polarity
2), and every ``python`` fence must run on the simulator as its info string
says (polarity 3). Resolution alone is not enough: a member can exist and
still reject the documented call, which is how the guide came to publish
``fv.count_frame(chunk_data=None, frame_ts=None)`` against a signature that
had neither parameter. Binding is not enough either: a call can bind and
still fail on the shipped settings, as the guide's unlit still, its
``new_protocol(..., use_zstacking=True)`` with no stack configured, and its
moves on an unhomed scope did. When this test goes red the DOC is what gets
corrected -- the guide follows the API, never the reverse. Adding a method
because a doc line names it is the failure mode this guard exists to make
visible, not to license.

It replaces the per-defect string assertions that accumulated in
test_audit_fixes.py (one ``assert 'scope.pixel_size()' not in doc`` per
audit finding): those pinned individual known-bad names, where this checks
the whole document against the live object, so the next drift is caught
without anyone having to notice it first.

Running the fences
------------------
Each ``## `` chapter's ``python`` fences run in order in one namespace, on a
simulated session brought up and homed as a host leaves it, bound as
``session`` and ``scope`` (``_ChapterSessions``). The info string says what
a fence needs beyond that: ``tier=`` and ``model=`` name the board it runs
on, ``raises=<Class>`` the refusal it teaches, raised by its last statement,
and ``norun`` a fence no simulator can run (serving forever, say), which is
still resolved and bound.

The ``bash`` fences are the REST walkthrough (polarity 4): each ``curl`` line
is sent, in order, to ``TestClient(build_app(session))`` on such a session,
and the status and reason its comment states are asserted. A handle's or a
job's id is read from an earlier answer (``NAME=$(curl ... | jq -r .id)``),
never typed: ids count every call held as a job, so a typed one names another
call's job as soon as the server's history differs.

Prose, checked by kind
----------------------
A name the prose states is checked by its kind, each kind at its owner
(polarity 5): a ``session.`` / ``scope.`` / ``caps.`` form or a
``Class.member`` of a class the API names resolves to API; a route
(``/api/...``, with or without its method) is answered by the app's routes; a
dotted path whose first part is a setting is one (``get_setting``); and a
quoted reason (``'not_offered'``) is one a published outcome, the server or a
run's ending carries. Every other backticked token is not checked. The
Changelog names, by design, surfaces that do NOT resolve -- its ``removal``
and ``rename`` entries -- so its section is left out by its heading, a rule
rather than a list of exceptions to keep in step.

A backticked path into a project package (```modules.x.y```, ``drivers.``,
``rest.``) names something a reader is told to import, so it must be API like
a fenced import.

A link from the guide reaches its anchor (polarity 6): one of its own
headings, or an anchor the generated reference holds -- a member, field or
class the generator rendered, or one of its headings.

``PLUGIN_API_DESIGN_2026-05-09.md`` is deliberately NOT guarded. It is a
design and history document; 12 of its 40 call forms name proposals that were
never built (``scope.is_command_safe_for_rest``,
``session.focus.get_well_focus``) or pre-rename historical spellings. Its
accuracy is maintained by review, not by resolution.
"""

import ast
import collections
import dataclasses
import enum
import importlib
import inspect
import json
import re
import shlex
import traceback
import urllib.parse
import warnings

import pytest

from lib.lumascope_error import LumascopeError
from tests.ast_seams import REPO_ROOT, MemberNames, Packages
from tests.settings_fixtures import complete_settings

DOC = REPO_ROOT / 'docs' / 'LumascopeSkills.md'

# Receivers the reference uses for the live objects an L2 caller holds.
# `scope` is the Lumascope composition root, `session` the ScopeSession L2
# entry point. Both are resolved against real instances rather than classes:
# `session.scope` is an instance attribute, so a class-level hasattr would
# report every `session.scope.*` form in the document as unresolved.
#
# `caps` is an alias the guide establishes in its own example block
# (`caps = scope.capabilities`, in "What LumaViewPro is") and then reads the
# capabilities through. Without it those lines would sit outside the guard.
_CALL_FORM = re.compile(r'\b(scope|session|caps)((?:\.[A-Za-z_][A-Za-z0-9_]*)+)')

# Fence languages whose contents are checked. The empty string is an
# unlabelled ``` block; the reference uses those for python too, and they
# resolve today, so including them costs nothing and closes the gap where a
# forgotten language tag would silence the guard. A fence labelled for another
# language is skipped -- its syntax is not ours to resolve -- which is how the
# reference marks its diagrams and file layouts: ```text.
_CHECKED_FENCES = ('', 'python')


def fence_info(line):
    """``(language, options)`` of a fence's opening line.

    The info string is the language, then words the runner reads from a
    ``python`` fence: ``key=value`` (``tier=firmware``, ``model=LS850T``,
    ``raises=AxisStateUnknownError``) or a bare flag (``norun``), whose
    value is True.
    """
    words = line[3:].split()
    options = {}
    for word in words[1:]:
        key, sep, value = word.partition('=')
        options[key] = value if sep else True
    return (words[0] if words else ''), options


def _fenced_call_forms(text):
    """Extract (receiver, chain, lineno) for every call form in a code fence.

    Yields one entry per occurrence, not per distinct form, so the failure
    message can name every line a reader would have copied from.
    """
    fence = None
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.startswith('```'):
            fence = fence_info(line)[0] if fence is None else None
            continue
        if fence not in _CHECKED_FENCES:
            continue
        for match in _CALL_FORM.finditer(line):
            chain = tuple(match.group(2).lstrip('.').split('.'))
            yield match.group(1), chain, lineno


def _checked_fence_blocks(text):
    """(first_lineno, source) for each checked fence, as one parsable unit.

    A line is the wrong unit for reading arguments. It cannot see a call whose
    arguments wrap, and it rejects the assignments that establish the
    document's own aliases -- `scope = Lumascope(simulate=True)` is a
    statement, not an expression. The fence is the smallest unit that holds
    both, and it is also what a reader copies.
    """
    fence = None
    lines, start = [], 0
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.startswith('```'):
            if fence is None:
                fence, lines, start = fence_info(line)[0], [], lineno + 1
            else:
                if fence in _CHECKED_FENCES:
                    yield start, '\n'.join(lines)
                fence = None
            continue
        if fence is not None:
            lines.append(line)


def _documented_calls(text):
    """(receiver, chain, positional_count, keyword_names, lineno) per call.

    The argument SHAPE, never the values: a doc example's arguments are
    illustrative, while the shape is the part the live signature has to
    accept. A form spreading `**kwargs` is skipped -- the reference does not
    name what is in it, so there is nothing to bind.

    A fence that does not parse is skipped here; polarity 1 still reads it
    line by line, so no form is invisible to both checks.
    """
    for start, source in _checked_fence_blocks(text):
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func, chain = node.func, []
            while isinstance(func, ast.Attribute):
                chain.append(func.attr)
                func = func.value
            if not isinstance(func, ast.Name):
                continue
            if any(kw.arg is None for kw in node.keywords):
                continue
            chain.reverse()
            yield (
                func.id,
                tuple(chain),
                len(node.args),
                tuple(kw.arg for kw in node.keywords),
                start + node.lineno - 1,
            )


@pytest.fixture(scope='module')
def doc_text():
    """The published reference, read once for the module.

    pin-justified: the guard's subject IS the document text, so there is no
    AST seam to assert instead -- the doc-example case the source-pin ratchet
    names as justified. One read site serves both checks.
    """
    return DOC.read_text(encoding='utf-8')


@pytest.fixture(scope='module')
def live_objects(tmp_path_factory):
    """Every receiver the reference documents, as live instances.

    Module-scoped: the headless session builds the full executor topology,
    which is ~0.5s and pointless to repeat per test.
    """
    warnings.simplefilter('ignore', FutureWarning)
    from modules.scope_session import ScopeSession

    # The reference's `scope` is a session's: a scope reads the labware,
    # offset, turret map, objective and scale bar from its session's settings.
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path_factory.mktemp('live'))), simulate=True
    )
    scope = session.scope
    yield {
        'scope': scope,
        'session': session,
        'caps': scope.capabilities,
    }
    session.shutdown()


def _walk_chain(root, chain):
    """(resolved_object, failed_depth). depth is None when the whole chain resolved.

    One traversal serves both polarities: polarity 1 needs the name of the
    prefix that broke, polarity 2 needs the object at the end of it.
    """
    obj = root
    for depth, attr in enumerate(chain):
        if not hasattr(obj, attr):
            return None, depth
        obj = getattr(obj, attr)
    return obj, None


def _first_unmarked_step(root, receiver, chain):
    """The first ``Class.member`` the chain reaches that is not marked, or None."""
    from modules.api_surface import fields_of, mark_of

    obj = root
    for attr in chain:
        owner = next((k for k in type(obj).__mro__ if attr in vars(k)), None)
        if owner is None and attr in getattr(obj, '__dict__', {}):
            owner = type(obj)
        if owner is None or not owner.__module__.startswith('modules'):
            return None
        marked = mark_of(vars(owner).get(attr)) or any(
            attr in fields_of(k) for k in type(obj).__mro__
        )
        if not marked:
            return f'{owner.__name__}.{attr}'
        if not hasattr(obj, attr):
            return None
        obj = getattr(obj, attr)
    return None


def _first_missing_attr(root, receiver, chain):
    """The shortest prefix of the chain that does not resolve, or None."""
    _, depth = _walk_chain(root, chain)
    if depth is None:
        return None
    return '.'.join((receiver, *chain[: depth + 1]))


class TestLumascopeSkillsCallFormsResolve:
    """Polarity 1: documented call forms exist on the live API surface."""

    def test_every_fenced_call_form_resolves(self, live_objects, doc_text):
        failures = []
        for receiver, chain, lineno in _fenced_call_forms(doc_text):
            missing = _first_missing_attr(live_objects[receiver], receiver, chain)
            if missing:
                failures.append(
                    f'  {DOC}:{lineno}  {receiver}.{".".join(chain)}  -> no attribute {missing}'
                )

        assert not failures, (
            'LumascopeSkills.md documents call forms that do not resolve on the '
            'live API. An L2 reader copying these lines gets AttributeError.\n'
            'Correct the DOCUMENT to match the surface -- do not add a member to '
            'satisfy a doc line.\n' + '\n'.join(sorted(set(failures)))
        )

    def test_every_fenced_call_form_is_api(self, live_objects, doc_text):
        """Each step of a copyable call form is a marked member.

        Resolving is not enough: an unmarked member resolves too, and a reader
        who copies it builds on something that can change in any release and
        that no wire carries. A step on an object no project class defines --
        a dict, a number, a library type -- ends the walk: what it holds is
        not ours to mark.
        """
        unmarked = []
        for receiver, chain, lineno in _fenced_call_forms(doc_text):
            step = _first_unmarked_step(live_objects[receiver], receiver, chain)
            if step:
                unmarked.append(f'  {DOC}:{lineno}  {receiver}.{".".join(chain)}  -> {step}')
        assert not unmarked, (
            'LumascopeSkills.md teaches call forms through members that are not API '
            '(no @api, not in @api_fields). Rewrite the line to the marked door, or '
            'drop it.\n' + '\n'.join(sorted(set(unmarked)))
        )

    def test_guard_actually_reaches_the_reference(self, doc_text):
        """The extractor is wired to real content, not silently finding nothing.

        A regex or fence-tracking regression would make the check above pass
        vacuously. Every documented receiver is exercised in the reference
        today, so a run that misses one means the extractor broke, not that
        the doc got smaller.
        """
        receivers = {r for r, _, _ in _fenced_call_forms(doc_text)}
        assert receivers == {'scope', 'session', 'caps'}, (
            f'expected call forms for every documented receiver, saw {receivers or "none"} -- '
            'the fence tracker or the call-form pattern has regressed'
        )


# The receiver polarity 2 adds on top of the aliases. An L2 script names this
# class before any alias exists -- `ScopeSession.create(...)` is the first call
# a consumer makes -- so it carries the arguments most likely to be copied and
# the least likely to be re-read. Being a class rather than an instance, it is
# invisible to the alias-rooted attribute check.
_CONSTRUCTOR_RECEIVERS = MemberNames('modules/scope_session.py', {'ScopeSession'})

# Stands in for an argument value. bind() checks arity and parameter names; it
# never looks at what is passed, and a doc example's values are illustrative.
_SENTINEL = object()


@pytest.fixture(scope='module')
def bind_roots(live_objects):
    """Every receiver polarity 2 resolves against: the aliases plus the class."""
    from modules.scope_session import ScopeSession

    return {**live_objects, 'ScopeSession': ScopeSession}


class TestLumascopeSkillsCallFormsBind:
    """Polarity 2: documented call forms accept the arguments the doc passes them.

    Polarity 1 asks whether a member exists. A member can exist and still
    reject the documented call, and that gap is not hypothetical: the
    reference shipped `fv.count_frame(chunk_data=None, frame_ts=None)` after
    both parameters were gone and the single remaining one became required, so
    a reader who copied the line got a TypeError while the guard reported the
    document clean.

    Shape only -- arity and keyword names. A parameter RENAME under a
    positional call form still binds, and most documented forms are positional
    or take no arguments at all, so this closes the arity and keyword half of
    the drift, not the naming half.
    """

    def test_every_documented_call_binds(self, bind_roots, doc_text):
        failures = []
        for receiver, chain, nargs, kwnames, lineno in _documented_calls(doc_text):
            root = bind_roots.get(receiver)
            if root is None:
                continue
            target, _ = _walk_chain(root, chain)
            if target is None or not callable(target):
                continue
            form = '.'.join((receiver, *chain))
            try:
                inspect.signature(target).bind(
                    *(_SENTINEL,) * nargs, **dict.fromkeys(kwnames, _SENTINEL)
                )
            except TypeError as exc:
                failures.append(f'  {DOC}:{lineno}  {form}(...)  -> {exc}')

        assert not failures, (
            'LumascopeSkills.md documents calls the live signature will not accept. '
            'An L2 reader copying these lines gets TypeError.\n'
            'Correct the DOCUMENT to match the signature -- do not add a parameter '
            'to satisfy a doc line.\n' + '\n'.join(sorted(set(failures)))
        )

    def test_bind_check_reaches_the_receivers_that_carry_arguments(self, doc_text):
        """The extractor is wired to real content, not silently finding nothing.

        A fence-tracking or parse regression would make the check above pass
        vacuously. Asserted as a set of receivers rather than a count of forms:
        the count moves whenever anyone adds an example, and a test that goes
        red for that gets bumped blindly instead of read.

        `caps` is deliberately absent. The reference uses it almost entirely
        for property reads, so requiring a CALL form there would pin a wording
        choice rather than a contract.
        """
        receivers = {r for r, _, _, _, _ in _documented_calls(doc_text)}
        expected = {'scope', 'session', *_CONSTRUCTOR_RECEIVERS}
        assert expected <= receivers, (
            f'no documented calls found for {sorted(expected - receivers)} -- the fence '
            'tracker, the block parser or the call extractor has regressed'
        )


# The project's packages. A name a reader imports from one of them, or a path
# into one written in prose, is something the reader is told to build on, so
# it must be API -- the receivers above cover only what a reader reaches from
# a live object, and an internal helper's signature change broke eight
# examples in its module-level calls with this file green.
_PROJECT_PACKAGES = Packages({'modules', 'drivers', 'rest'})
_PROSE_PATH = re.compile(r'`((?:modules|drivers|rest)\.[A-Za-z_][A-Za-z0-9_.]*)')


def why_not_api(obj):
    """Why ``obj``, a name from a project package, is not API; None when it is.

    API is an exception, an enum or one of its members, a class that
    publishes fields or has marked members, or a marked function. A module
    is where names live, not a name: each of its names a reader uses is
    checked where it is used.
    """
    from modules.api_surface import fields_of, mark_of

    if inspect.ismodule(obj) or isinstance(obj, enum.Enum):
        return None
    if inspect.isclass(obj):
        if issubclass(obj, (BaseException, enum.Enum)):
            return None
        if any(fields_of(k) for k in obj.__mro__):
            return None
        if any(mark_of(v) for k in obj.__mro__ for v in vars(k).values()):
            return None
        return 'a class with no marked members or published fields'
    if callable(obj):
        return None if mark_of(obj) else 'an unmarked function'
    return f'a {type(obj).__name__} value, not a member'


def _resolve_project_name(dotted):
    """The object ``dotted`` names: its longest importable module, then attributes."""
    parts = dotted.split('.')
    for cut in range(len(parts), 0, -1):
        try:
            found = importlib.import_module('.'.join(parts[:cut]))
        except ImportError:
            continue
        for attr in parts[cut:]:
            found = getattr(found, attr)
        return found
    raise ImportError(dotted)


def _why_name_is_not_api(dotted):
    """Why the project name ``dotted`` is not API; None when it is.

    A field its class publishes is API whatever its value, so it is judged
    by the publishing, not by what it holds.
    """
    from modules.api_surface import fields_of

    owner_path, _, attr = dotted.rpartition('.')
    if owner_path:
        try:
            owner = _resolve_project_name(owner_path)
        except (ImportError, AttributeError):
            owner = None
        if inspect.isclass(owner) and any(attr in fields_of(k) for k in owner.__mro__):
            return None
    return why_not_api(_resolve_project_name(dotted))


def _project_module(name):
    """``'<package>.<name>'`` for the first project package with a module ``name``, or None."""
    return next(
        (f'{p}.{name}' for p in _PROJECT_PACKAGES if importlib.util.find_spec(f'{p}.{name}')),
        None,
    )


def project_names(source, start=1, assigned=frozenset()):
    """``(dotted, positional_count, keyword_names, lineno)`` per project name ``source`` uses.

    A name imported from a project package, and each attribute read on a
    project module -- one imported in ``source``, or one named bare and never
    assigned in ``assigned`` (a later fence uses a module an earlier one
    imported). The counts are the call's when the name is called, else None.
    """
    tree = ast.parse(source)
    assigned = assigned | _assigned_names(tree)
    bound, imported = {}, []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split('.')[0] in _PROJECT_PACKAGES:
                    bound[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom) and (node.module or '').split('.')[0] in (
            _PROJECT_PACKAGES
        ):
            for alias in node.names:
                bound[alias.asname or alias.name] = f'{node.module}.{alias.name}'
                imported.append((f'{node.module}.{alias.name}', start + node.lineno - 1))

    def dotted_of(func):
        if isinstance(func, ast.Name):
            return bound.get(func.id)
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            owner = bound.get(func.value.id)
            if owner is None and func.value.id not in assigned:
                owner = _project_module(func.value.id)
            if owner is not None:
                return f'{owner}.{func.attr}'
        return None

    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and not any(kw.arg is None for kw in node.keywords):
            dotted = dotted_of(node.func)
            if dotted is not None:
                called.add(id(node.func))
                kwnames = tuple(kw.arg for kw in node.keywords)
                yield dotted, len(node.args), kwnames, start + node.lineno - 1
    for dotted, lineno in imported:
        yield dotted, None, None, lineno
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and id(node) not in called:
            dotted = dotted_of(node)
            if dotted is not None:
                yield dotted, None, None, start + node.lineno - 1


def _assigned_names(tree):
    """Every name ``tree`` assigns or takes as a parameter."""
    return frozenset(
        node.id if isinstance(node, ast.Name) else node.arg
        for node in ast.walk(tree)
        if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store))
        or isinstance(node, ast.arg)
    )


def _assigned_in(text):
    """Every name a checked fence of ``text`` assigns or takes as a parameter."""
    names = frozenset()
    for _, source in _checked_fence_blocks(text):
        try:
            names |= _assigned_names(ast.parse(source))
        except SyntaxError:
            continue
    return names


def project_name_failures(label, source, start=1, assigned=frozenset()):
    """A line per project name in ``source`` that is not API, or whose call does not bind."""
    failures = []
    for dotted, nargs, kwnames, lineno in project_names(source, start, assigned):
        try:
            found = _resolve_project_name(dotted)
            why = _why_name_is_not_api(dotted)
        except (ImportError, AttributeError):
            failures.append(f'  {label}:{lineno}  {dotted}  -> does not resolve')
            continue
        if why:
            failures.append(f'  {label}:{lineno}  {dotted}  -> {why}')
        elif nargs is not None and callable(found):
            try:
                inspect.signature(found).bind(
                    *(_SENTINEL,) * nargs, **dict.fromkeys(kwnames, _SENTINEL)
                )
            except TypeError as exc:
                failures.append(f'  {label}:{lineno}  {dotted}(...)  -> {exc}')
    return failures


def _doc_project_name_failures(text, label):
    assigned = _assigned_in(text)
    failures = []
    for start, source in _checked_fence_blocks(text):
        try:
            failures += project_name_failures(label, source, start, assigned)
        except SyntaxError as exc:
            failures.append(
                f'  {label}:{start}  the fence does not parse ({exc.msg}): label it text'
            )
    fence = None
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.startswith('```'):
            fence = fence_info(line)[0] if fence is None else None
            continue
        if fence is not None:
            continue
        for match in _PROSE_PATH.finditer(line):
            dotted = match.group(1).rstrip('.')
            try:
                why = _why_name_is_not_api(dotted)
            except (ImportError, AttributeError):
                why = 'does not resolve'
            if why:
                failures.append(f'  {label}:{lineno}  {dotted}  -> {why}')
    return failures


class TestLumascopeSkillsTeachesOnlyApi:
    """What the reference has a reader import from the project, or names by path, is API."""

    def test_every_project_name_the_reference_teaches_is_api(self, doc_text):
        failures = _doc_project_name_failures(doc_text, DOC)
        assert not failures, (
            'LumascopeSkills.md teaches a project name that is not API (an exception, an '
            'enum, a class with marked members or published fields, or a marked function), '
            'or calls one with arguments it will not take. Teach the member that does the '
            'job, or say in words what the API does.\n' + '\n'.join(failures)
        )

    def test_a_planted_helper_import_call_and_path_are_reported(self):
        planted = (
            '```python\n'
            'from modules.image_save import save_image\n'
            'from modules import settings_init\n'
            'save_image(None)\n'
            'settings_init.current_json_path(".")\n'
            'from modules.scope_session import ScopeSession\n'
            'ScopeSession.user_settings_file(".", "extra")\n'
            '```\n'
            '\n'
            'Prose naming `modules.post_processing.with_pixels_per_um`.\n'
            '\n'
            '```\n'
            'not python at all (\n'
            '```\n'
        )
        reported = '\n'.join(_doc_project_name_failures(planted, 'planted'))
        for expected in (
            'planted:2  modules.image_save.save_image  -> an unmarked function',
            'planted:5  modules.settings_init.current_json_path  -> an unmarked function',
            'planted:7  modules.scope_session.ScopeSession.user_settings_file(...)',
            'planted:10  modules.post_processing.with_pixels_per_um',
            'planted:13  the fence does not parse',
        ):
            assert expected in reported, reported


def test_the_reference_names_each_protocol_constructor_and_each_is_api():
    """A presence claim the call-form check does not make: the reference
    teaches the three ways to get a protocol, and each is a marked member."""
    from modules.api_surface import API, mark_of
    from modules.scope_session import ScopeSession

    doc = DOC.read_text(encoding='utf-8')
    for member in ('load_protocol', 'new_protocol', 'create_empty_protocol'):
        assert f'session.{member}(' in doc, member
        assert mark_of(getattr(ScopeSession, member)) == API, member


# The tier a fence runs on when it names none: the simulator a reader's
# script gets from the shipped settings is the realistic one, but every
# fence that does not depend on timing reads the same on the fast tier, at a
# tenth of the cost.
_DEFAULT_TIER = 'fast'


def guide_chapters(text):
    """``(heading, [(start, language, options, source), ...])`` per ``## `` chapter.

    ``start`` is the line of the fence's first source line. Text before the
    first chapter heading is a chapter with the heading ``''``.
    """
    heading, fences = '', []
    fence, lines, start = None, [], 0
    for lineno, line in enumerate(text.splitlines(), 1):
        if fence is None and line.startswith('## '):
            yield heading, fences
            heading, fences = line[3:].strip(), []
            continue
        if line.startswith('```'):
            if fence is None:
                fence, lines, start = fence_info(line), [], lineno + 1
            else:
                fences.append((start, *fence, '\n'.join(lines)))
                fence = None
            continue
        if fence is not None:
            lines.append(line)
    yield heading, fences


class _ChapterSessions:
    """The simulated sessions one chapter's fences run against, shut down together.

    A chapter starts on the default tier and model, brought up and homed as a
    host leaves it; a fence naming another tier or model swaps the session
    for one on that board, and later fences keep it. A session a fence makes
    for itself is shut down with the chapter's.
    """

    def __init__(self, namespace):
        self._namespace = namespace
        self._key = None
        self._made = []

    def use(self, options):
        """Bring up the session a fence's ``options`` ask for, unless it is up."""
        from modules import settings_init
        from modules.scope_session import ScopeSession

        if self._key is not None and 'tier' not in options and 'model' not in options:
            return
        tier, model = options.get('tier', _DEFAULT_TIER), options.get('model')
        if (tier, model) == self._key:
            return
        overrides = {'simulator_tier': tier}
        if model is not None:
            overrides['microscope'] = model
        settings = complete_settings(**overrides)
        # A fence teaching load_user_settings reads what this process
        # prepared, as the GUI's own scripts do: the chapter's settings,
        # never this machine's current.json.
        settings_init.settings = settings
        session = ScopeSession.create(settings.copy(), simulate=True, warn_pre_release=False)
        self._made.append(session)
        session.run_application_session()
        self._namespace.update(session=session, scope=session.scope)
        self.session = session
        self._key = (tier, model)

    def close(self):
        from modules.scope_session import ScopeSession

        made = self._made + [v for v in self._namespace.values() if isinstance(v, ScopeSession)]
        for session in {id(s): s for s in made}.values():
            session.shutdown()


def _last_statement_lines(source):
    """The line span of ``source``'s last top-level statement, 1-based."""
    last = ast.parse(source).body[-1]
    return last.lineno, last.end_lineno


def _raised_at(exc, filename):
    """The line of ``filename`` the exception left, the innermost frame there."""
    lines = [
        lineno
        for frame, lineno in traceback.walk_tb(exc.__traceback__)
        if frame.f_code.co_filename == filename
    ]
    return lines[-1] if lines else None


def _run_fence(label, start, options, source, namespace):
    """A failure line for a fence that does not do what its info string says, or None."""
    filename = f'{label}:{start}'
    expected = options.get('raises')
    code = compile('\n' * (start - 1) + source, filename, 'exec')
    try:
        exec(code, namespace)
    except (Exception, LumascopeError) as exc:
        raised = type(exc).__name__
        if expected is None:
            return f'  {label}:{start}  raised {raised}: {exc}'
        if raised != expected:
            return f'  {label}:{start}  raised {raised}, not the {expected} it states: {exc}'
        first, last = _last_statement_lines(source)
        at = _raised_at(exc, filename)
        if at is None or not start + first - 1 <= at <= start + last - 1:
            return f'  {label}:{start}  raised {expected} at line {at}, before its last statement'
        return None
    if expected is not None:
        return f'  {label}:{start}  states raises={expected} and raised nothing'
    return None


def guide_fence_failures(text, label):
    """A line per ``python`` fence of ``text`` that does not run as its info string states.

    Each chapter's fences run in order in one namespace holding the chapter's
    ``session`` and ``scope`` (``_ChapterSessions``); a fence flagged
    ``norun`` is skipped here and still resolved and bound by the checks
    above. A fence that raises ends the chapter: what follows it assumed it
    ran.
    """
    from modules import settings_init

    failures = []
    prepared = settings_init.settings
    try:
        for heading, fences in guide_chapters(text):
            runnable = [f for f in fences if f[1] == 'python' and not f[2].get('norun')]
            if not runnable:
                continue
            namespace = {'__name__': '__guide__'}
            sessions = _ChapterSessions(namespace)
            try:
                for start, _, options, source in runnable:
                    sessions.use(options)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', FutureWarning)
                        failure = _run_fence(label, start, options, source, namespace)
                    if failure:
                        failures.append(f'{failure}  (chapter "{heading}")')
                        break
            finally:
                sessions.close()
    finally:
        settings_init.settings = prepared
    return failures


class TestLumascopeSkillsFencesRun:
    """Polarity 3: every ``python`` fence runs on the simulator as its info string states."""

    @pytest.mark.slow
    def test_every_python_fence_runs(self, doc_text):
        failures = guide_fence_failures(doc_text, DOC)
        assert not failures, (
            'LumascopeSkills.md has python fences that do not run on the simulator as '
            'written. A fence that teaches a refusal states it (```python '
            'raises=<Class>) and ends in it; one that needs a board says tier= or '
            'model=; one no simulator can run is ```python norun.\n' + '\n'.join(failures)
        )

    def test_a_fence_that_does_not_do_what_it_states_is_reported(self):
        planted = (
            '## Lit\n'
            '```python\n'
            "still = session.capture_still('BF')\n"
            'assert still.record.dark is False\n'
            '```\n'
            '## Refused\n'
            '```python raises=PositionOutOfRangeError\n'
            'scope.motion.move_z(1e9)\n'
            'x = 1\n'
            '```\n'
            '## Wrong class\n'
            '```python raises=ValueError\n'
            "session.get_setting('nope')\n"
            '```\n'
            '## Never raised\n'
            '```python raises=ValueError\n'
            'x = 1\n'
            '```\n'
            '## Kept\n'
            '```python model=LS720\n'
            'x = 1\n'
            '```\n'
            '```python\n'
            "assert session.model == 'LS720'\n"
            '```\n'
            '## Skipped\n'
            '```python norun\n'
            'raise RuntimeError\n'
            '```\n'
        )
        reported = '\n'.join(guide_fence_failures(planted, 'planted'))
        for expected in (
            'planted:3  raised AssertionError',
            'planted:8  raised PositionOutOfRangeError at line 8, before its last statement',
            'planted:13  raised ArgumentRefusedError, not the ValueError it states',
            'planted:17  states raises=ValueError and raised nothing',
        ):
            assert expected in reported, reported
        assert 'planted:24' not in reported, reported
        assert 'planted:28' not in reported, reported


# A walkthrough line's trailing comment: the status it answers, then for a
# problem the reason it carries.
_STATED = re.compile(r'(\d{3})\b:?\s*([a-z][a-z0-9_]*)?')
# An id typed into a URL where it should be read from the answer that made it.
_LITERAL_ID = re.compile(r'/(?:jobs|handles/[A-Za-z]+)/\d+\b')
_CAPTURE = re.compile(r'^([A-Z_][A-Z0-9_]*)=\$\((.*)\| *jq -r (\S+)\)$')
_ASSIGN = re.compile(r'^([A-Z_][A-Z0-9_]*)=(\S+)$')


def _code_and_comment(line):
    """A shell line split at its comment: a ``#`` after a space, outside quotes."""
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
            elif ch == '\\' and quote == '"':
                continue
        elif ch in '\'"':
            quote = ch
        elif ch == '#' and (i == 0 or line[i - 1].isspace()):
            return line[:i].strip(), line[i + 1 :].strip()
    return line.strip(), ''


def _expand(token, variables):
    return re.sub(
        r'\$\{?([A-Z_][A-Z0-9_]*)\}?', lambda m: variables.get(m.group(1), m.group(0)), token
    )


def _curl_request(command, variables):
    """``(method, path, headers, body, output)`` of a ``curl`` command, its variables expanded.

    Reads the flags the guide and the examples use: ``-s``, ``-X``, ``-H``,
    ``--json``, ``-o`` and ``-w`` (what curl prints after the answer, which
    sends nothing); any other is refused, so a line the runner cannot send
    is never read as sent.
    """
    words = shlex.split(command)
    if words[0] != 'curl':
        raise ValueError(f'not a curl command: {words[0]}')
    method, headers, body, url, output = None, {}, None, None, None
    rest = iter(words[1:])
    for word in rest:
        if word == '-s':
            continue
        if word == '-w':
            next(rest)
        elif word == '-X':
            method = next(rest)
        elif word == '-o':
            output = _expand(next(rest), variables)
        elif word == '-H':
            name, _, value = _expand(next(rest), variables).partition(':')
            headers[name.strip()] = value.strip()
        elif word == '--json':
            body = _expand(next(rest), variables)
            headers.setdefault('Content-Type', 'application/json')
            headers.setdefault('Accept', 'application/json')
        elif word.startswith('-'):
            raise ValueError(f'the runner does not read {word}')
        else:
            url = urllib.parse.urlsplit(_expand(word, variables))
    path = url.path + (f'?{url.query}' if url.query else '')
    return method or ('POST' if body is not None else 'GET'), path, headers, body, output


def _jq_raw(answer, path):
    """``jq -r <path>`` of a JSON answer, for the paths the guide and the
    examples use (``.``, ``.a.b``, ``'.a[0].b'``)."""
    for key, index in re.findall(r'\.([A-Za-z_]\w*)|\[(\d+)\]', shlex.split(path)[0]):
        answer = answer[key] if key else answer[int(index)]
    return answer if isinstance(answer, str) else json.dumps(answer)


@dataclasses.dataclass
class ShellRun:
    """What a run of shell lines left behind, as a shell would have it."""

    variables: dict = dataclasses.field(default_factory=dict)
    # Each answer ``-o`` saved, by the name its line gave.
    written: dict = dataclasses.field(default_factory=dict)


def bash_fence_lines(text):
    """``(lineno, line)`` for each line inside ``text``'s ``bash`` fences."""
    fence = None
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.startswith('```'):
            fence = fence_info(line)[0] if fence is None else None
        elif fence == 'bash':
            yield lineno, line


def walkthrough_failures(text, label, client):
    """A line per ``curl`` line of ``text``'s ``bash`` fences that does not answer as stated."""
    return curl_failures(bash_fence_lines(text), label, client, ShellRun())


def curl_failures(lines, label, client, run):
    """A line per ``curl`` line of ``lines`` that does not answer as stated.

    The lines run in order, as one shell, into ``run``: ``NAME=value`` sets a
    variable, ``NAME=$(curl ... | jq -r <path>)`` sets it from the answer,
    and ``-o <name>`` keeps the answer under that name. A comment's status,
    and a problem's reason, are asserted; a line that states no status must
    succeed. A line that fails ends the run: what follows it assumed it ran.
    """
    variables = run.variables
    for lineno, line in lines:
        code, comment = _code_and_comment(line)
        assign = _ASSIGN.match(code)
        if assign and not code.startswith(f'{assign.group(1)}=$('):
            variables[assign.group(1)] = assign.group(2)
            continue
        capture = _CAPTURE.match(code)
        command = capture.group(2).strip() if capture else code
        if not command.startswith('curl '):
            continue
        if _LITERAL_ID.search(command):
            return [
                f'  {label}:{lineno}  an id typed into the URL: read it from the answer that made it'
            ]
        method, path, headers, body, output = _curl_request(command, variables)
        answer = client.request(method, path, headers=headers, content=body)
        stated = _STATED.match(comment)
        status = int(stated.group(1)) if stated else None
        reason = stated.group(2) if stated and status >= 400 else None
        if status is None and answer.status_code >= 300:
            return [f'  {label}:{lineno}  answered {answer.status_code}: {answer.text[:200]}']
        if status is not None and answer.status_code != status:
            return [
                f'  {label}:{lineno}  answered {answer.status_code}, not the {status} it states: '
                f'{answer.text[:200]}'
            ]
        if reason is not None and answer.json().get('reason') != reason:
            return [
                f'  {label}:{lineno}  answered reason {answer.json().get("reason")!r}, '
                f'not the {reason} it states'
            ]
        if capture:
            variables[capture.group(1)] = _jq_raw(answer.json(), capture.group(3))
        if output is not None:
            run.written[output] = answer.content
    return []


def _walk(text, label):
    from fastapi.testclient import TestClient

    from modules import settings_init
    from rest.app import build_app

    prepared = settings_init.settings
    sessions = _ChapterSessions({})
    try:
        sessions.use({})
        with TestClient(build_app(sessions.session)) as client:
            return walkthrough_failures(text, label, client)
    finally:
        sessions.close()
        settings_init.settings = prepared


class TestLumascopeSkillsWalkthroughRuns:
    """Polarity 4: the REST walkthrough's ``curl`` lines answer what their comments state."""

    @pytest.mark.slow
    def test_every_curl_line_answers_as_stated(self, doc_text):
        failures = _walk(doc_text, DOC)
        assert not failures, (
            'LumascopeSkills.md has a curl line that does not answer as its comment '
            'states, on a simulated server. Correct the line or its comment.\n'
            + '\n'.join(failures)
        )

    def test_a_line_that_does_not_answer_as_stated_is_reported(self):
        cases = {
            'a typed job id': '```bash\nAPI=http://h/api/v1\ncurl -s $API/jobs/1\n```\n',
            'a wrong status': (
                '```bash\nAPI=http://h/api/v1\ncurl -s "$API/scope/motion/get_axis_limits?axis=W"'
                '   # 200\n```\n'
            ),
            'a wrong reason': (
                '```bash\nAPI=http://h/api/v1\ncurl -s "$API/scope/motion/get_axis_limits?axis=W"'
                '   # 422 axis_absent\n```\n'
            ),
            'an unstated failure': '```bash\nAPI=http://h/api/v1\ncurl -s $API/no_such_member\n```\n',
        }
        reported = {name: _walk(text, 'planted') for name, text in cases.items()}
        assert reported['a typed job id'] == [
            '  planted:3  an id typed into the URL: read it from the answer that made it'
        ]
        assert 'answered 422, not the 200 it states' in reported['a wrong status'][0]
        assert "answered reason 'not_offered', not the axis_absent" in reported['a wrong reason'][0]
        assert 'answered 404' in reported['an unstated failure'][0]


# The names the guide's prose is held to, each kind by its owner; every other
# backticked token is not checked. The Changelog names what no longer
# resolves, by design, so its section is left out by its heading.
_RECEIVER_FORM = re.compile(r'^(scope|session|caps)((?:\.[A-Za-z_]\w*)+)(?:\(.*\))?$')
_CLASS_MEMBER = re.compile(r'^([A-Z]\w*)\.([A-Za-z_]\w*)(?:\(.*\))?$')
_ROUTE = re.compile(r'^(?:(GET|POST|PUT|DELETE) )?(/api(?:/[^\s?]*)?)(?:\?\S*)?$')
_DOTTED = re.compile(r'^([A-Za-z_]\w*)(?:\.[A-Za-z_]\w*)+$')
_QUOTED_REASON = re.compile(r"^'([a-z][a-z0-9_]*)'$")
_UNCHECKED_SECTION = 'Changelog'


def prose_tokens(text):
    """``(token, lineno)`` per backticked token in ``text``'s prose: outside fences and the Changelog."""
    fence, section = None, ''
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.startswith('```'):
            fence = fence_info(line)[0] if fence is None else None
            continue
        if fence is not None:
            continue
        if line.startswith('## '):
            section = line[3:].strip()
        if section == _UNCHECKED_SECTION:
            continue
        for token in re.findall(r'`([^`\n]+)`', line):
            yield token, lineno


def _route_matches(route_path, asked):
    """Whether the app's ``route_path`` answers the guide's ``asked`` path.

    A guide's ``<name>`` segment stands for any one segment, as the app's
    ``{name}`` does; the app's ``{name:path}`` takes the rest.
    """
    have, want = route_path.strip('/').split('/'), asked.strip('/').split('/')
    for i, segment in enumerate(have):
        if segment.endswith(':path}'):
            return len(want) > i
        if i >= len(want):
            return False
        if not (segment.startswith('{') or want[i].startswith('<') or segment == want[i]):
            return False
    return len(have) == len(want)


class _ProseOwners:
    """What each kind of prose name is resolved against: the owner of that kind."""

    def __init__(self, live_objects, app):
        from modules import run_outcome, wire_encoding
        from rest.problems import _SERVER_STATUS
        from tests.api_reference import outcome_classes
        from tests.marked_set import marked_set

        self.live = live_objects
        self.universe = wire_encoding.project_classes()
        self.marked = marked_set(self.universe)
        self.routes = [
            (route.path, getattr(route, 'methods', None) or set()) for route in app.routes
        ]
        self.settings = live_objects['session'].get_settings_snapshot().keys()
        self.reasons = (
            # A kind itself (Refusal, Fault, ...) carries no reasons; its classes do.
            {r for cls in outcome_classes().values() for r in getattr(cls, 'causes', ())}
            | set(_SERVER_STATUS)
            | set(run_outcome.ENDINGS)
        )

    def judge(self, token):
        """``(kind, why it is wrong or None)``, or None for a token of no checked kind."""
        if match := _RECEIVER_FORM.match(token):
            receiver, chain = match.group(1), tuple(match.group(2).lstrip('.').split('.'))
            root = self.live[receiver]
            missing = _first_missing_attr(root, receiver, chain)
            if missing:
                return 'member', f'no attribute {missing}'
            unmarked = _first_unmarked_step(root, receiver, chain)
            return 'member', unmarked and f'{unmarked} is not API'
        if (match := _ROUTE.match(token)) and match.group(2).startswith('/api'):
            method, path = match.group(1), match.group(2)
            answering = [m for p, m in self.routes if _route_matches(p, path)]
            if not answering:
                return 'route', 'no route answers it'
            if method and not any(method in m for m in answering):
                return 'route', f'no route answers it to {method}'
            return 'route', None
        if (match := _CLASS_MEMBER.match(token)) and match.group(1) in self.universe:
            cls, name = self.universe[match.group(1)], match.group(2)
            if f'{cls.__name__}.{name}' in self.marked:
                return 'member', None
            if issubclass(cls, enum.Enum) and name in cls.__members__:
                return 'member', None
            return 'member', f'{cls.__name__}.{name} is not API'
        if (match := _DOTTED.match(token)) and match.group(1) in self.settings:
            try:
                self.live['session'].get_setting(token)
            except LumascopeError as refused:
                if getattr(refused, 'reason', None) == 'not_a_setting':
                    return 'setting', 'not a setting'
            return 'setting', None
        if match := _QUOTED_REASON.match(token):
            reason = match.group(1)
            return 'reason', None if reason in self.reasons else 'no published outcome carries it'
        return None


@pytest.fixture(scope='module')
def prose_owners(live_objects):
    from rest.app import build_app

    return _ProseOwners(live_objects, build_app(live_objects['session']))


def prose_name_failures(text, label, owners):
    """``(failures, counts)``: a line per prose name that does not resolve, and each kind's count."""
    failures, counts = [], collections.Counter()
    for token, lineno in prose_tokens(text):
        judged = owners.judge(token)
        if judged is None:
            continue
        kind, why = judged
        counts[kind] += 1
        if why:
            failures.append(f'  {label}:{lineno}  `{token}`  -> {why}')
    return failures, counts


class TestLumascopeSkillsProseNamesResolve:
    """Polarity 5: a name the guide's prose states resolves, by its kind, at its owner."""

    def test_every_prose_name_resolves(self, doc_text, prose_owners):
        failures, _ = prose_name_failures(doc_text, DOC, prose_owners)
        assert not failures, (
            'LumascopeSkills.md names something in its prose that does not resolve: a '
            'member or field that is not API, a route no route answers, a settings path '
            'that is no setting, or a quoted reason no published outcome carries. Correct '
            'the DOCUMENT.\n' + '\n'.join(failures)
        )

    def test_the_check_reaches_every_kind(self, doc_text, prose_owners):
        _, counts = prose_name_failures(doc_text, DOC, prose_owners)
        kinds = {'member', 'route', 'setting', 'reason'}
        assert kinds <= counts.keys(), (
            f'prose names found by kind: {dict(counts)}; a kind with none means the '
            'token patterns have regressed'
        )

    def test_a_wrong_name_of_each_kind_is_reported(self, prose_owners):
        planted = (
            '## A chapter\n'
            'Read `scope.motion.get_turret_slot()` and `session.controls_locked`.\n'
            "The record is `BringUpRecord.substitution`, its reason `'merge_timeout'`.\n"
            'Read `GET /api/v1/handles/Protocol/<id>/period` and `video.max_fsp`.\n'
            'Every right one: `scope.motion.turret_status()`, `BringUpRecord.substitutions`,\n'
            "`'not_offered'`, `GET /api/v1/handles/Protocol/<id>/period_s`, `video.max_fps`.\n"
            '## Changelog\n'
            '`session.retired_member` was removed.\n'
        )
        failures, _ = prose_name_failures(planted, 'planted', prose_owners)
        reported = '\n'.join(failures)
        for expected in (
            'planted:2  `scope.motion.get_turret_slot()`  -> no attribute',
            'planted:2  `session.controls_locked`  -> no attribute',
            'planted:3  `BringUpRecord.substitution`  -> BringUpRecord.substitution is not API',
            "planted:3  `'merge_timeout'`  -> no published outcome carries it",
            'planted:4  `GET /api/v1/handles/Protocol/<id>/period`  -> no route answers it',
            'planted:4  `video.max_fsp`  -> not a setting',
        ):
            assert expected in reported, reported
        assert len(failures) == 6, reported


_LINK = re.compile(r'\]\(([^)\s]+)\)')


def anchors_of(text):
    """Every anchor a markdown file holds: its ``<a id>`` tags and its headings' slugs."""
    ids = set(re.findall(r'<a id="([^"]+)"></a>', text))
    for heading in re.findall(r'^#{1,6} (.+)$', text, re.MULTILINE):
        slug = re.sub(r'[^\w\- ]', '', heading.strip().lower()).replace(' ', '-')
        ids.add(slug)
    return ids


def link_failures(text, label, reference_text):
    """A line per link from ``text`` to an anchor that is not there: its own, or the reference's."""
    targets = {'': anchors_of(text), 'API_REFERENCE.md': anchors_of(reference_text)}
    failures = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for link in _LINK.findall(line):
            page, sep, anchor = link.partition('#')
            if sep and page in targets and anchor not in targets[page]:
                failures.append(f'  {label}:{lineno}  {link}  -> no such anchor')
    return failures


class TestLumascopeSkillsLinksResolve:
    """Polarity 6: a link from the guide reaches an anchor, in it or in the generated reference."""

    def test_every_link_reaches_its_anchor(self, doc_text):
        from tests.api_reference import REFERENCE

        failures = link_failures(doc_text, DOC, REFERENCE.read_text(encoding='utf-8'))
        assert not failures, (
            'LumascopeSkills.md links to an anchor that is not there. A reference anchor '
            'is a member, field or class the generator rendered, or one of its headings.\n'
            + '\n'.join(failures)
        )

    def test_a_dead_link_is_reported(self):
        planted = (
            '## Here\n'
            'See [create](API_REFERENCE.md#ScopeSession.create), [gone](API_REFERENCE.md#'
            'BringUpRecord.substitution), [here](#here), [there](#there) and '
            '[events](API_REFERENCE.md#events).\n'
        )
        reference = '<a id="ScopeSession.create"></a>\n## Events\n'
        assert link_failures(planted, 'planted', reference) == [
            '  planted:2  API_REFERENCE.md#BringUpRecord.substitution  -> no such anchor',
            '  planted:2  #there  -> no such anchor',
        ]
