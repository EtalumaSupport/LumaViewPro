"""Structural guard: the guide's code is the API's, and it runs.

The guide is the hub, ``docs/lumascope-skills.md``, and the files its map
names under ``docs/guide/`` (``guide_files``): every check below reads each
file on its own -- its chapters, its fences' names, its Changelog -- and the
coverage checks read their union. The map is the one list of the files: a
file under ``docs/guide/`` that no row names, or a row naming a file git does
not track, is reported (``guide_set_failures``).

Every call form a reader could copy out of the guide must resolve on a real
object (polarity 1) AND accept the arguments the guide passes it (polarity
2), and every ``python`` fence must run on the simulator as its info string
says (polarity 3). Resolution alone is not enough: a member can exist and
still reject the documented call, which is how the guide came to publish
``fv.count_frame(chunk_data=None, frame_ts=None)`` against a signature that
had neither parameter. Binding is not enough either: a call can bind and
still fail on the shipped settings, as the guide's unlit still, its
``new_protocol(..., use_zstacking=True)`` with no stack configured, and its
moves on an unhomed scope did. When this test goes red the guide is what gets
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
Each ``## `` chapter's ``python`` fences run in order in one namespace, as a
reader's program runs on a new install: the guide's bring-up fence, the one
whose info string says ``bringup``, runs first as the chapter's preamble, and
the ``session`` and ``scope`` it leaves are the chapter's
(``_ChapterSessions``). The info string says what a fence needs beyond that:
``tier=`` and ``model=`` name the board it runs on, ``raises=<Class>:<reason>``
the refusal it teaches and the reason it carries, raised by its last statement
(``raises=<Class>`` alone for an error that carries no reason), and ``norun``
a fence no simulator can run (serving forever, say), which is still resolved
and bound.

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

A backticked path into a project package (```_lumascope.modules.x.y```, ``drivers.``,
``rest.``) names something a reader is told to import, so it must be API like
a fenced import.

A relative link in a published document -- the README, ``docs/`` and the
examples' README -- reaches its file, and its anchor in a markdown file
(polarity 6): a heading, or an anchor the generated reference holds -- a
member, field or class the generator rendered.

The docs beyond the guide
-------------------------
The plugin tutorial (its level table aside, a changelog), the examples'
README, the manual and the generated reference teach the API too, so a
member or a route each names is held to the guide's rules, and no line of
them or of the guide names the internal package but the one sentence that
says it is internal (polarity 7). There, and in the guide, a member marked
engineering is off the surface. What they teach today off it is the seed,
which only shrinks.
"""

import ast
import collections
import contextlib
import dataclasses
import enum
import importlib
import inspect
import json
import pathlib
import re
import shlex
import subprocess
import traceback
import urllib.parse
from unittest import mock
import warnings

import pytest

from _lumascope.lib.lumascope_error import LumascopeError
from tests.ast_seams import REPO_ROOT, MemberNames, Packages, Words, tracked_files
from tests.settings_fixtures import complete_settings, installation_of_its_own

HUB = 'docs/lumascope-skills.md'

# Receivers the reference uses for the live objects an L2 caller holds.
# `scope` is the Lumascope composition root, `session` the ScopeSession L2
# entry point. Both are resolved against real instances rather than classes:
# `session.scope` is an instance attribute, so a class-level hasattr would
# report every `session.scope.*` form in the document as unresolved.
#
# `caps` is an alias the guide establishes in its own example block
# (`caps = scope.capabilities`, in "What LumaViewPro is") and then reads the
# capabilities through. Without it those lines would sit outside the guard.
# `ctx` is the plugin tutorial's name for the session a plugin is handed.
_CALL_FORM = re.compile(r'\b(scope|session|caps|ctx)((?:\.[A-Za-z_][A-Za-z0-9_]*)+)')

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
    ``raises=AxisStateUnknownError:position_unknown``) or a bare flag
    (``norun``), whose value is True.
    """
    words = line[3:].split()
    options = {}
    for word in words[1:]:
        key, sep, value = word.partition('=')
        options[key] = value if sep else True
    return (words[0] if words else ''), options


_MAP_HEADING = '## The map'
_MAP_LINK = re.compile(r'\]\((guide/[^)#\s]+\.md)\)')


def guide_map(root=None):
    """``[(task, (file, ...)), ...]``: each row of the hub's map and the files it loads.

    The files are repo-relative, the hub left out (every task loads it). The
    map is the one list of the guide's files; a row's links into ``guide/``
    are what it loads.
    """
    root = REPO_ROOT if root is None else root
    rows, inside = [], False
    for line in (root / HUB).read_text(encoding='utf-8').splitlines():
        if line.startswith('## '):
            inside = line == _MAP_HEADING
            continue
        if inside and line.startswith('|'):
            task, *_, load = (cell.strip() for cell in line.strip('|').split('|'))
            files = tuple(f'docs/{f}' for f in _MAP_LINK.findall(load))
            if files:
                rows.append((task, files))
    return rows


def guide_files(root=None):
    """The guide's files, repo-relative: the hub, then each file the map names, in its order."""
    names = [HUB]
    for _, files in guide_map(root):
        names += [f for f in files if f not in names]
    return names


def guide_texts(root=None):
    """``[(path, text), ...]`` for each of the guide's files."""
    root = REPO_ROOT if root is None else root
    return [(root / rel, (root / rel).read_text(encoding='utf-8')) for rel in guide_files(root)]


def guide_set_failures(root=None):
    """A line per file the map and git disagree on."""
    root = REPO_ROOT if root is None else root
    named = set(guide_files(root)) - {HUB}
    tracked = set(tracked_files(('docs/guide',), '*.md', root=root))
    failures = [f'  {f}  no row of the map names it' for f in sorted(tracked - named)]
    failures += [
        f'  {f}  the map names it and git tracks no such file' for f in sorted(named - tracked)
    ]
    return failures


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
def guide():
    """The guide's files and their texts, read once for the module.

    pin-justified: the guard's subject IS the document text, so there is no
    AST seam to assert instead -- the doc-example case the source-pin ratchet
    names as justified. One read site serves every check.
    """
    return guide_texts()


@pytest.fixture(scope='module')
def live_objects(tmp_path_factory):
    """Every receiver the reference documents, as live instances.

    Module-scoped: the headless session builds the full executor topology,
    which is ~0.5s and pointless to repeat per test.
    """
    warnings.simplefilter('ignore', FutureWarning)
    from _lumascope.modules.scope_session import ScopeSession

    # The reference's `scope` is a session's: a scope reads the labware,
    # offset, turret map, objective and scale bar from its session's settings.
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path_factory.mktemp('live'))), simulate=True
    )
    # The namespaces a plugin is handed (`ctx.plugins`) exist once the host
    # loads plugins. A stand-in by design: no installed plugin is discovered,
    # since what is installed in this environment is not what the docs teach.
    with mock.patch('importlib.metadata.entry_points', return_value=[]):
        session.load_plugins()
    scope = session.scope
    yield {
        'scope': scope,
        'session': session,
        'caps': scope.capabilities,
        'ctx': session,
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
    """The first ``Class.member`` the chain reaches that is not on the surface, or None.

    The surface is what is marked and not marked engineering: an
    engineering member is a bench or support member, outside what a reader
    is taught.
    """
    from _lumascope.modules.api_surface import fields_of, is_engineering, mark_of

    obj = root
    for attr in chain:
        owner = next((k for k in type(obj).__mro__ if attr in vars(k)), None)
        if owner is None and attr in getattr(obj, '__dict__', {}):
            owner = type(obj)
        if owner is None or not owner.__module__.startswith('_lumascope.modules'):
            return None
        value = vars(owner).get(attr)
        marked = (mark_of(value) and not is_engineering(value)) or any(
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


def unresolved_call_forms(guide, live_objects):
    """A line per fenced call form of ``guide``'s files that does not resolve on the live API."""
    failures = []
    for path, text in guide:
        for receiver, chain, lineno in _fenced_call_forms(text):
            missing = _first_missing_attr(live_objects[receiver], receiver, chain)
            if missing:
                failures.append(
                    f'  {path}:{lineno}  {receiver}.{".".join(chain)}  -> no attribute {missing}'
                )
    return failures


def _planted_tree(root, rows, files):
    """A git tree at ``root`` holding a hub whose map has ``rows``, and ``files`` by repo path."""
    (root / 'docs' / 'guide').mkdir(parents=True)
    hub = '# Hub\n\n## The map\n\n| Task | Load |\n|---|---|\n' + ''.join(
        f'| {task} | ' + ', '.join(f'[`{f}`](guide/{f})' for f in loads) + ' |\n'
        for task, loads in rows
    )
    (root / HUB).write_text(hub + '\n## Changelog\n', encoding='utf-8')
    for rel, text in files.items():
        (root / rel).write_text(text, encoding='utf-8')
    subprocess.run(['git', 'init', '-q'], cwd=root, check=True)
    subprocess.run(['git', 'add', '.'], cwd=root, check=True)
    return root


class TestTheGuideIsTheMapsFiles:
    """The hub's map is the one list of the guide's files."""

    def test_the_map_names_every_file(self):
        failures = guide_set_failures()
        assert not failures, (
            "The guide's map and its files disagree: name each file under docs/guide/ in a "
            'row of the map.\n' + '\n'.join(failures)
        )

    def test_a_file_the_map_names_is_read(self, tmp_path, live_objects):
        root = _planted_tree(
            tmp_path,
            [('Plant', ['planted.md'])],
            {'docs/guide/planted.md': '# Planted\n\n```python\nsession.no_such_member()\n```\n'},
        )
        assert guide_files(root) == [HUB, 'docs/guide/planted.md']
        assert unresolved_call_forms(guide_texts(root), live_objects) == [
            f'  {root / "docs/guide/planted.md"}:4  session.no_such_member  '
            '-> no attribute session.no_such_member'
        ]

    def test_a_file_the_map_does_not_name_and_a_named_file_git_lacks_are_reported(self, tmp_path):
        root = _planted_tree(
            tmp_path,
            [('Fits', ['small.md']), ('Gone', ['gone.md'])],
            {
                'docs/guide/small.md': '# Small\n',
                'docs/guide/stray.md': '# Stray\n',
            },
        )
        assert guide_set_failures(root) == [
            '  docs/guide/stray.md  no row of the map names it',
            '  docs/guide/gone.md  the map names it and git tracks no such file',
        ]


class TestLumascopeSkillsCallFormsResolve:
    """Polarity 1: documented call forms exist on the live API surface."""

    def test_every_fenced_call_form_resolves(self, live_objects, guide):
        failures = unresolved_call_forms(guide, live_objects)
        assert not failures, (
            'The guide documents call forms that do not resolve on the '
            'live API. An L2 reader copying these lines gets AttributeError.\n'
            'Correct the DOCUMENT to match the surface -- do not add a member to '
            'satisfy a doc line.\n' + '\n'.join(sorted(set(failures)))
        )

    def test_every_fenced_call_form_is_api(self, live_objects, guide):
        """Each step of a copyable call form is a marked member.

        Resolving is not enough: an unmarked member resolves too, and a reader
        who copies it builds on something that can change in any release and
        that no wire carries. A step on an object no project class defines --
        a dict, a number, a library type -- ends the walk: what it holds is
        not ours to mark.
        """
        unmarked = []
        for path, text in guide:
            for receiver, chain, lineno in _fenced_call_forms(text):
                step = _first_unmarked_step(live_objects[receiver], receiver, chain)
                if step:
                    unmarked.append(f'  {path}:{lineno}  {receiver}.{".".join(chain)}  -> {step}')
        assert not unmarked, (
            'The guide teaches call forms through members that are not API '
            '(no @api, not in @api_fields). Rewrite the line to the marked door, or '
            'drop it.\n' + '\n'.join(sorted(set(unmarked)))
        )

    def test_guard_actually_reaches_the_reference(self, guide):
        """The extractor is wired to real content, not silently finding nothing.

        A regex or fence-tracking regression would make the check above pass
        vacuously. Every documented receiver is exercised in the reference
        today, so a run that misses one means the extractor broke, not that
        the doc got smaller.
        """
        receivers = {r for _, text in guide for r, _, _ in _fenced_call_forms(text)}
        assert receivers == {'scope', 'session', 'caps'}, (
            f'expected call forms for every documented receiver, saw {receivers or "none"} -- '
            'the fence tracker or the call-form pattern has regressed'
        )


# The receiver polarity 2 adds on top of the aliases. An L2 script names this
# class before any alias exists -- `ScopeSession.create(...)` is the first call
# a consumer makes -- so it carries the arguments most likely to be copied and
# the least likely to be re-read. Being a class rather than an instance, it is
# invisible to the alias-rooted attribute check.
_CONSTRUCTOR_RECEIVERS = MemberNames('src/_lumascope/modules/scope_session.py', {'ScopeSession'})

# Stands in for an argument value. bind() checks arity and parameter names; it
# never looks at what is passed, and a doc example's values are illustrative.
_SENTINEL = object()


@pytest.fixture(scope='module')
def bind_roots(live_objects):
    """Every receiver polarity 2 resolves against: the aliases plus the class."""
    from _lumascope.modules.scope_session import ScopeSession

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

    def test_every_documented_call_binds(self, bind_roots, guide):
        failures = []
        for path, text in guide:
            for receiver, chain, nargs, kwnames, lineno in _documented_calls(text):
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
                    failures.append(f'  {path}:{lineno}  {form}(...)  -> {exc}')

        assert not failures, (
            'The guide documents calls the live signature will not accept. '
            'An L2 reader copying these lines gets TypeError.\n'
            'Correct the DOCUMENT to match the signature -- do not add a parameter '
            'to satisfy a doc line.\n' + '\n'.join(sorted(set(failures)))
        )

    def test_bind_check_reaches_the_receivers_that_carry_arguments(self, guide):
        """The extractor is wired to real content, not silently finding nothing.

        A fence-tracking or parse regression would make the check above pass
        vacuously. Asserted as a set of receivers rather than a count of forms:
        the count moves whenever anyone adds an example, and a test that goes
        red for that gets bumped blindly instead of read.

        `caps` is deliberately absent. The reference uses it almost entirely
        for property reads, so requiring a CALL form there would pin a wording
        choice rather than a contract.
        """
        receivers = {r for _, text in guide for r, _, _, _, _ in _documented_calls(text)}
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
_PROJECT_PACKAGES = Packages(
    {'_lumascope/modules', '_lumascope/drivers', '_lumascope/rest'}, under='src'
)
_PROJECT_MODULES = tuple(sorted(p.replace('/', '.') for p in _PROJECT_PACKAGES))
_PROSE_PATH = re.compile(r'`(_lumascope\.(?:modules|drivers|rest)\.[A-Za-z_][A-Za-z0-9_.]*)')


# What a reader imports names from: the public module, which every published
# doc teaches, and the project packages, which a doc must not (the docs'
# import guard) but whose names are still judged here if one does.
_IMPORTED_FROM = ('lumascope', *_PROJECT_MODULES)


def _in_project(module):
    """Whether the module name ``module`` is the public module, a project package's, or one of theirs."""
    return any(module == p or module.startswith(f'{p}.') for p in _IMPORTED_FROM)


def why_not_api(obj):
    """Why ``obj``, a name from a project package, is not API; None when it is.

    API is an exception, an enum or one of its members, a class that
    publishes fields or has marked members, or a marked function. A module
    is where names live, not a name: each of its names a reader uses is
    checked where it is used.
    """
    from _lumascope.modules.api_surface import fields_of, mark_of

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
    from _lumascope.modules.api_surface import fields_of

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
        (f'{p}.{name}' for p in _PROJECT_MODULES if importlib.util.find_spec(f'{p}.{name}')),
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
                if _in_project(alias.name):
                    bound[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom) and _in_project(node.module or ''):
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

    def test_every_project_name_the_reference_teaches_is_api(self, guide):
        failures = [f for path, text in guide for f in _doc_project_name_failures(text, path)]
        assert not failures, (
            'The guide teaches a project name that is not API (an exception, an '
            'enum, a class with marked members or published fields, or a marked function), '
            'or calls one with arguments it will not take. Teach the member that does the '
            'job, or say in words what the API does.\n' + '\n'.join(failures)
        )

    def test_a_planted_helper_import_call_and_path_are_reported(self):
        planted = (
            '```python\n'
            'from _lumascope.modules.image_save import save_image\n'
            'from _lumascope.modules import settings_init\n'
            'save_image(None)\n'
            'settings_init.current_json_path(".")\n'
            'from _lumascope.modules.scope_session import ScopeSession\n'
            'ScopeSession.user_settings_file(".", "extra")\n'
            'from lumascope import ScopeSession as Public\n'
            'Public.user_settings_file(".", "extra")\n'
            '```\n'
            '\n'
            'Prose naming `_lumascope.modules.post_processing.with_pixels_per_um`.\n'
            '\n'
            '```\n'
            'not python at all (\n'
            '```\n'
        )
        reported = '\n'.join(_doc_project_name_failures(planted, 'planted'))
        for expected in (
            'planted:2  _lumascope.modules.image_save.save_image  -> an unmarked function',
            'planted:5  _lumascope.modules.settings_init.current_json_path  -> an unmarked function',
            'planted:7  _lumascope.modules.scope_session.ScopeSession.user_settings_file(...)',
            'planted:9  lumascope.ScopeSession.user_settings_file(...)',
            'planted:12  _lumascope.modules.post_processing.with_pixels_per_um',
            'planted:15  the fence does not parse',
        ):
            assert expected in reported, reported


def test_the_reference_names_each_protocol_constructor_and_each_is_api(guide):
    """A presence claim the call-form check does not make: the guide
    teaches the three ways to get a protocol, and each is a marked member."""
    from _lumascope.modules.api_surface import API, mark_of
    from _lumascope.modules.scope_session import ScopeSession

    doc = '\n'.join(text for _, text in guide)
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


@dataclasses.dataclass(frozen=True)
class Bringup:
    """The guide's bring-up fence: the file and line it starts at, and its source."""

    label: str
    start: int
    source: str


def guide_bringup(guide):
    """``(bringup, failures)``: the one fence of ``guide`` that says ``bringup``.

    ``guide`` is ``(label, text)`` per file. ``failures`` is a line per way
    the marked fences are not one runnable ``python`` fence, and ``bringup``
    is None when there is one.
    """
    marked = [
        (label, start, language, options, source)
        for label, text in guide
        for _, fences in guide_chapters(text)
        for start, language, options, source in fences
        if options.get('bringup')
    ]
    if len(marked) != 1:
        where = ', '.join(f'{label}:{start}' for label, start, *_ in marked) or 'none'
        return None, [
            f'  {len(marked)} fences say bringup ({where}): the guide has exactly one, '
            'the bring-up every chapter starts from'
        ]
    label, start, language, options, source = marked[0]
    failures = []
    if language != 'python':
        failures.append(
            f'  {label}:{start}  the bring-up fence is {language or "untyped"}, not python'
        )
    if options.get('norun'):
        failures.append(f'  {label}:{start}  the bring-up fence says norun: every chapter runs it')
    return (None if failures else Bringup(label, start, source)), failures


class _ChapterSessions:
    """The simulated sessions one chapter's fences run against, shut down together.

    A chapter starts as a reader's program does: the guide's bring-up runs
    in the chapter's namespace on a new install of the default tier and
    model, and leaves the chapter its ``session`` and ``scope``. A fence
    naming another tier or model runs the bring-up again on that board, and
    later fences keep it. A session a fence makes for itself is shut down
    with the chapter's.
    """

    def __init__(self, namespace, bringup):
        self._namespace = namespace
        self._bringup = bringup
        self._key = None
        self._made = []

    def use(self, options):
        """Bring up the session a fence's ``options`` ask for, unless it is up.

        A failure line when the bring-up does not run, else None.
        """
        from _lumascope.modules.scope_session import ScopeSession

        if self._key is not None and 'tier' not in options and 'model' not in options:
            return None
        tier, model = options.get('tier', _DEFAULT_TIER), options.get('model')
        if (tier, model) == self._key:
            return None
        overrides = {'simulator_tier': tier, 'objective_confirmed': None, 'lvp_lock_port': 0}
        if model is not None:
            overrides['microscope'] = model
        # The bring-up's load_user_settings reads the installation the
        # chapter runs from, as a reader's does: the chapter's board as a new
        # install, never this machine's current.json.
        pathlib.Path('data', 'current.json').write_text(
            json.dumps(complete_settings(**overrides)), encoding='utf-8'
        )
        # The bring-up rebinds the namespace's names; a session made under
        # them is kept here, or nothing would shut it down.
        self._made.extend(v for v in self._namespace.values() if isinstance(v, ScopeSession))
        self._key = (tier, model)
        bringup = self._bringup
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            failure = _run_fence(bringup.label, bringup.start, {}, bringup.source, self._namespace)
        # Kept as soon as the bring-up leaves it: a fence that binds
        # ``session`` to a session of its own would otherwise leave it open.
        self._made.extend(v for v in self._namespace.values() if isinstance(v, ScopeSession))
        return failure

    def close(self):
        from _lumascope.modules.scope_session import ScopeSession

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
    expected, _, stated_reason = (options.get('raises') or '').partition(':')
    expected, stated_reason = expected or None, stated_reason or None
    code = compile('\n' * (start - 1) + source, filename, 'exec')
    try:
        exec(code, namespace)
    except (Exception, LumascopeError) as exc:
        raised = type(exc).__name__
        if expected is None:
            return f'  {label}:{start}  raised {raised}: {exc}'
        if raised != expected:
            return f'  {label}:{start}  raised {raised}, not the {expected} it states: {exc}'
        reason = getattr(exc, 'reason', None)
        if reason != stated_reason:
            states = (
                f'not the {stated_reason!r} it states' if stated_reason else 'and states no reason'
            )
            return (
                f'  {label}:{start}  raised {expected} carrying reason {reason!r}, {states}: {exc}'
            )
        first, last = _last_statement_lines(source)
        at = _raised_at(exc, filename)
        if at is None or not start + first - 1 <= at <= start + last - 1:
            return f'  {label}:{start}  raised {expected} at line {at}, before its last statement'
        return None
    if expected is not None:
        return f'  {label}:{start}  states raises={expected} and raised nothing'
    return None


def _chapter_failure(label, runnable, bringup):
    """The failure line of one chapter's ``runnable`` fences after the bring-up, or None.

    The chapter runs from an installation of its own, as a reader runs from
    theirs: a fence's ``'.'`` is that folder, so the bring-up's
    ``settings_file`` and a fence's ``user_settings_file(root)`` save there,
    never into this checkout's ``data/current.json`` nor into another
    chapter's installation.
    """
    namespace = {'__name__': '__guide__'}
    sessions = _ChapterSessions(namespace, bringup)
    with contextlib.chdir(installation_of_its_own()):
        try:
            failure = sessions.use(runnable[0][2] if runnable else {})
            for start, _, options, source in runnable:
                failure = failure or sessions.use(options)
                if failure:
                    return failure
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', FutureWarning)
                    failure = _run_fence(label, start, options, source, namespace)
                if failure:
                    return failure
            return failure
        finally:
            sessions.close()


def guide_fence_failures(guide, bringup):
    """A line per ``python`` fence of ``guide`` that does not run as its info string states.

    ``guide`` is ``(label, text)`` per file and ``bringup`` its bring-up
    fence (``guide_bringup``), which runs on its own first: every chapter
    starts from it, so when it fails that is the one line, and no chapter
    runs. Each chapter's fences then run in order in one namespace after the
    bring-up (``_ChapterSessions``); a fence flagged ``norun`` is skipped
    here and still resolved and bound by the checks above. A fence that
    raises ends the chapter: what follows it assumed it ran.
    """
    alone = _chapter_failure(bringup.label, [], bringup)
    if alone:
        return [f'{alone}  (the bring-up, run on its own: no chapter ran)']
    failures = []
    for label, text in guide:
        for heading, fences in guide_chapters(text):
            runnable = [f for f in fences if f[1] == 'python' and not f[2].get('norun')]
            failure = runnable and _chapter_failure(label, runnable, bringup)
            if failure:
                failures.append(f'{failure}  (chapter "{heading}")')
    return failures


# A bring-up as the guide's teaches it, for the planted guides: a new install
# answers which objective is mounted.
_PLANTED_BRINGUP = Bringup(
    'bringup',
    1,
    'from lumascope import ScopeSession\n'
    "settings = ScopeSession.load_user_settings('.')\n"
    "session = ScopeSession.create(settings, source_path='.', simulate=True)\n"
    'session.run_application_session()\n'
    'scope = session.scope\n'
    'question = session.objective_question()\n'
    'if question is not None:\n'
    '    session.confirm_objective(question.proposed, slot=question.turret_slot)\n',
)


class TestLumascopeSkillsFencesRun:
    """Polarity 3: every ``python`` fence runs on the simulator as its info string states."""

    @pytest.mark.slow
    def test_every_python_fence_runs(self, guide):
        bringup, failures = guide_bringup(guide)
        failures = failures or guide_fence_failures(guide, bringup)
        assert not failures, (
            'The guide has python fences that do not run on the simulator as '
            'written. A fence that teaches a refusal states it (```python '
            'raises=<Class>:<reason>) and ends in it; one that needs a board says tier= or '
            'model=; one no simulator can run is ```python norun; the one every chapter '
            'starts from is ```python bringup.\n' + '\n'.join(failures)
        )

    def test_a_fence_that_does_not_do_what_it_states_is_reported(self):
        planted = (
            '## Lit\n'
            '```python\n'
            "still = session.capture_still('BF')\n"
            'assert still.record.dark is False\n'
            '```\n'
            '## Refused\n'
            '```python raises=PositionOutOfRangeError:position_out_of_range\n'
            'scope.motion.move_z(1e9)\n'
            'x = 1\n'
            '```\n'
            '## Wrong reason\n'
            '```python raises=PositionOutOfRangeError:not_a_number\n'
            'scope.motion.move_z(1e9)\n'
            '```\n'
            '## Reason unstated\n'
            '```python raises=PositionOutOfRangeError\n'
            'scope.motion.move_z(1e9)\n'
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
        reported = '\n'.join(guide_fence_failures([('planted', planted)], _PLANTED_BRINGUP))
        for expected in (
            'planted:3  raised AssertionError',
            'planted:8  raised PositionOutOfRangeError at line 8, before its last statement',
            "planted:13  raised PositionOutOfRangeError carrying reason 'position_out_of_range', "
            "not the 'not_a_number' it states",
            "planted:17  raised PositionOutOfRangeError carrying reason 'position_out_of_range', "
            'and states no reason',
            'planted:21  raised ArgumentRefusedError, not the ValueError it states',
            'planted:25  states raises=ValueError and raised nothing',
        ):
            assert expected in reported, reported
        assert 'planted:32' not in reported, reported
        assert 'planted:36' not in reported, reported

    def test_a_bringup_that_does_not_answer_a_new_install_is_reported_at_the_saved_still(self):
        unanswering = Bringup('bringup', 1, _PLANTED_BRINGUP.source.split('question = ')[0])
        planted = (
            '## Saved\n'
            '```python\n'
            'session.show_channel("BF")\n'
            'session.manual_capture.capture().result(timeout=30)\n'
            '```\n'
        )
        reported = guide_fence_failures([('planted', planted)], unanswering)
        assert len(reported) == 1, reported
        assert reported[0].startswith('  planted:3  raised ObjectiveUnknownError'), reported
        assert guide_fence_failures([('planted', planted)], _PLANTED_BRINGUP) == []

    def test_a_failing_bringup_is_reported_once_and_no_chapter_runs(self):
        failing = Bringup('bringup', 4, "raise RuntimeError('the bring-up broke')\n")
        planted = '## One\n```python\nx = 1\n```\n## Two\n```python\nraise ValueError\n```\n'
        assert guide_fence_failures([('planted', planted)], failing) == [
            '  bringup:4  raised RuntimeError: the bring-up broke  '
            '(the bring-up, run on its own: no chapter ran)'
        ]

    def test_the_guide_has_one_bringup_and_it_is_a_runnable_python_fence(self):
        one = '```python bringup\nx = 1\n```\n'
        cases = {
            'none': [('a', '```python\nx = 1\n```\n')],
            'two': [('a', one), ('b', one)],
            'bash': [('a', '```bash bringup\ncurl -s $API/status\n```\n')],
            'norun': [('a', '```python bringup norun\nx = 1\n```\n')],
        }
        reported = {name: guide_bringup(guide) for name, guide in cases.items()}
        assert reported['none'][1] == [
            '  0 fences say bringup (none): the guide has exactly one, '
            'the bring-up every chapter starts from'
        ]
        assert reported['two'][1][0].startswith('  2 fences say bringup (a:2, b:2)')
        assert reported['bash'][1] == ['  a:2  the bring-up fence is bash, not python']
        assert reported['norun'][1] == [
            '  a:2  the bring-up fence says norun: every chapter runs it'
        ]
        assert all(bringup is None for bringup, _ in reported.values())
        assert guide_bringup([('a', one)]) == (Bringup('a', 2, 'x = 1'), [])


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
    # ``${NAME}`` or ``$NAME``: a brace closes the name only after one opened
    # it, so the ``}`` of JSON after ``$NAME`` stays the JSON's.
    return re.sub(
        r'\$(?:\{([A-Z_][A-Z0-9_]*)\}|([A-Z_][A-Z0-9_]*))',
        lambda m: variables.get(m.group(1) or m.group(2), m.group(0)),
        token,
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
    """``text``'s walkthrough, sent to a server on a new install, as
    ``python -m lumascope.rest --simulate`` serves one: the scope brought up
    and its startup motion run, and nothing answered for the walkthrough, so
    a question a new install asks is the walkthrough's own to answer."""
    from fastapi.testclient import TestClient

    from _lumascope.modules.scope_session import ScopeSession
    from _lumascope.rest.app import build_app

    settings = complete_settings(simulator_tier=_DEFAULT_TIER, objective_confirmed=None)
    session = ScopeSession.create(settings, simulate=True, warn_pre_release=False)
    try:
        session.run_application_session()
        with TestClient(build_app(session)) as client:
            return walkthrough_failures(text, label, client)
    finally:
        session.shutdown()


class TestLumascopeSkillsWalkthroughRuns:
    """Polarity 4: the REST walkthrough's ``curl`` lines answer what their comments state."""

    @pytest.mark.slow
    def test_every_curl_line_answers_as_stated(self, guide):
        # One shell per file: a walkthrough's lines read the ids its own earlier lines made.
        walked = [(path, text) for path, text in guide if any(bash_fence_lines(text))]
        assert walked, 'no file of the guide holds a bash fence: the walkthrough went unread'
        failures = [f for path, text in walked for f in _walk(text, path)]
        assert not failures, (
            'The guide has a curl line that does not answer as its comment '
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

    def test_a_variable_before_a_json_brace_leaves_the_brace(self):
        variables = {'SLOT': 'null', 'ID': '7'}
        assert _expand('{"slot": $SLOT}', variables) == '{"slot": null}'
        assert _expand('${ID}x/$ID', variables) == '7x/7'


# The names the guide's prose is held to, each kind by its owner; every other
# backticked token is not checked. The Changelog names what no longer
# resolves, by design, so its section is left out by its heading.
_RECEIVER_FORM = re.compile(r'^(scope|session|caps|ctx)((?:\.[A-Za-z_]\w*)+)(?:\(.*\))?$')
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
        from _lumascope.modules import run_outcome, wire_encoding
        from _lumascope.rest.problems import _SERVER_STATUS
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
        from _lumascope.modules.api_surface import is_engineering

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
            if path.endswith('/') and not method:
                # A prefix: the routes under it, as "every route under /api/v1/scope/" says.
                prefixed = any(p.startswith(path) for p, _ in self.routes)
                return 'route', None if prefixed else 'no route is under it'
            answering = [m for p, m in self.routes if _route_matches(p, path)]
            if not answering:
                return 'route', 'no route answers it'
            if method and not any(method in m for m in answering):
                return 'route', f'no route answers it to {method}'
            return 'route', None
        if (match := _CLASS_MEMBER.match(token)) and match.group(1) in self.universe:
            cls, name = self.universe[match.group(1)], match.group(2)
            item = self.marked.get(f'{cls.__name__}.{name}')
            if item is not None and (item.field or not is_engineering(item.value)):
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
    from _lumascope.rest.app import build_app

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

    def test_every_prose_name_resolves(self, guide, prose_owners):
        failures = [
            f for path, text in guide for f in prose_name_failures(text, path, prose_owners)[0]
        ]
        assert not failures, (
            'The guide names something in its prose that does not resolve: a '
            'member or field that is not API, a route no route answers, a settings path '
            'that is no setting, or a quoted reason no published outcome carries. Correct '
            'the DOCUMENT.\n' + '\n'.join(failures)
        )

    def test_the_check_reaches_every_kind(self, guide, prose_owners):
        counts = sum(
            (prose_name_failures(text, path, prose_owners)[1] for path, text in guide),
            collections.Counter(),
        )
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


# Polarity 7: the docs beyond the guide teach only the surface.
#
# The guide's own checks above hold the guide. The plugin tutorial, the
# examples' README, the manual and the rendered reference teach as well, and
# a name they teach is relied on as the guide's is: so each of them is read
# by the same two readers -- the fenced call forms and the prose names -- and
# judged by the same owners, with an engineering member off the surface. The
# examples' code is not read here: `tests/test_api_examples.py` runs each
# `.py` and its `.sh` twin on a simulated server, and
# `tests/guards/test_no_client_reads_an_unmarked_member.py` counts every read
# its code makes round the surface, receivers followed through its types.
#
# Exempt: the tutorial's level table, a changelog (a row that taught an
# internal is superseded by a later row saying it is not API, never
# rewritten); and the one sentence that says `_lumascope` is internal.
_TUTORIAL = 'docs/plugin-tutorial.md'
_LEVEL_TABLE_SECTION = '## The plugin API level'
_INTERNAL = re.compile(r'\b_lumascope\b')
# The kinds of prose name judged here: a member and a route. A settings path
# is the settings' vocabulary, not a member, and a quoted value in the
# reference is as often a value as a reason.
_TAUGHT_KINDS = Words({'member', 'route'})
_SAYS_IT_IS_INTERNAL = (
    'docs/guide/concepts.md',
    'under `_lumascope` is internal and may change in any release.',
)

# The rows the docs teach today, by (file, the name that is not on the
# surface): each leaves in the commit that rewrites its line, and none is
# added. Drained to empty before 4.0.
_TAUGHT_SEED: dict[tuple[str, str], int] = {
    ('docs/api-reference.md', 'ActivityClaim.holder'): 1,
    ('docs/api-reference.md', 'CameraAPI.active'): 1,
    ('docs/api-reference.md', 'CameraAPI.get_held_channel'): 1,
    ('docs/api-reference.md', 'CapturedStill.significant_bits'): 3,
    ('docs/api-reference.md', 'Clock.schedule_once'): 3,
    ('docs/api-reference.md', 'FrameRecord.blown'): 1,
    ('docs/api-reference.md', 'FrameValidity.SKIP_FRAMES'): 1,
    ('docs/api-reference.md', 'FrameValidity.check_timing_record'): 1,
    ('docs/api-reference.md', 'FrameValidity.judge'): 1,
    ('docs/api-reference.md', 'FrameValidity.judgments'): 1,
    ('docs/api-reference.md', 'FrameValidity.motion_settled'): 2,
    ('docs/api-reference.md', 'FrameValidity.owed'): 1,
    ('docs/api-reference.md', 'Lumascope.disconnect'): 2,
    ('docs/api-reference.md', 'Lumascope.initialize'): 1,
    ('docs/api-reference.md', 'MetricsLogger.start'): 1,
    ('docs/api-reference.md', 'MissingPart.XY_STAGE'): 1,
    ('docs/api-reference.md', 'MissingPart.channel'): 1,
    ('docs/api-reference.md', 'ObjectiveLoader.get_objective_info'): 1,
    ('docs/api-reference.md', 'ObjectiveLoader.objectives'): 1,
    ('docs/api-reference.md', 'Operation.all_of'): 3,
    ('docs/api-reference.md', 'PluginRegistry.all_health'): 1,
    ('docs/api-reference.md', 'PluginRegistry.run_protocol_complete_processors'): 1,
    ('docs/api-reference.md', 'Protocol.adopt_focus_from'): 1,
    ('docs/api-reference.md', 'Protocol.apply_tiling'): 1,
    ('docs/api-reference.md', 'Protocol.from_file'): 1,
    ('docs/api-reference.md', 'Protocol.to_file'): 1,
    ('docs/api-reference.md', 'ProtocolRunner.run_X'): 1,
    ('docs/api-reference.md', 'ProtocolsAPI.load_protocol'): 3,
    ('docs/api-reference.md', 'ProtocolsAPI.step_targets'): 1,
    ('docs/api-reference.md', 'SequencedCaptureRunner.state_refusal'): 1,
    ('docs/api-reference.md', 'WellPlateLoader.resolve_plate_key'): 1,
    ('docs/plugin-tutorial.md', 'PluginSpec.subscribes_to'): 1,
    ('docs/plugin-tutorial.md', 'RESTRegistry.register'): 2,
}


def _without_level_table(text):
    """The tutorial with its level table's rows blanked, every line kept in place."""
    lines, inside = [], False
    for line in text.splitlines():
        if line.startswith('## '):
            inside = line == _LEVEL_TABLE_SECTION
        lines.append('' if inside and line.startswith('|') else line)
    return '\n'.join(lines)


def taught_doc_labels(root=None):
    """Every published doc that teaches the API beyond the guide, repo-relative."""
    return [
        _TUTORIAL,
        'docs/api_examples/README.md',
        *tracked_files(('docs/manual',), '*.qmd', root=root),
        'docs/api-reference.md',
    ]


def taught_docs(root=None):
    """``[(label, text), ...]`` for each of ``taught_doc_labels``, the tutorial's level table blanked."""
    root = REPO_ROOT if root is None else root
    docs = []
    for label in taught_doc_labels(root):
        text = (root / label).read_text(encoding='utf-8')
        docs.append((label, _without_level_table(text) if label == _TUTORIAL else text))
    return docs


def internal_mentions(text, label):
    """``(lineno, '_lumascope')`` per line of ``text`` naming the internal package, the exempt sentence aside."""
    exempt_label, exempt_line = _SAYS_IT_IS_INTERNAL
    return [
        (lineno, '_lumascope')
        for lineno, line in enumerate(text.splitlines(), 1)
        if _INTERNAL.search(line) and not (label == exempt_label and exempt_line in line)
    ]


def off_surface_rows(text, label, owners):
    """``(lineno, name, why)`` per name ``text`` teaches that is not on the surface.

    ``name`` is what the row is counted by: the member that is not API, the
    prefix that does not resolve, or the prose token of another kind.
    """
    rows = []
    for receiver, chain, lineno in _fenced_call_forms(text):
        root = owners.live[receiver]
        missing = _first_missing_attr(root, receiver, chain)
        step = None if missing else _first_unmarked_step(root, receiver, chain)
        if missing:
            rows.append((lineno, missing, 'does not resolve'))
        elif step:
            rows.append((lineno, step, 'is not API'))
    for token, lineno in prose_tokens(text):
        judged = owners.judge(token)
        if judged is None or judged[0] not in _TAUGHT_KINDS or judged[1] is None:
            continue
        kind, why = judged
        name = token
        if kind == 'member' and why.endswith(' is not API'):
            name = why.removesuffix(' is not API')
        rows.append((lineno, name, why))
    rows += [
        (lineno, name, 'names the internal package')
        for lineno, name in internal_mentions(text, label)
    ]
    return rows


class TestTheDocsTeachOnlyTheSurface:
    """Polarity 7: what the published docs teach beyond the guide is the surface."""

    def test_no_doc_teaches_a_name_off_the_surface(self, prose_owners):
        found, lines = collections.Counter(), collections.defaultdict(list)
        for label, text in taught_docs():
            for lineno, name, why in off_surface_rows(text, label, prose_owners):
                found[label, name] += 1
                lines[label, name].append(f'  {label}:{lineno}  {name}  -> {why}')
        for path, text in guide_texts():
            label = path.relative_to(REPO_ROOT).as_posix()
            for lineno, name in internal_mentions(text, label):
                found[label, name] += 1
                lines[label, name].append(
                    f'  {label}:{lineno}  {name}  -> names the internal package'
                )
        seed = collections.Counter(_TAUGHT_SEED)
        new, gone = found - seed, seed - found
        assert not new, (
            'A published doc teaches a name that is not on the surface: a member not '
            'marked @api (or marked engineering), a name that does not resolve, a route '
            'no route answers, or the internal package. Teach the marked member, or say in '
            'words what the API does.\n'
            + '\n'.join(line for key in sorted(new) for line in lines[key])
        )
        assert not gone, (
            'A row of the seed is no longer taught: lower its count in _TAUGHT_SEED in '
            'this commit.\n'
            + '\n'.join(f'  {key}: {seed[key]} -> {found[key]}' for key in sorted(gone))
        )

    def test_the_walk_judges_names_in_each_kind_of_doc(self, prose_owners):
        judged = collections.Counter()
        for label, text in taught_docs():
            judged[label] += sum(1 for token, _ in prose_tokens(text) if prose_owners.judge(token))
            judged[label] += sum(1 for _ in _fenced_call_forms(text))
        for label in (_TUTORIAL, 'docs/api-reference.md'):
            assert judged[label], f'the walk judged no name in {label}: {dict(judged)}'
        assert any(label.startswith('docs/manual/') for label in judged), (
            f'the walk read no manual chapter: {sorted(judged)}'
        )

    def test_a_planted_doc_reports_each_kind_and_not_the_level_table(self, prose_owners):
        planted = (
            '## A chapter\n'
            'Call `ImagingAPI.capture_still` and `ctx.scope.motion.get_current_position()`.\n'
            '```python\n'
            'ctx.scope.imaging.capture_still()\n'
            '```\n'
            'Import `_lumascope.modules.scope_session` if you must.\n'
            'Every route under `/api/v1/scope/camera/`, and none under `/api/v1/nowhere/`.\n'
            'Right: `ScopeSession.capture_still`, `ctx.scope.motion.axis_positions()`.\n'
            f'{_LEVEL_TABLE_SECTION}\n'
            '| Level | Added |\n'
            '|---|---|\n'
            '| level 1 | `CameraAPI.active` |\n'
        )
        rows = off_surface_rows(_without_level_table(planted), _TUTORIAL, prose_owners)
        assert sorted(rows) == [
            (2, 'ImagingAPI.capture_still', 'ImagingAPI.capture_still is not API'),
            (2, 'MotionAPI.get_current_position', 'MotionAPI.get_current_position is not API'),
            (4, 'ImagingAPI.capture_still', 'is not API'),
            (6, '_lumascope', 'names the internal package'),
            (7, '/api/v1/nowhere/', 'no route is under it'),
        ], rows

    def test_only_the_sentence_that_says_the_package_is_internal_is_exempt(self):
        label, line = _SAYS_IT_IS_INTERNAL
        assert internal_mentions(line, label) == []
        assert internal_mentions(line, _TUTORIAL) == [(1, '_lumascope')]
        assert line in (REPO_ROOT / label).read_text(encoding='utf-8'), (
            f'{label} no longer holds the sentence its exemption names'
        )


_LINK = re.compile(r'\]\(([^)\s]+)\)')


def anchors_of(text):
    """Every anchor a markdown file holds: its ``<a id>`` tags and its headings' slugs."""
    ids = set(re.findall(r'<a id="([^"]+)"></a>', text))
    for heading in re.findall(r'^#{1,6} (.+)$', text, re.MULTILINE):
        slug = re.sub(r'[^\w\- ]', '', heading.strip().lower()).replace(' ', '-')
        ids.add(slug)
    return ids


def published_docs():
    """Every document LumaViewPro publishes: the repo's README and each markdown file under ``docs/``."""
    return [
        REPO_ROOT / 'README.md',
        *[REPO_ROOT / rel for rel in tracked_files(('docs',), '**/*.md')],
    ]


def link_failures(path, root):
    """A line per relative link in ``path`` to a file or anchor that is not there.

    A link names a file beside ``path``, or none for ``path`` itself; an anchor
    is checked in the markdown file it names. Labelled relative to ``root``.
    """
    label = path.relative_to(root).as_posix()
    text = path.read_text(encoding='utf-8')
    failures = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for link in _LINK.findall(line):
            if urllib.parse.urlsplit(link).scheme:
                continue
            page, sep, anchor = link.partition('#')
            target = (path.parent / page).resolve() if page else path
            if not target.exists():
                failures.append(f'  {label}:{lineno}  {link}  -> no such file')
            elif (
                sep
                and target.suffix == '.md'
                and anchor not in anchors_of(target.read_text(encoding='utf-8'))
            ):
                failures.append(f'  {label}:{lineno}  {link}  -> no such anchor')
    return failures


class TestEveryPublishedLinkResolves:
    """Polarity 6: a link in a published document reaches its file, and its anchor."""

    @pytest.mark.slow
    def test_every_link_reaches_its_file_and_anchor(self):
        failures = [f for doc in published_docs() for f in link_failures(doc, REPO_ROOT)]
        assert not failures, (
            'A published document links to a file or an anchor that is not there. A '
            'reference anchor is a member, field or class the generator rendered, or one '
            'of its headings.\n' + '\n'.join(failures)
        )

    def test_a_dead_link_is_reported(self, tmp_path):
        (tmp_path / 'sub').mkdir()
        (tmp_path / 'api-reference.md').write_text(
            '<a id="ScopeSession.create"></a>\n## Events\n', encoding='utf-8'
        )
        planted = tmp_path / 'guide.md'
        planted.write_text(
            '## Here\n'
            'See [create](api-reference.md#ScopeSession.create), [gone](api-reference.md#'
            'BringUpRecord.substitution), [here](#here), [there](#there) and '
            '[events](api-reference.md#events).\n'
            '[moved](OldName.md), [up](sub/../api-reference.md), [site](https://etaluma.com).\n',
            encoding='utf-8',
        )
        assert link_failures(planted, tmp_path) == [
            '  guide.md:2  api-reference.md#BringUpRecord.substitution  -> no such anchor',
            '  guide.md:2  #there  -> no such anchor',
            '  guide.md:3  OldName.md  -> no such file',
        ]
