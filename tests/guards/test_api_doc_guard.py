"""Structural guard: LumascopeSkills.md call forms resolve and bind.

Every call form an L2 reader could copy out of the reference must resolve on
a real object (polarity 1) AND accept the arguments the reference passes it
(polarity 2), so a doc line and the surface it describes cannot drift apart
silently. Resolution alone is not enough: a member can exist and still reject
the documented call, which is how the reference came to publish
``fv.count_frame(chunk_data=None, frame_ts=None)`` against a signature that
had neither parameter -- a TypeError on a line every test called clean. When this test goes red the DOC is
what gets corrected -- the published reference follows the API, never the
reverse. Adding a method because a doc line names it is the failure mode this
guard exists to make visible, not to license.

It replaces the per-defect string assertions that accumulated in
test_audit_fixes.py (one `assert 'scope.pixel_size()' not in doc` per audit
finding). Those pinned individual known-bad names; this checks the whole
document against the live object, so the NEXT drift is caught without anyone
having to notice it first. The older assertions are kept: they also pin that
the canonical replacement surface is still DOCUMENTED, which is a presence
claim this guard does not make.

Scope, and why it is not the whole file
---------------------------------------
Only fenced code blocks are checked -- ``python``-labelled and unlabelled.
LumascopeSkills.md contains, by design, sections whose job is to name
surfaces that do NOT resolve:

* the Changelog, which records ``removal`` and ``rename`` entries against the
  frozen 4.x surface;
* retirement and forward-looking notes in prose ("was previously available as
  ``scope.compute_focus_score(image)``; retired in Wave 7", and the
  explanation that ``session.led_on(...)`` deliberately does not exist
  without a suffix).

A whole-file resolution check would need a hand-maintained allowlist of those
mentions -- a mirror requiring manual sync, which is the defect class this
guard is meant to close, not reproduce. Code fences carry the contract the
guard is actually protecting: a reader copies from an example block.

``PLUGIN_API_DESIGN_2026-05-09.md`` is deliberately NOT guarded. It is a
design and history document; 12 of its 40 call forms name proposals that were
never built (``scope.is_command_safe_for_rest``,
``session.focus.get_well_focus``) or pre-rename historical spellings. Its
accuracy is maintained by review, not by resolution.
"""

import ast
import inspect
import re
import warnings

import pytest

from tests.ast_seams import REPO_ROOT
from tests.settings_fixtures import complete_settings

DOC = REPO_ROOT / 'docs' / 'LumascopeSkills.md'

# Receivers the reference uses for the live objects an L2 caller holds.
# `scope` is the Lumascope composition root, `session` the ScopeSession L2
# entry point. Both are resolved against real instances rather than classes:
# `session.scope` is an instance attribute, so a class-level hasattr would
# report every `session.scope.*` form in the document as unresolved.
#
# `caps` is an alias the reference establishes in its own example block
# (`caps = scope.capabilities` opens the scope.capabilities section) and then
# uses for 22 copyable forms. Without it those lines -- the whole structure
# report, which is also the REST /capabilities payload -- would sit outside
# the guard.
_CALL_FORM = re.compile(r'\b(scope|session|caps)((?:\.[A-Za-z_][A-Za-z0-9_]*)+)')

# Fence languages whose contents are checked. The empty string is an
# unlabelled ``` block; the reference uses those for python too, and they
# resolve today, so including them costs nothing and closes the gap where a
# forgotten language tag would silence the guard. A fence labelled for another
# language would be skipped -- its syntax is not ours to resolve -- but the
# reference has none today: all 65 fences are python or unlabelled.
_CHECKED_FENCES = ('', 'python')


def _fenced_call_forms(text):
    """Extract (receiver, chain, lineno) for every call form in a code fence.

    Yields one entry per occurrence, not per distinct form, so the failure
    message can name every line a reader would have copied from.
    """
    fence = None
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.startswith('```'):
            fence = line[3:].strip() if fence is None else None
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
                fence, lines, start = line[3:].strip(), [], lineno + 1
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
_CONSTRUCTOR_RECEIVERS = ('ScopeSession',)

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
