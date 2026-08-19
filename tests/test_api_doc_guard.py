"""Structural guard: LumascopeSkills.md call forms resolve on the live API.

Polarity 1 of the API doc guard. Every call form an L2 reader could copy out
of the reference must resolve on a real object, so a doc line and the surface
it describes cannot drift apart silently. When this test goes red the DOC is
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
  ``scope.compute_focus_score(image)``; retired in Wave 7", "a future release
  may add a public ``scope.imaging.start_grabbing()``", and the explanation
  that ``session.led_on(...)`` deliberately does not exist without a suffix).

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

import pathlib
import re
import warnings

import pytest

DOC = pathlib.Path('docs/LumascopeSkills.md')

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
# forgotten language tag would silence the guard. A labelled non-python fence
# (the MATLAB REST example) is skipped: its syntax is not ours to resolve.
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


@pytest.fixture(scope='module')
def live_objects():
    """Every receiver the reference documents, as live instances.

    Module-scoped: the headless session builds the full executor topology,
    which is ~0.5s and pointless to repeat per test.
    """
    warnings.simplefilter('ignore', FutureWarning)
    from modules.lumascope_api import Lumascope
    from modules.scope_session import ScopeSession

    scope = Lumascope(simulate=True, register_atexit=False, register_metrics=False)
    session = ScopeSession.create_headless()
    yield {'scope': scope, 'session': session, 'caps': scope.capabilities}
    session.shutdown()


def _first_missing_attr(root, receiver, chain):
    """The shortest prefix of the chain that does not resolve, or None."""
    obj = root
    for depth, attr in enumerate(chain):
        if not hasattr(obj, attr):
            return '.'.join((receiver, *chain[: depth + 1]))
        obj = getattr(obj, attr)
    return None


class TestLumascopeSkillsCallFormsResolve:
    """Polarity 1: documented call forms exist on the live API surface."""

    def test_every_fenced_call_form_resolves(self, live_objects):
        text = DOC.read_text(encoding='utf-8')
        failures = []
        for receiver, chain, lineno in _fenced_call_forms(text):
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

    def test_guard_actually_reaches_the_reference(self, live_objects):
        """The extractor is wired to real content, not silently finding nothing.

        A regex or fence-tracking regression would make the check above pass
        vacuously. Every documented receiver is exercised in the reference
        today, so a run that misses one means the extractor broke, not that
        the doc got smaller.
        """
        receivers = {r for r, _, _ in _fenced_call_forms(DOC.read_text(encoding='utf-8'))}
        assert receivers == {'scope', 'session', 'caps'}, (
            f'expected call forms for every documented receiver, saw {receivers or "none"} -- '
            'the fence tracker or the call-form pattern has regressed'
        )
