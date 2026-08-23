# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Polarity 2 of the API doc guard: every public member is accounted for.

Polarity 1 (test_api_doc_guard.py) checks that every call form in
LumascopeSkills.md resolves on the live API -- the doc cannot invent
surface. This guard checks the OTHER direction: every public member on
the live surface is either

  * DOCUMENTED -- its name appears as an attribute reference inside a
    checked code fence of LumascopeSkills.md (the copyable contract),
  * INTERNAL -- its docstring carries the internal marking phrase
    ("not part of the L2 API surface", whitespace-normalized), or
  * RULED -- listed below with the ruling that keeps it public-named
    but off the documented surface.

An undocumented, unruled public member is a test failure: the next
member added to a sub-API must be documented, marked, or ruled before
it ships -- surface can no longer grow silently.

The universe is ONE live enumeration (a simulated scope), not a parsed
class list, because the sub-API objects are composed at runtime. The
same enumeration core is imported by the private repo's census tooling,
so the instrument and this guard cannot drift apart. The SUBS list is
hand-maintained: a NEW sub-API must be added to it (and a new
lease-like collaborator class to EXTRA_CLASSES) or its members escape
the guard. Enumeration is sim-only by construction -- driver-contributed
members that only exist on real hardware are outside its reach.
"""

import re
import pathlib

DOC = pathlib.Path(__file__).resolve().parents[1] / 'docs' / 'LumascopeSkills.md'

# Hand-maintained sub-API roster -- add every new sub-API here, or its
# members escape this guard entirely.
SUBS = [
    'motion',
    'illumination',
    'imaging',
    'diagnostics',
    'protocols',
    'io',
    'runtime_state',
    'capabilities',
]

INTERNAL_PHRASE = 'not part of the L2 API surface'

# Members ruled public-named but off the documented surface. FED FROM
# THE RULING STORE, never hand-invented: the capability table
# (Firmware/docs/R2_CAPABILITY_TABLE_2026-08-18.md) is the authority;
# refresh by re-reading its KEEP-ENG / HOLD rows (the S10.3 rulings
# section) and reconciling this literal against them. A hardcoded copy
# that drifts from the table is the incident that created this rule.
RULED = {
    # KEEP-ENG (engineering surface; documented in ENGINEERING_PLUGIN_NOTES.md)
    'imaging.suppress_value_warnings',
    'diagnostics.run_grab_lifecycle_benchmark',
    'diagnostics.run_camera_bandwidth_test',
    'motion.set_precision_mode',
    # HOLD: KEEP ruling, doc row deferred until the software-AE work
    # exercises it (S10.3 Q-1)
    'imaging.set_auto_exposure_time',
    # INTERNAL by ruling, but an instance ATTRIBUTE -- no docstring
    # surface to mark (S10.3 mini-batch; session owns its lifecycle via
    # the documented start_metrics/stop_metrics)
    'Lumascope.metrics_logger',
}


def load_surface():
    """The live public surface, keyed '<receiver>.<member>'.

    Introspected from a simulated scope rather than parsed, because the
    sub-API objects are composed at runtime; a static read of the class
    bodies misses members a driver or mixin contributes.
    """
    import modules.lumascope_api as la
    from modules.lumascope_api.illumination import LedLease
    from modules.scope_session import ScopeSession

    scope = la.Lumascope(simulate=True, register_atexit=False, register_metrics=False)
    surface = {}
    for sub in SUBS:
        obj = getattr(scope, sub, None)
        if obj is None:
            continue
        for m in dir(obj):
            if not m.startswith('_'):
                surface[f'{sub}.{m}'] = obj
    for m in dir(scope):
        if not m.startswith('_') and m not in SUBS:
            surface[f'Lumascope.{m}'] = scope
    extra_classes = {'ScopeSession': ScopeSession, 'LedLease': LedLease}
    for cls_name, cls in extra_classes.items():
        for m in dir(cls):
            if not m.startswith('_'):
                surface[f'{cls_name}.{m}'] = cls
    return surface


def _docstring(owner, name):
    """The member's own docstring -- class-dict first so property and
    classmethod docs are found, and a plain data attribute (whose VALUE
    would otherwise contribute its type's docstring) reads as empty."""
    klass = owner if isinstance(owner, type) else type(owner)
    for k in klass.__mro__:
        if name in k.__dict__:
            return getattr(k.__dict__[name], '__doc__', None) or ''
    attr = getattr(owner, name, None)
    if callable(attr):
        return getattr(attr, '__doc__', None) or ''
    return ''


def test_every_public_member_is_documented_marked_or_ruled():
    doc = DOC.read_text(encoding='utf-8')
    # Checked fences only -- python-labelled and unlabelled -- matching
    # polarity 1's scope: prose may name retired or future members.
    fences = '\n'.join(re.findall(r'```(?:python)?\n(.*?)```', doc, re.S))

    failures = []
    for key, owner in sorted(load_surface().items()):
        name = key.split('.', 1)[1]
        if key in RULED:
            continue
        if re.search(r'\.' + re.escape(name) + r'\b', fences):
            continue
        # Whitespace-normalized: docstrings wrap the phrase across lines.
        if INTERNAL_PHRASE in ' '.join(_docstring(owner, name).split()):
            continue
        failures.append(key)

    assert not failures, (
        f'{len(failures)} public member(s) neither documented in a '
        f'LumascopeSkills.md code fence, internal-marked, nor ruled: '
        f'{failures}. Document it, mark its docstring '
        f'("{INTERNAL_PHRASE}"), or record its ruling in the capability '
        f'table and this list.'
    )


def test_ruled_list_members_exist():
    """A RULED entry naming a dead member is stale -- the list must track
    the live surface it exempts."""
    surface = load_surface()
    stale = [k for k in RULED if k not in surface]
    assert not stale, f'RULED entries no longer on the live surface: {stale}'
