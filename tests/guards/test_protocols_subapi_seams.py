# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Seams for the `scope.protocols` sub-API.

Three things are locked here:

1. Both Lumascope constructor paths expose the SAME sub-API roster.
   `__init__` and `create_diagnostic` open-code their wiring separately,
   so a sub-API added to one and forgotten in the other raises
   AttributeError only on the diagnostic path -- which nobody exercises
   until a customer needs a support report.
2. `source_path` is stored in exactly one place, on the scope, given at
   its construction; ProtocolsAPI reads it and keeps no copy.
3. The migration itself: production callers reach the cluster through
   `scope.protocols` (Guard A) and Lumascope no longer carries it
   (Guard B). Both are xfail-strict until their migration stage lands,
   so they flip LOUDLY -- an unexpected pass fails the suite -- rather
   than quietly going green while sites remain unmigrated.
"""

from __future__ import annotations

import ast
import pathlib

from tests.ast_seams import REPO_ROOT, find_def, iter_package_modules, parse_module
from tests.scope_fakes import build_scope

_LUMASCOPE_SRC = 'modules/lumascope_api/_lumascope.py'
_PROTOCOLS_SRC = 'modules/lumascope_api/protocols.py'

# Every sub-API both constructor paths must wire.
_SUB_APIS = (
    'motion',
    'illumination',
    'imaging',
    'diagnostics',
    'capabilities',
    'io',
    'protocols',
    'runtime_state',
)

# Members moving off Lumascope onto ProtocolsAPI. sanitize_step_name is
# on this list because it retires rather than moves -- its callers go
# direct to Protocol, which is already its canonical home.
_MOVED_MEMBERS = frozenset(
    {
        'load_protocol',
        'create_protocol',
        'register_source_path',
        'sanitize_step_name',
    }
)

_RETIRED_FROM_LUMASCOPE = (
    'load_protocol',
    'create_protocol',
    'register_source_path',
    'tiling_configs_path',
    'sanitize_step_name',
)


def _new_scope():

    return build_scope(simulate=True, register_atexit=False)


def _legacy_call_sites():
    """Yield `(rel_path, lineno, attr)` for each surviving legacy call site.

    Matches BOTH shapes the call sites actually take: an attribute chain
    (`ctx.scope.load_protocol`, `lumaview.scope.register_source_path`) and
    a bare local named `scope` (`scope.register_source_path`, which is how
    ScopeSession calls it). Matching only the former would let the two
    ScopeSession sites migrate-by-omission and take the guard green with
    the work undone.

    Deliberately does NOT match `self._scope.X` inside the sub-API, nor
    `<widget>.load_protocol` -- ProtocolSettings has a same-named UI
    method that is not this API.
    """
    sources = list(iter_package_modules(('modules', 'ui')))
    sources.append(('lumaviewpro.py', parse_module('lumaviewpro.py')))

    for rel_path, tree in sources:
        if rel_path in (_PROTOCOLS_SRC, _LUMASCOPE_SRC):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or node.attr not in _MOVED_MEMBERS:
                continue
            value = node.value
            reaches_scope = (isinstance(value, ast.Attribute) and value.attr == 'scope') or (
                isinstance(value, ast.Name) and value.id == 'scope'
            )
            if reaches_scope:
                yield (rel_path, node.lineno, node.attr)


class TestSubApiRoster:
    """Both constructor paths wire an identical sub-API roster."""

    def test_init_path_wires_every_sub_api(self):
        scope = _new_scope()
        try:
            missing = [name for name in _SUB_APIS if not hasattr(scope, name)]
        finally:
            scope.disconnect()
        assert not missing, f'__init__ did not wire: {missing}'

    def test_diagnostic_path_wires_every_sub_api(self, diagnostic_scope):
        missing = [name for name in _SUB_APIS if not hasattr(diagnostic_scope, name)]
        assert not missing, (
            f'create_diagnostic did not wire: {missing}. The two constructor '
            f'paths wire sub-APIs separately; both must be updated together.'
        )


class TestSourcePathHasOneHome:
    """source_path is stored on the scope and nowhere else; ProtocolsAPI reads it."""

    def test_slot_lives_on_the_sub_api(self):
        scope = _new_scope()
        try:
            assert not hasattr(scope.protocols, '_source_path'), (
                'ProtocolsAPI must not carry its own source_path; a second copy '
                'is a store that can disagree with the scope.'
            )
            assert not hasattr(scope.protocols, 'register_source_path')
        finally:
            scope.disconnect()

    def test_composition_root_holds_no_copy(self):
        scope = _new_scope()
        try:
            assert 'source_path' in vars(scope), 'the scope holds its data folder'
            assert '_source_path' not in vars(scope), (
                'Lumascope must carry its data folder once; a second copy '
                'is a store that can disagree with the first.'
            )
        finally:
            scope.disconnect()


class TestConstructorsRefuseToGuess:
    """The folder the sub-API resolves against, asserted against the sub-API
    directly so it survives the forwarder retirement unchanged."""

    def test_registration_resolves_the_tiling_config(self):
        # The folder is given at construction; the sub-API resolves from it.
        root = REPO_ROOT
        scope = build_scope(simulate=True, register_atexit=False, source_path=root)
        try:
            assert pathlib.Path(scope.source_path) == root
            assert scope.protocols.tiling_configs_path() == root / 'data' / 'tiling.json'
        finally:
            scope.disconnect()


def test_no_caller_reaches_the_cluster_through_scope():
    survivors = sorted(_legacy_call_sites())
    assert not survivors, 'still reaching the protocol cluster via scope: ' + ', '.join(
        f'{path}:{line} .{attr}' for path, line, attr in survivors
    )


def test_lumascope_no_longer_carries_the_protocol_cluster():
    survivors = [
        name
        for name in _RETIRED_FROM_LUMASCOPE
        if find_def(_LUMASCOPE_SRC, name, class_name='Lumascope') is not None
    ]
    assert not survivors, f'still defined on Lumascope: {survivors}'
