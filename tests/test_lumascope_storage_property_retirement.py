# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Wave 7 Phase 2c.5 lock-in: storage @property/setter pairs on Lumascope stay retired.

History
-------
During Wave 7 Phase 2c, motion state-storage fields migrated from
Lumascope to MotionAPI. To unblock incremental test migration, a
temporary band-aid layer of @property/setter pairs on Lumascope
forwarded reads/writes to self.motion.<field>. Phase 2c.5 retires
those pairs once all in-tree test callers use the canonical
scope.motion.<field> surface.

Retired pairs (14 storage fields x 2 = 28 method defs):
- _pos_cache, _pos_cache_lock
- _axis_state, _axis_state_lock
- _arrival_events
- _move_profile, _move_profile_lock
- _position_listeners, _position_listeners_lock
- _motion_wake
- _motion_monitor_stop, _motion_monitor_thread
- _homing_event, _turreting_event

Re-introduction of any of these as @property/setter on Lumascope means
the band-aid came back -- callers should reach through `scope.motion.X`
or be migrated to MotionAPI directly.

Out of scope (intentionally still on Lumascope, retire later):
- is_homing / is_turreting public @property pair (lines 881-899; Phase 2e/2f)
- method-name forwarders zhome / home / thome / get_current_position /
  move_absolute_position etc. (lines 3974-4090; Phase 2e/2f)
- get_axis_state / _set_axis_state method-name forwarders (kept for now;
  production callers still use them directly on Lumascope).
"""

from __future__ import annotations

import ast
import pathlib


REPO = pathlib.Path(__file__).resolve().parent.parent
LUMASCOPE_SRC = REPO / 'modules' / 'lumascope_api' / '_lumascope.py'


RETIRED_STORAGE_PROPERTIES = (
    '_pos_cache',
    '_pos_cache_lock',
    '_axis_state',
    '_axis_state_lock',
    '_arrival_events',
    '_move_profile',
    '_move_profile_lock',
    '_position_listeners',
    '_position_listeners_lock',
    '_motion_wake',
    '_motion_monitor_stop',
    '_motion_monitor_thread',
    '_homing_event',
    '_turreting_event',
)


def _lumascope_class_node() -> ast.ClassDef:
    source = LUMASCOPE_SRC.read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'Lumascope':
            return node
    raise AssertionError('class Lumascope not found in _lumascope.py')


def _property_names(class_node: ast.ClassDef) -> set[str]:
    """Return set of method names that have @property OR a @<name>.setter
    decorator. Catches both halves of a property/setter pair."""
    names = set()
    for child in class_node.body:
        if not isinstance(child, ast.FunctionDef):
            continue
        for dec in child.decorator_list:
            # @property
            if isinstance(dec, ast.Name) and dec.id == 'property':
                names.add(child.name)
            # @<name>.setter
            if isinstance(dec, ast.Attribute) and dec.attr == 'setter':
                names.add(child.name)
    return names


class TestStoragePropertyRetirement:
    """Phase 2c.5 lock-in: the 14 retired storage @property/setter pairs
    must stay gone from Lumascope. Each name re-appearing on Lumascope as
    a property or setter means the 2c band-aid forwarder pattern returned."""

    def test_no_retired_storage_property_resurrected(self):
        cls = _lumascope_class_node()
        present = _property_names(cls)
        leaked = sorted(set(RETIRED_STORAGE_PROPERTIES) & present)
        assert not leaked, (
            f'The following storage fields are back as @property on '
            f'Lumascope: {leaked}. Per Wave 7 Phase 2c.5, callers reach '
            f'these via scope.motion.<field>; do not re-introduce the '
            f'band-aid forwarders. See class docstring for the full list '
            f'and the public-API exceptions that still belong here.'
        )
