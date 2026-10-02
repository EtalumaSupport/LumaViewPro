# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every edit the GUI makes to a protocol is counted, so the count can only fall.

The standing goal is that a new GUI can be written on the API alone. A
protocol edit the GUI makes by calling the ``Protocol`` object's writer
itself -- deleting a step, renaming it, writing a Z, tiling -- is an edit a
second GUI would have to rebuild, with every decision the first one makes
around it. The architecture ratchets do not see these: the ``Protocol`` is a
public object, so a call on it is neither a private reach nor an import, and
two such calls stopped working for a fortnight without a test noticing.

Counted per file and member, from ``ui/`` and ``lumaviewpro.py``:

* a call of a ``Protocol`` writer's name on any receiver but ``self`` -- the
  GUI holds its protocols under several names (``self._protocol``,
  ``protocol``, a run's ``sequence``), while the panel's own handlers of the
  same names (``delete_step``, ``insert_step``) are called on ``self``;
* an assignment into the frame ``steps()`` returns, which is the protocol's
  live frame, so the write lands with no writer called at all.

The pin is an EQUALITY. A rise is a new edit in the GUI: put it behind the
API instead. A fall is the migration working: lower the pin in the same
commit. Every public ``Protocol`` member is classified writer or not, so a
new member fails here until someone decides which it is.
"""

import ast

from tests import ratchets as _ratchets
from tests.ast_seams import iter_package_modules, parse_module

# Public members of Protocol that change the protocol.
WRITERS = frozenset(
    {
        'adopt_focus_from',
        'apply_focus_all_layer_steps',
        'apply_tiling',
        'apply_zstack_group_focus',
        'apply_zstacking',
        'delete_step',
        'insert_step',
        'modify_autofocus',
        'modify_autofocus_all_steps',
        'modify_capture_root',
        'modify_labware',
        'modify_name',
        'modify_step',
        'modify_step_z_height',
        'modify_time_params',
        'optimize_step_ordering',
    }
)

# Public members of Protocol that do not change it: readers, constructors,
# file output, and helpers.
NOT_WRITERS = frozenset(
    {
        'capture_root',
        'copy_for_execution',
        'create_empty',
        'duration',
        'estimate_write_mb',
        'from_config',
        'from_file',
        'labware',
        'layer_acquires',
        'layer_settings',
        'num_steps',
        'period',
        'sanitize_step_name',
        'size_advisory',
        'step',
        'step_list_revision',
        'steps',
        'to_file',
        'validate_for_run',
        'validate_steps',
        'valid_colors',
        'zstack_group_focus_anchor',
    }
)

_STEPS_WRITE = 'steps()[...] ='

# Pinned at the tree that first counted them (fifteen writer calls and one
# write into the live frame), lowered as each edit moved behind the API.
_PIN = {
    ('ui/layer_control.py', 'apply_focus_all_layer_steps'): 1,
    ('ui/layer_control.py', 'modify_step_z_height'): 1,
    ('ui/protocol_settings.py', 'apply_tiling'): 1,
    ('ui/protocol_settings.py', 'apply_zstacking'): 1,
    ('ui/protocol_settings.py', 'delete_step'): 1,
    ('ui/protocol_settings.py', 'modify_capture_root'): 1,
    ('ui/protocol_settings.py', 'modify_labware'): 1,
    ('ui/protocol_settings.py', 'modify_name'): 1,
    ('ui/protocol_settings.py', 'modify_time_params'): 3,
    ('ui/protocol_settings.py', 'optimize_step_ordering'): 2,
}


def _not_self(receiver: ast.expr) -> bool:
    return not (isinstance(receiver, ast.Name) and receiver.id == 'self')


def _is_steps_call(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'steps'
        and _not_self(node.func.value)
    )


def _edits_in(tree: ast.Module) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in WRITERS
            and _not_self(node.func.value)
        ):
            counts[node.func.attr] = counts.get(node.func.attr, 0) + 1
        elif isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                while isinstance(target, (ast.Subscript, ast.Attribute)):
                    target = target.value
                    if _is_steps_call(target):
                        counts[_STEPS_WRITE] = counts.get(_STEPS_WRITE, 0) + 1
                        break
    return counts


def _gui_protocol_edit_counts() -> dict[tuple[str, str], int]:
    modules = list(iter_package_modules(('ui',)))
    modules.append(('lumaviewpro.py', parse_module('lumaviewpro.py')))
    counts = {}
    for path, tree in modules:
        for member, n in _edits_in(tree).items():
            counts[(path, member)] = n
    return counts


def test_every_public_protocol_member_is_classified():
    tree = parse_module('modules/protocol.py')
    protocol = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Protocol')
    public = {
        n.name
        for n in protocol.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith('_')
    }
    assert not WRITERS & NOT_WRITERS
    assert public == WRITERS | NOT_WRITERS, (
        f'unclassified: {sorted(public - WRITERS - NOT_WRITERS)}; '
        f'no longer on Protocol: {sorted((WRITERS | NOT_WRITERS) - public)}'
    )


def test_the_gui_makes_no_new_protocol_edit():
    assert _gui_protocol_edit_counts() == _PIN


def test_the_census_sees_each_shape_of_edit():
    source = (
        'def f(self, protocol):\n'
        '    self._protocol.delete_step(step_idx=0)\n'
        '    protocol.modify_name(step_idx=0, step_name="a")\n'
        '    sequence.modify_autofocus_all_steps(enabled=True)\n'
        "    self._protocol.steps()['Z'] = 1\n"
        "    self._protocol.steps().loc[0, 'Z'] = 1\n"
        '    self.delete_step()\n'
        '    protocol.steps()\n'
    )
    assert _edits_in(ast.parse(source)) == {
        'delete_step': 1,
        'modify_name': 1,
        'modify_autofocus_all_steps': 1,
        _STEPS_WRITE: 2,
    }


_ratchets.register(
    'GUI: protocol edits made outside the API',
    lambda: sum(_gui_protocol_edit_counts().values()),
    sum(_PIN.values()),
    'equal',
)
