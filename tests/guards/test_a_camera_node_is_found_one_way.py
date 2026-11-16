# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A camera node is found one way: the driver's ``_node`` (``Camera._node``).

Whether a camera offers a node is decided in one place per driver, the
lookup on a nodemap in hand that ``_node`` calls: ``PylonCamera._node_in``
and ``IDSCamera._node_in``. pypylon's ``GetNode(name)`` answers a
placeholder for a node the map does not hold, which raises
``genicam.LogicalErrorException`` at its first read or write, and the
instant camera's attribute (``camera.Width``) answers the same placeholder;
only the map's ``Contains`` says whether the node is there. IDS's
``FindNode`` raises for an absent name, and only ``HasNode`` says. A lookup
anywhere else decides absence its own way: a body's exposure read raising
where it falls back to ``ExposureTime``, the idle wait at disconnect giving
up, a WARNING for a pixel pitch the camera does not state.

This walk over LVP's production tree (``src/``, ``ui/``, ``tools/``, the
root's modules) refuses, outside the two lookups:

- a ``GetNode`` call with an argument, ``FindNode``, ``HasNode`` or
  ``Contains`` (the no-argument ``GetNode()`` that turns a node handle into
  its node is not a lookup and passes);
- in ``PylonCamera``, a capitalised member reached on the instant camera
  (``self.active`` and any name bound to it) that is not one of the
  instant camera's own methods, and a ``getattr`` on it anywhere but
  ``PylonCamera._nodemap``, which takes a nodemap by its getter's name.

The setting writers the driver contract's Stage 3 commit C rewrites through
one write function per driver are the one exception list; C empties it.
"""

from __future__ import annotations

import ast

from tests.ast_seams import Words, iter_package_modules, parse_module, tracked_files

_PYLON = 'src/_lumascope/drivers/pyloncamera.py'
_IDS = 'src/_lumascope/drivers/idscamera.py'

_OWNERS = {(_PYLON, 'PylonCamera._node_in'), (_IDS, 'IDSCamera._node_in')}
_NODEMAP_GETTER = (_PYLON, 'PylonCamera._nodemap')

_LOOKUPS = Words({'FindNode', 'HasNode', 'Contains'})

# The instant camera's own methods, the members of the handle that are not
# nodes (pypylon ``InstantCamera``).
_INSTANT_CAMERA_METHODS = Words(
    {
        'GetNodeMap',
        'GetStreamGrabberNodeMap',
        'GetTLNodeMap',
        'GetInstantCameraNodeMap',
        'GetDeviceInfo',
        'IsGrabbing',
        'StartGrabbing',
        'StopGrabbing',
        'RegisterConfiguration',
        'RegisterImageEventHandler',
        'Open',
        'Close',
        'DetachDevice',
        'DestroyDevice',
        'StreamGrabber',
    }
)

# The setting writers commit C rewrites, each a sequence of calls through the
# driver's one write function; their lookups go with them.
_REWRITTEN_BY_C = {
    _PYLON: {
        'PylonCamera.connect',
        'PylonCamera.latch_clock',
        'PylonCamera.init_camera_config',
        'PylonCamera._enable_validity_chunks',
        'PylonCamera._write_bench_knob',
        'PylonCamera.set_pixel_format',
        'PylonCamera._set_hardware_binning',
        'PylonCamera.set_conversion_gain_mode',
        'PylonCamera.set_line_noise_reduction',
        'PylonCamera.set_black_level',
        'PylonCamera.init_auto_gain_focus',
        'PylonCamera.update_auto_gain_target_brightness',
        'PylonCamera.update_auto_gain_min_max',
        'PylonCamera._set_auto_exposure_time_bounds',
        'PylonCamera._set_hardware_window',
        'PylonCamera.gain',
        'PylonCamera.auto_gain',
        'PylonCamera.auto_gain_once',
        'PylonCamera.exposure_t',
        'PylonCamera.auto_exposure_t',
        'PylonCamera.set_test_pattern',
    },
    _IDS: {
        'IDSCamera._configure_free_run',
        'IDSCamera._query_dynamic_capabilities',
        'IDSCamera._rollback_failed_start',
        'IDSCamera._set_data_stream_int_node',
        'IDSCamera._set_free_run_frame_rate',
        'IDSCamera._set_hardware_binning',
        'IDSCamera._set_hardware_window',
        'IDSCamera._set_remote_node',
        'IDSCamera._start_grabbing',
        'IDSCamera._stop_grabbing',
        'IDSCamera._write_reverse',
        'IDSCamera.exposure_t',
        'IDSCamera.gain',
        'IDSCamera.init_camera_config',
        'IDSCamera.probe_afl_packed_acceptance',
        'IDSCamera.set_black_level',
        'IDSCamera.set_max_acquisition_frame_rate',
        'IDSCamera.set_pixel_format',
        'IDSCamera.set_test_pattern',
    },
}


def _production_tree():
    yield from iter_package_modules(('src', 'ui', 'tools'))
    for rel_path in tracked_files(pattern='*.py'):
        yield rel_path, parse_module(rel_path)


def _excepted(rel_path: str, qualname: str) -> bool:
    """Whether ``qualname`` is, or is nested in, a writer C rewrites."""
    return any(
        qualname == name or qualname.startswith(f'{name}.')
        for name in _REWRITTEN_BY_C.get(rel_path, ())
    )


def _is_the_handle(node, aliases) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == 'active'
        and isinstance(node.value, ast.Name)
        and node.value.id == 'self'
    ) or (isinstance(node, ast.Name) and node.id in aliases)


def _handle_aliases(function) -> set[str]:
    """The names a method binds to ``self.active`` (``camera = self.active``)."""
    return {
        n.targets[0].id
        for n in ast.walk(function)
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and _is_the_handle(n.value, set())
    }


def _lookup_call(node) -> bool:
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return False
    if node.func.attr == 'GetNode':
        return bool(node.args or node.keywords)
    return node.func.attr in _LOOKUPS


def _handle_lookup(node, aliases, qualname: str) -> bool:
    """A node reached on the Pylon instant camera by attribute or ``getattr``."""
    if (
        isinstance(node, ast.Attribute)
        and node.attr[:1].isupper()
        and node.attr not in _INSTANT_CAMERA_METHODS
        and _is_the_handle(node.value, aliases)
    ):
        return True
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == 'getattr'
        and bool(node.args)
        and _is_the_handle(node.args[0], aliases)
        and (_PYLON, qualname) != _NODEMAP_GETTER
    )


def _lookups_outside_the_owner(rel_path: str, tree: ast.Module) -> list[str]:
    """Every node lookup in ``tree`` that is not the driver's one lookup."""
    found = []

    def visit(node, qualname, aliases):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, f'{qualname}.{child.name}' if qualname else child.name, set())
                continue
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                inner = f'{qualname}.{child.name}' if qualname else child.name
                visit(child, inner, aliases | _handle_aliases(child))
                continue
            if (rel_path, qualname) not in _OWNERS and not _excepted(rel_path, qualname):
                pylon = rel_path == _PYLON and qualname.startswith('PylonCamera.')
                if _lookup_call(child) or (pylon and _handle_lookup(child, aliases, qualname)):
                    found.append(f'{rel_path}:{child.lineno} {qualname}: {ast.unparse(child)}')
            visit(child, qualname, aliases)

    visit(tree, '', set())
    return found


def test_no_camera_node_is_looked_up_but_through_the_drivers_one_lookup():
    found = [
        line
        for rel_path, tree in _production_tree()
        for line in _lookups_outside_the_owner(rel_path, tree)
    ]
    assert not found, (
        "A camera node looked up outside the driver's one lookup; find it with _node "
        '(or read it with read_node):\n' + '\n'.join(found)
    )


def test_every_excepted_writer_is_a_member_of_its_driver():
    """A writer renamed or deleted leaves no stale name on the exception list."""
    stale = []
    for rel_path, names in _REWRITTEN_BY_C.items():
        tree = parse_module(rel_path)
        defined = {
            f'{cls.name}.{fn.name}'
            for cls in tree.body
            if isinstance(cls, ast.ClassDef)
            for fn in cls.body
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        stale.extend(f'{rel_path}: {name}' for name in sorted(names - defined))
    assert not stale, '\n'.join(stale)


def test_the_walk_finds_each_way_a_node_is_looked_up():
    planted_pylon = ast.parse(
        'class PylonCamera:\n'
        '    @staticmethod\n'
        '    def _node_in(nodemap, name):\n'
        '        if not nodemap.Contains(name):\n'
        '            return None\n'
        '        return nodemap.GetNode(name)\n'
        '    def _nodemap(self, node_map):\n'
        "        return getattr(self.active, 'GetNodeMap')()\n"
        '    def by_name(self):\n'
        "        return self.nm.GetNode('X')\n"
        '    def by_alias(self):\n'
        '        camera = self.active\n'
        '        return camera.Width.GetValue()\n'
        '    def by_computed_name(self, name):\n'
        '        return getattr(self.active, name).GetValue()\n'
        '    def by_assignment(self):\n'
        '        self.active.AutoFunctionROIUseBrightness = True\n'
        '    def instant_camera_methods(self):\n'
        '        self.active.StopGrabbing()\n'
        '        return self.active.GetNodeMap()\n'
        '    def exposure_t(self, ms):\n'
        '        self.active.ExposureTime.SetValue(ms)\n'
    )
    planted_ids = ast.parse(
        'class IDSCamera:\n'
        '    @staticmethod\n'
        '    def _node_in(nodemap, name):\n'
        '        if not nodemap.HasNode(name):\n'
        '            return None\n'
        '        return nodemap.FindNode(name)\n'
        '    def get_gain(self):\n'
        "        return self.remote_nodemap.FindNode('Gain').Value()\n"
        '    def supports_black_level(self):\n'
        "        return self.remote_nodemap.HasNode('BlackLevel')\n"
    )
    planted_tool = ast.parse("def detect(camera):\n    return camera.GetTLNodeMap().GetNode('X')\n")
    handle = ast.parse('def name(nd):\n    return nd.GetNode().GetName()\n')

    assert _lookups_outside_the_owner(_PYLON, planted_pylon) == [
        f"{_PYLON}:10 PylonCamera.by_name: self.nm.GetNode('X')",
        f'{_PYLON}:13 PylonCamera.by_alias: camera.Width',
        f'{_PYLON}:15 PylonCamera.by_computed_name: getattr(self.active, name)',
        f'{_PYLON}:17 PylonCamera.by_assignment: self.active.AutoFunctionROIUseBrightness',
    ]
    assert _lookups_outside_the_owner(_IDS, planted_ids) == [
        f"{_IDS}:8 IDSCamera.get_gain: self.remote_nodemap.FindNode('Gain')",
        f"{_IDS}:10 IDSCamera.supports_black_level: self.remote_nodemap.HasNode('BlackLevel')",
    ]
    assert _lookups_outside_the_owner('tools/sweep.py', planted_tool) == [
        "tools/sweep.py:2 detect: camera.GetTLNodeMap().GetNode('X')"
    ]
    assert _lookups_outside_the_owner('tools/sweep.py', handle) == []
