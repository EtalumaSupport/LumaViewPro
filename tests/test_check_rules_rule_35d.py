"""Tests for tools/check_rules.py rule_35d.

rule_35d blocks bare ``scope.<method>`` calls where the method was relocated
to a sub-API namespace by the Wave-7 decomposition. Lumascope no longer
exposes a same-named forwarder; bare ``scope.<method>(...)`` raises
AttributeError at runtime. MagicMock scopes in tests silently absorb the
access, so the test suite alone can't catch the regression -- pre-commit is
the right gate.

Surfaced 2026-05-26 when bench-day exercise of etaluma-engineering plugin's
camera_characterization.py crashed on the first scope.set_pixel_format call;
audit found 68 sibling calls in the same file.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.check_rules import check_source


def _violations(content: str, path: str) -> list:
    return [v for v in check_source(content, path) if v.rule == 'rule_35d']


class TestRule35dBlocksBareScopeMethodCalls:
    def test_bare_scope_set_pixel_format_blocks(self):
        src = 'def cell(scope, pf):\n    scope.set_pixel_format(pf)\n'
        violations = _violations(src, 'tools/pylon_probe_sweep.py')
        assert len(violations) == 1
        assert 'set_pixel_format' in violations[0].message

    def test_bare_scope_move_absolute_position_blocks(self):
        src = 'def sweep(scope, z):\n    scope.move_absolute_position("Z", z)\n'
        violations = _violations(src, 'etaluma_engineering/camera_characterization.py')
        assert len(violations) == 1
        assert 'move_absolute_position' in violations[0].message

    def test_self_scope_get_current_position_blocks(self):
        src = (
            'class Widget:\n'
            '    def tick(self):\n'
            '        return self.scope.get_current_position("X")\n'
        )
        violations = _violations(src, 'ui/some_widget.py')
        assert len(violations) == 1
        assert 'get_current_position' in violations[0].message

    def test_lumaview_scope_chain_blocks(self):
        src = (
            'def helper(lumaview):\n'
            '    return lumaview.scope.get_target_status("Z")\n'
        )
        violations = _violations(src, 'ui/motion_settings.py')
        assert len(violations) == 1


class TestRule35dAllowsSubApiAccess:
    def test_scope_motion_move_absolute_position_passes(self):
        src = 'def sweep(scope, z):\n    scope.motion.move_absolute_position("Z", z)\n'
        assert _violations(src, 'etaluma_engineering/camera_characterization.py') == []

    def test_scope_imaging_set_pixel_format_passes(self):
        src = 'def cell(scope, pf):\n    scope.imaging.set_pixel_format(pf)\n'
        assert _violations(src, 'tools/pylon_probe_sweep.py') == []

    def test_scope_diagnostics_run_pylon_probe_passes(self):
        src = (
            'def probe(scope):\n'
            '    return scope.diagnostics.run_pylon_diagnostic_probe(duration_s=3.0)\n'
        )
        assert _violations(src, 'tools/pylon_probe_sweep.py') == []


class TestRule35dExempts:
    def test_test_file_exempt(self):
        src = (
            'def test_thing(scope):\n'
            '    scope.move_absolute_position.return_value = None\n'
            '    scope.set_pixel_format("Mono8")\n'
        )
        # Path under tests/ -- exempt because MagicMock attribute targeting
        # is the intentional pattern in tests.
        assert _violations(src, 'tests/test_something.py') == []

    def test_lumascope_api_module_exempt(self):
        src = 'def forwarder(scope):\n    return scope.move_absolute_position("Z", 0)\n'
        assert _violations(src, 'modules/lumascope_api/_lumascope.py') == []


class TestRule35dDoesNotMisfire:
    def test_unrelated_attr_passes(self):
        src = 'def fn(scope):\n    return scope.capabilities.axes\n'
        assert _violations(src, 'modules/foo.py') == []

    def test_non_scope_object_with_same_method_name_passes(self):
        # `motor.move_absolute_position(...)` -- not a scope attribute, no fire.
        src = 'def fn(motor):\n    motor.move_absolute_position("Z", 0)\n'
        assert _violations(src, 'drivers/motorboard.py') == []

    def test_string_literal_passes(self):
        # Mentioning the method name in a string is not a call.
        src = 'def fn():\n    return "scope.set_pixel_format is deprecated"\n'
        assert _violations(src, 'modules/foo.py') == []
