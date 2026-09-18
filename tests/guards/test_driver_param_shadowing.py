# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""No driver method takes a parameter named after itself.

A sweep over every module in drivers/, so it gates every commit from
tests/guards/; the per-driver regression tests it grew out of stay in
tests/test_audit_fixes.py.
"""

import ast

from tests.ast_seams import REPO_ROOT, parse_module


class TestDriverParametersNotShadowingMethods:
    """A parameter named like its own method shadows the method.

    `def gain(self, gain)` had the parameter shadow the method name in
    several camera drivers. Inside such a method body the symbol
    resolves to the parameter -- the bound method `self.gain` is still
    reachable, but a future refactor that calls the method recursively
    (or reads `self.gain` expecting the method) fails in a confusing
    way. The de-shadowed parameter name is `value`; the method names
    themselves are L2-public and unchanged (Rule 30 stability).

    Originally a PylonCamera-only signature pin (audit finding A15);
    widened to a driver-wide AST scan when the same shape was found in
    camera.py / idscamera.py / simulated_camera.py.
    """

    def test_no_driver_method_param_shadows_its_method_name(self):
        """No function in any drivers/*.py module may take a parameter
        named identically to the function itself."""
        offenders = []
        for path in sorted((REPO_ROOT / 'drivers').glob('*.py')):
            rel = path.relative_to(REPO_ROOT).as_posix()
            tree = parse_module(rel)
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                args = node.args
                params = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
                if args.vararg:
                    params.append(args.vararg.arg)
                if args.kwarg:
                    params.append(args.kwarg.arg)
                if node.name in params:
                    offenders.append(f'{rel}:{node.lineno} def {node.name}')
        assert not offenders, (
            'Driver function parameters must not shadow the method name '
            '(use `value` for single-value setters): ' + ', '.join(offenders)
        )
