# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The three preconditions of a Session-owned settings-to-scope bring-up.

A session factory configures the scope it built before it releases the
camera, on the calling thread, possibly with executor lanes that are
registered but not started. Three things had to be true first, and each
is pinned here by what it does rather than by what it says:

- the model catalogue is readable by a non-GUI caller and refuses loudly
  when the file has no usable `Models` section, because an empty
  catalogue silently disables the slot-1 objective adoption;
- `Lumascope.initialize` never dispatches onto an executor lane, so it
  completes on a scope whose IO lane exists but has no worker;
- the image-mode spinner's handler returns early during app init, so the
  synchronous pixel-format apply inside `initialize` is the only one.
"""

import json
import time

import pytest

import tests.ast_seams as ast_seams
from modules.exceptions import ConfigError


class TestScopeModelsCatalogue:
    def test_the_shipped_catalogue_loads_its_models(self):
        from modules.layer_record import load_scope_models

        models = load_scope_models()
        assert isinstance(models, dict) and 'LS850T' in models
        assert models['LS850T']['Turret'] is True

    def test_a_file_without_a_models_section_refuses_by_name(self, tmp_path):
        from modules.layer_record import load_scope_models

        path = tmp_path / 'scopes.json'
        path.write_text(json.dumps({'Layers': []}))
        with pytest.raises(ConfigError) as info:
            load_scope_models(str(path))
        assert 'Models' in str(info.value) and str(path) in str(info.value)

    def test_a_models_section_that_is_not_a_dict_refuses(self, tmp_path):
        from modules.layer_record import load_scope_models

        path = tmp_path / 'scopes.json'
        path.write_text(json.dumps({'Models': ['LS850T']}))
        with pytest.raises(ConfigError):
            load_scope_models(str(path))

    def test_an_unreadable_file_refuses_instead_of_returning_empty(self, tmp_path):
        from modules.layer_record import load_scope_models

        with pytest.raises(ConfigError):
            load_scope_models(str(tmp_path / 'missing.json'))


class TestInitializeStaysOnTheCallingThread:
    def test_initialize_completes_with_a_registered_but_unstarted_io_lane(self):
        """The factory case: executors registered, no worker yet.

        The public LED dispatcher would submit the safety-off to the IO
        lane and wait the full write timeout for a worker that never
        comes. Bound to the impl, the write happens here and now.
        """
        import modules.lumascope_api as lumascope_api
        from modules.scope_init_config import ScopeInitConfig
        from modules.sequential_io_executor import SequentialIOExecutor
        from tests.test_composite_run_config import _settings

        scope = lumascope_api.Lumascope(simulate=True, register_atexit=False)
        io = SequentialIOExecutor(name='IO_UNSTARTED')
        cam = SequentialIOExecutor(name='CAMERA_UNSTARTED')
        try:
            scope.register_executors(io_executor=io, camera_executor=cam)
            config = ScopeInitConfig.from_settings(_settings(), labware=None)
            started = time.monotonic()
            scope.initialize(config)
            elapsed = time.monotonic() - started
            assert elapsed < 2.0, f'initialize blocked {elapsed:.1f}s on an unstarted lane'
            assert scope.runtime_state.get_current_objective_id() == config.objective_id
        finally:
            scope.disconnect()

    def test_no_led_write_without_a_board(self, monkeypatch):
        """Preservation pin: the board check the dispatcher applied survives
        the move. With a Null board the impl must not run at all, or the
        state cache would record a safety-off the hardware never saw."""
        import modules.lumascope_api as lumascope_api
        from drivers.null_ledboard import NullLEDBoard
        from modules.scope_init_config import ScopeInitConfig
        from tests.test_composite_run_config import _settings

        scope = lumascope_api.Lumascope(simulate=True, register_atexit=False)
        try:
            # IlluminationAPI._driver is a read-only view of the scope's slot.
            monkeypatch.setattr(scope, '_led_driver', NullLEDBoard())
            calls = []
            monkeypatch.setattr(scope.illumination, '_leds_off_impl', lambda: calls.append(1))
            scope.initialize(ScopeInitConfig.from_settings(_settings(), labware=None))
            assert calls == []
        finally:
            scope.disconnect()


class TestImageModeHandlerDuringInit:
    def test_select_image_mode_returns_before_its_camera_push_during_init(self):
        """Its sibling `select_binning_size` carries the same guard for the
        same reason; this pins that the image-mode handler reads
        `ctx.initializing` before it reaches `camera_executor.put`."""
        import ast

        node = ast_seams.find_def(
            'ui/microscope_settings.py', 'select_image_mode', class_name='MicroscopeSettings'
        )
        assert node is not None
        guard_line = None
        put_line = None
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute) and sub.attr == 'initializing':
                guard_line = guard_line or sub.lineno
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == 'put'
            ):
                put_line = put_line or sub.lineno
        assert guard_line is not None, 'select_image_mode has no ctx.initializing guard'
        assert put_line is not None
        assert guard_line < put_line, 'the guard must precede the camera-lane push'
