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
from modules.exceptions import InstallationFileError
from tests.scope_fakes import bind_settings_like_a_session, build_scope


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
        with pytest.raises(InstallationFileError, match='Models') as info:
            load_scope_models(str(path))
        assert info.value.file_path == path

    def test_a_models_section_that_is_not_a_dict_refuses(self, tmp_path):
        from modules.layer_record import load_scope_models

        path = tmp_path / 'scopes.json'
        path.write_text(json.dumps({'Models': ['LS850T']}))
        with pytest.raises(InstallationFileError):
            load_scope_models(str(path))

    def test_an_unreadable_file_refuses_instead_of_returning_empty(self, tmp_path):
        from modules.layer_record import load_scope_models

        with pytest.raises(InstallationFileError):
            load_scope_models(str(tmp_path / 'missing.json'))


class TestInitializeStaysOnTheCallingThread:
    def test_initialize_takes_no_lane(self):
        """Bring-up is the scope configuring itself, not a command.

        Every write in it binds the impl, so it runs here and now, whatever
        the lanes would answer: with both lanes refusing work, a dispatched
        safety-off or camera apply would raise.
        """
        from modules.scope_init_config import ScopeInitConfig
        from tests.test_composite_run_config import _settings

        scope = build_scope(simulate=True, register_atexit=False)
        try:
            scope.io_lane().protocol_start()
            scope.camera_lane().protocol_start()
            settings = bind_settings_like_a_session(scope, **_settings())
            config = ScopeInitConfig.from_settings(settings, turreted=False)
            started = time.monotonic()
            scope.initialize(config)
            elapsed = time.monotonic() - started
            assert elapsed < 2.0, f'initialize took {elapsed:.1f}s'
            assert scope.runtime_state.get_current_objective_id() == settings['objective_id']
        finally:
            scope.disconnect()

    def test_no_led_write_without_a_board(self, monkeypatch):
        """With a Null board the bring-up safety-off writes nothing, and
        records no LED change the hardware never saw."""
        from drivers.null_ledboard import NullLEDBoard
        from modules.scope_init_config import ScopeInitConfig
        from tests.test_composite_run_config import _settings

        scope = build_scope(simulate=True, register_atexit=False)
        try:
            # a stand-in by design: the simulator has no scope without an LED
            # board, so the production Null board goes into the slot a scope
            # with no board holds. IlluminationAPI._driver is a read-only view
            # of that slot.
            board = NullLEDBoard()
            monkeypatch.setattr(scope, '_led_driver', board)
            calls = []
            monkeypatch.setattr(board, 'leds_off', lambda: calls.append('leds_off'))
            led_changes_before = scope.imaging.frame_validity.invalidation_counts.get('led', 0)
            settings = bind_settings_like_a_session(scope, **_settings())
            scope.initialize(ScopeInitConfig.from_settings(settings, turreted=False))
            led_changes = scope.imaging.frame_validity.invalidation_counts.get('led', 0)
            assert led_changes == led_changes_before
            assert calls == []
        finally:
            scope.disconnect()


class TestImageModeHandlerDuringInit:
    def test_select_image_mode_returns_before_its_camera_push_during_init(self):
        """Its sibling `select_binning_size` carries the same guard for the
        same reason; this pins that the image-mode handler reads
        `ctx.initializing` before it submits the Session apply to the camera
        lane (`submit_reported`)."""
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
                and isinstance(sub.func, ast.Name)
                and sub.func.id == 'submit_reported'
            ):
                put_line = put_line or sub.lineno
        assert guard_line is not None, 'select_image_mode has no ctx.initializing guard'
        assert put_line is not None
        assert guard_line < put_line, 'the guard must precede the camera-lane push'
