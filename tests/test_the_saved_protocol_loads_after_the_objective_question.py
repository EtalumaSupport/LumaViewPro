# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The saved protocol is loaded once the objective is settled, not before.

Whether the scope can perform a protocol depends on what the turret
carries. On a turreted scope, the slot at the current position is only
assigned when the startup objective question is answered -- and the panel
used to load the saved protocol from its own constructor, before that
question was even asked. So the protocol was judged against a turret
configuration that was about to change.

The question is asked from three places: startup, the provisional-settings
dialog resolving, and the turret arriving at an unassigned slot. Each is a
good reason to ask again; none is a reason to re-load the saved protocol
over whatever the user has done since. Hence the latch.

And the continuation cannot hang on the answer alone. Two paths in the
renderer report a failure to the user and return without one, and a third
returns when no question is owed -- a load hung only on an answer would
never run on any of them, leaving the app with no protocol and no reason
given.
"""

from __future__ import annotations

import ast

from tests.ast_seams import find_def


class TestTheStartupOrder:
    def test_the_panel_does_not_load_the_saved_protocol_in_init(self):
        """_init_ui runs from the panel's constructor, before the question."""
        init_ui = find_def('ui/protocol_settings.py', '_init_ui')
        assert init_ui is not None

        calls = [
            node.func.attr
            for node in ast.walk(init_ui)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        assert 'load_protocol' not in calls, (
            '_init_ui loads the saved protocol again, before the objective '
            'question has said what the turret carries'
        )
        assert 'load_persisted_protocol' not in calls, (
            '_init_ui calls the startup loader directly; the startup sequence owns when that runs'
        )

    def test_the_startup_sequence_hangs_the_load_on_the_question(self):
        fn = find_def('lumaviewpro.py', '_prompt_objective_if_needed')
        assert fn is not None
        src = ast.unparse(fn)

        assert 'on_resolved' in src, (
            'the startup prompt no longer passes a continuation, so the saved '
            'protocol is never loaded'
        )
        assert '_load_persisted_protocol_once' in src


class TestTheLatch:
    def test_the_load_runs_once_however_often_the_question_is_asked(self):
        """The turret re-ask and the settings re-ask must not re-load."""
        loads = []

        class _Panel:
            def load_persisted_protocol(self):
                loads.append(1)

        class _App:
            _persisted_protocol_loaded = False
            _load_persisted_protocol_once = __import__(
                'lumaviewpro', fromlist=['LumaViewProApp']
            ).LumaViewProApp._load_persisted_protocol_once

        import modules.app_context as _app_ctx

        app = _App()
        panel = _Panel()
        original = _app_ctx.ctx
        try:
            _app_ctx.ctx = type(
                'C',
                (),
                {'motion_settings': type('M', (), {'ids': {'protocol_settings_id': panel}})()},
            )()
            import lumaviewpro

            lumaviewpro.ctx = _app_ctx.ctx
            app._load_persisted_protocol_once()
            app._load_persisted_protocol_once()
            app._load_persisted_protocol_once()
        finally:
            _app_ctx.ctx = original

        assert loads == [1], f'the saved protocol was loaded {len(loads)} times'

    def test_the_latch_is_set_before_the_load_not_after(self):
        """A load that raises must not leave the door open for a re-ask."""
        fn = find_def('lumaviewpro.py', '_load_persisted_protocol_once')
        assert fn is not None
        body = ast.unparse(fn)

        latch_at = body.index('_persisted_protocol_loaded = True')
        load_at = body.index('load_persisted_protocol()')
        assert latch_at < load_at, (
            'the latch is set after the load, so a raising load lets the next question try again'
        )


class TestTheContinuationRunsOnEveryOutcome:
    def _prompt_source(self) -> str:
        fn = find_def('ui/vertical_control.py', 'prompt_if_objective_unknown')
        assert fn is not None
        return ast.unparse(fn)

    def test_it_runs_when_no_question_is_owed(self):
        src = self._prompt_source()
        assert '_resolve_objective' in src, (
            'nothing resolves the continuation when no question is owed, so an '
            'already-confirmed scope never loads its saved protocol'
        )

    def test_it_is_withheld_while_settings_are_provisional(self):
        """The one None that must NOT resolve: the host re-asks later."""
        src = self._prompt_source()
        assert 'settings_are_provisional' in src, (
            'the continuation runs even while settings are provisional, where '
            'the question is owed but its answer could not be kept'
        )

    def test_it_runs_when_the_question_itself_fails(self):
        """Both renderer failure paths report and return; neither may strand it."""
        fn = find_def('ui/vertical_control.py', 'prompt_if_objective_unknown')
        handlers = [node for node in ast.walk(fn) if isinstance(node, ast.ExceptHandler)]
        assert handlers, 'the question no longer guards itself'
        for handler in handlers:
            names = [
                node.func.attr
                for node in ast.walk(handler)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            ]
            assert '_resolve_objective' in names, (
                'a failed objective question returns without resolving the '
                'continuation, so the saved protocol never loads'
            )

    def test_it_runs_after_the_answer_even_if_rendering_it_fails(self):
        fn = find_def('ui/vertical_control.py', '_apply_objective_answer')
        assert fn is not None
        finallies = [node for node in ast.walk(fn) if isinstance(node, ast.Try) and node.finalbody]
        assert finallies, (
            'the answer path has no finally, so a widget write that raises '
            'strands the startup step waiting on the answer'
        )
        names = [
            node.func.attr
            for f in finallies
            for node in ast.walk(ast.Module(body=f.finalbody, type_ignores=[]))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        assert '_resolve_objective' in names

    def test_the_continuation_cannot_take_the_app_down(self):
        """It runs on a Clock callback, where a raise exits the app."""
        fn = find_def('ui/vertical_control.py', '_resolve_objective')
        assert fn is not None, 'the continuation runner is gone'
        handlers = [node for node in ast.walk(fn) if isinstance(node, ast.ExceptHandler)]
        assert handlers, 'the continuation runs unguarded on a Clock callback'


class TestARefusedStartupLoadKeepsThePath:
    def test_the_loader_only_clears_a_path_with_no_file_behind_it(self):
        fn = find_def('ui/protocol_settings.py', 'load_persisted_protocol')
        assert fn is not None, 'the startup loader is gone'
        src = ast.unparse(fn)

        assert 'exists()' in src, (
            'the loader clears the remembered path without asking whether the '
            'file is still there, so a refusal reads as a missing protocol'
        )
        cleared = src.count("settings['protocol']['filepath'] = ''")
        assert cleared == 1, (
            f'the path is cleared on {cleared} branches; only the absent-file branch may clear it'
        )

    def test_a_kept_path_is_also_shown(self):
        """Keeping the path in settings is no use if the panel reads blank.

        load_protocol writes the filename label only on the branch where it
        succeeds, so the branch that keeps a refused protocol's path has to
        put the name on screen itself.
        """
        fn = find_def('ui/protocol_settings.py', 'load_persisted_protocol')
        src = ast.unparse(fn)

        assert 'protocol_filename' in src, (
            'the refused-but-kept branch leaves the panel showing no filename '
            'while settings still hold the path'
        )
