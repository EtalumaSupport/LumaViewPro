"""Engineering mode is a run value the engine is handed, never a flag it reads.

The engine used to learn whether it was running in an engineering build by
reading the GUI's application context at two points on the save path: the
protocol image writer, when it composed a step's filename, and the free
save-path function two hops below it. In the GUI both reads see the one live
flag. In any headless process the context is unset, so both reads resolved to
False and the turret position silently vanished from every filename an
engineering-mode headless run wrote. When the flag WAS set, each read stamped
its own token, so one file carried the turret position twice in two spellings
(``A1_Blue_Turret2_T2.tiff``), the second of which no filename reader
recognises.

Now the caller states the mode once. The session records the mode it was
built in, the run plan freezes it, and the writer takes it as a required
constructor argument. Only the writer's canonical ``Turret<n>`` token remains.
"""

import ast
import re
import threading
from unittest.mock import MagicMock

import pytest

from tests.ast_seams import parse_module
from tests.test_composite_run_e2e import headless_settings, open_composite_session, single_run_dir


def _turret_token(session) -> str:
    """The token the writer renders for the turret position the scope holds."""
    return f'Turret{int(session.scope.motion.get_current_position("T"))}'


def _saved_frames(tmp_path) -> list[str]:
    return sorted(p.name for p in single_run_dir(tmp_path).glob('*.tiff'))


class TestTheSessionRecordsTheMode:
    def test_a_session_built_in_engineering_mode_stamps_the_turret_token(self, tmp_path):
        """A headless run honours the mode its session was built in, with
        nothing in the process for the engine to read it from."""
        settings = headless_settings(tmp_path)

        with open_composite_session(settings, engineering_mode=True) as (session, runner):
            assert session.engineering_mode is True
            runner.run_composite(sequence_name='eng', parent_dir=str(tmp_path))
            token = _turret_token(session)

        frames = _saved_frames(tmp_path)
        assert frames, 'the run wrote no frames'
        for name in frames:
            assert token in name, f'{name} carries no turret position; expected {token}'

    def test_outside_engineering_mode_no_turret_token_is_written(self, tmp_path):
        """Behaviour preserved on both sides of the change: a production-mode
        run names its files without the turret position."""
        with open_composite_session(headless_settings(tmp_path)) as (session, runner):
            assert session.engineering_mode is False
            runner.run_composite(sequence_name='prod', parent_dir=str(tmp_path))

        for name in _saved_frames(tmp_path):
            assert 'Turret' not in name, f'{name} carries a turret token outside engineering mode'


class TestTheGuiRouteCarriesTheLiveFlag:
    def test_the_composite_starter_forwards_a_mode_the_session_did_not_record(self, tmp_path):
        """The GUI's flag is flipped by a plugin after the session exists, so
        the GUI composite hands the LIVE value to the run rather than trusting
        the session's as-built one."""
        with open_composite_session(headless_settings(tmp_path)) as (session, runner):
            outcome = runner.start_composite(
                sequence_name='gui',
                parent_dir=str(tmp_path),
                run_trigger_source='composite',
                engineering_mode=True,
            )
            settled = outcome.wait(timeout_s=120)
            assert settled is not None and settled.merged, 'the run never settled'
            token = _turret_token(session)

        for name in _saved_frames(tmp_path):
            assert token in name, f'{name} carries no turret position; expected {token}'


class TestOneTokenOneSpelling:
    def test_a_run_filename_carries_exactly_one_turret_token(self, tmp_path):
        """The two reads each stamped their own spelling; the writer's
        canonical token is the only one left."""
        with open_composite_session(headless_settings(tmp_path), engineering_mode=True) as (
            session,
            runner,
        ):
            runner.run_composite(sequence_name='once', parent_dir=str(tmp_path))
            token = _turret_token(session)

        for name in _saved_frames(tmp_path):
            stem = name.rsplit('.', 1)[0]
            assert stem.count(token) == 1, f'{name} does not carry {token} exactly once'
            assert re.search(r'_T\d+(_|$)', stem) is None, (
                f'{name} still carries the legacy _T<n> spelling beside {token}'
            )


class TestTheWriterIsHandedTheMode:
    def test_the_writer_cannot_be_built_without_the_mode(self):
        """Every writer parameter is required so no writer can decide a run
        value by itself; the mode is one of them."""
        from modules.image_mode import ImageCaptureConfig
        from modules.protocol_callbacks import ProtocolCallbacks
        from modules.protocol_image_writer import ProtocolImageWriter

        kwargs = {
            'scope': MagicMock(),
            'callbacks': ProtocolCallbacks(),
            'aborted': threading.Event(),
            'file_io_executor': MagicMock(),
            'abort_fn': lambda: None,
            'fatal_abort_event': threading.Event(),
            'execution_record': None,
            'leds_off_fn': lambda: None,
            'is_run_in_progress_fn': lambda: True,
            'image_capture_config': ImageCaptureConfig.from_image_mode('8bit'),
            'timestamp_overlay': True,
            'video_max_fps': 0,
        }

        with pytest.raises(TypeError, match='engineering_mode'):
            ProtocolImageWriter(**kwargs)


class TestTheSavePathReadsNoProcessGlobal:
    def test_the_save_path_module_no_longer_imports_the_context(self):
        """Structural: with its one read gone, the module has no reason to
        know the application context exists. (The writer still imports it
        for the live-display hold, a separate read with its own retirement.)
        """
        tree = parse_module('modules/image_save.py')

        context_imports = [
            node.lineno
            for node in ast.walk(tree)
            if (
                isinstance(node, ast.Import)
                and any(a.name == 'modules.app_context' for a in node.names)
            )
            or (isinstance(node, ast.ImportFrom) and node.module == 'modules.app_context')
        ]
        assert not context_imports, (
            f'modules/image_save.py imports the application context at {context_imports}'
        )

    @pytest.mark.parametrize(
        'rel_path', ['modules/image_save.py', 'modules/protocol_image_writer.py']
    )
    def test_the_module_reads_the_flag_off_nothing_but_the_writer_itself(self, rel_path):
        """Structural: the read cannot come back quietly. The writer may read
        the mode it was HANDED (``self._engineering_mode``); neither module
        may read the flag off anything else. Walks the AST so a mention in
        a comment is not a failure."""
        tree = parse_module(rel_path)

        # Both retired reads were ``getattr(<ctx>, 'engineering_mode', False)``,
        # a string constant no attribute-node walk sees, so that form is
        # checked beside the plain attribute read.
        foreign_reads = [
            node.lineno
            for node in ast.walk(tree)
            if (
                isinstance(node, ast.Attribute)
                and node.attr == 'engineering_mode'
                and not (isinstance(node.value, ast.Name) and node.value.id == 'self')
            )
            or (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == 'getattr'
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == 'engineering_mode'
            )
        ]
        assert not foreign_reads, (
            f'{rel_path} reads engineering_mode off something other than the writer '
            f'itself at line(s) {foreign_reads}; the mode is a run value the caller hands in'
        )
