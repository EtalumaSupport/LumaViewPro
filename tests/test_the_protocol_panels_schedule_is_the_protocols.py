# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The Period and Duration fields edit the panel's protocol and show what it holds.

The settings store's period and duration are the default a new protocol
starts from at start-up, and nothing the panel does writes them. A field
edit goes to the protocol's own writer, which takes it or refuses it; the
field then shows the protocol's value either way. Run and Save take the
protocol as it is -- they copy nothing into it -- and New starts from the
schedule on screen.

A Run, Scan or Save clicked straight after a refused edit does nothing but
leave the refusal on screen: the click's own touch committed the edit, and
acting on the schedule the person just tried to change would run or save
one they did not mean.
"""

import ast
import datetime
import pathlib
import re
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest


class _StubWidget:
    def __init__(self, **kwargs):
        pass


for _name in (
    'kivy.app',
    'kivy.properties',
    'kivy.uix',
    'kivy.uix.label',
    'kivy.uix.popup',
    'kivy.lang',
    'kivy.metrics',
    'kivy.graphics',
):
    sys.modules.setdefault(_name, MagicMock())
for _name, _attr in (
    ('kivy.uix.floatlayout', 'FloatLayout'),
    ('kivy.uix.boxlayout', 'BoxLayout'),
    ('kivy.uix.scrollview', 'ScrollView'),
    ('kivy.uix.widget', 'Widget'),
):
    if _name not in sys.modules:
        _mod = types.ModuleType(_name)
        setattr(_mod, _attr, _StubWidget)
        sys.modules[_name] = _mod

import modules.app_context as _app_ctx
import ui.protocol_settings as ps
import ui.ui_helpers as ui_helpers
from modules.notification_center import notifications
from modules.protocol import Protocol
from tests.ast_seams import parse_module
from tests.scope_fakes import spec_scope

REPO = pathlib.Path(__file__).resolve().parent.parent
PERIOD = datetime.timedelta(minutes=7)
DURATION = datetime.timedelta(hours=3)
STORED = {'period': 20, 'duration': 48, 'filepath': 'plate.tsv'}


def _protocol() -> Protocol:
    return Protocol(
        tiling_configs_file_loc=REPO / 'data' / 'tiling.json',
        config={
            'version': Protocol.CURRENT_VERSION,
            'steps': pd.DataFrame(),
            'custom_step_count': 0,
            'period': PERIOD,
            'duration': DURATION,
            'capture_root': '',
            'labware_id': 'Center Plate',
        },
    )


class _Panel(ps.ProtocolSettings):
    """The real class, with only the widget tree stubbed."""

    def __init__(self, protocol):
        self.ids = {
            'capture_period': SimpleNamespace(text=''),
            'capture_dur': SimpleNamespace(text=''),
            'protocol_filename': SimpleNamespace(text='plate.tsv'),
            'capture_root': SimpleNamespace(text=''),
            'tiling_size_spinner': SimpleNamespace(text='1x1'),
            'acquire_zstack_id': SimpleNamespace(active=False),
        }
        self._protocol = protocol
        self._runs_started_here = {}
        self.curr_step = 0

    def go_to_step(self, step_idx, protocol=True):
        pass

    def update_step_ui(self):
        pass

    def _draw_protocol_steps(self):
        pass

    def draw_protocol_buttons(self):
        pass


@pytest.fixture
def ctx(monkeypatch):
    context = SimpleNamespace(
        session=MagicMock(),
        scope=spec_scope(),
        settings={'protocol': dict(STORED)},
        sequenced_capture_runner=MagicMock(),
    )
    context.sequenced_capture_runner.is_live_run.return_value = False
    monkeypatch.setattr(_app_ctx, 'ctx', context)
    monkeypatch.setattr(ps.gui_logger, 'text_input', lambda *a, **kw: None)
    monkeypatch.setattr(ps.gui_logger, 'protocol_action', lambda *a, **kw: None)
    monkeypatch.setattr(ps.gui_logger, 'button', lambda *a, **kw: None)
    return context


@pytest.fixture
def reported(monkeypatch):
    seen = []
    monkeypatch.setattr(
        notifications,
        'report_outcome',
        lambda exc, **kw: seen.append((type(exc).__name__, kw['category'])),
    )
    return seen


class TestAFieldEdit:
    def test_a_runnable_period_goes_to_the_protocol_and_not_the_store(self, ctx, reported):
        panel = _Panel(_protocol())
        panel.ids['capture_period'].text = '10'

        panel.update_period()

        assert reported == []
        assert panel._protocol.period() == datetime.timedelta(minutes=10)
        assert panel._protocol.duration() == DURATION
        assert ctx.settings['protocol'] == STORED
        assert float(panel.ids['capture_period'].text) == 10

    def test_a_runnable_duration_goes_to_the_protocol_and_not_the_store(self, ctx, reported):
        panel = _Panel(_protocol())
        panel.ids['capture_dur'].text = '5'

        panel.update_duration()

        assert reported == []
        assert panel._protocol.duration() == datetime.timedelta(hours=5)
        assert ctx.settings['protocol'] == STORED

    @pytest.mark.parametrize('typed', ['0.005', '', '.'])
    def test_a_period_that_cannot_run_is_refused_and_the_field_shows_the_protocols(
        self, ctx, reported, typed
    ):
        panel = _Panel(_protocol())
        panel.ids['capture_period'].text = typed

        panel.update_period()

        assert reported == [('ProtocolScheduleRefusedError', 'UI:PROTOCOL_PERIOD')]
        assert (panel._protocol.period(), panel._protocol.duration()) == (PERIOD, DURATION)
        assert float(panel.ids['capture_period'].text) == 7
        assert ctx.settings['protocol'] == STORED

    def test_a_duration_that_cannot_run_is_refused_and_the_field_shows_the_protocols(
        self, ctx, reported
    ):
        panel = _Panel(_protocol())
        panel.ids['capture_dur'].text = '-2'

        panel.update_duration()

        assert reported == [('ProtocolScheduleRefusedError', 'UI:PROTOCOL_DURATION')]
        assert panel._protocol.duration() == DURATION
        assert float(panel.ids['capture_dur'].text) == 3


class TestRunSaveAndNewTakeTheProtocolsSchedule:
    def test_the_full_run_runs_the_protocols_schedule_not_the_stores(self, ctx, monkeypatch):
        panel = _Panel(_protocol())
        handed = []
        monkeypatch.setattr(
            panel, '_sequenced_capture_start', lambda **kw: handed.append(kw['protocol'])
        )

        panel._protocol_start()

        assert (handed[0].period(), handed[0].duration()) == (PERIOD, DURATION)
        assert handed[0] is not panel._protocol

    def test_save_writes_the_protocols_schedule_and_leaves_it_as_it_was(
        self, ctx, monkeypatch, tmp_path
    ):
        panel = _Panel(_protocol())
        monkeypatch.setattr(_Panel, '_gather_layer_settings_for_save', lambda self: {})

        panel.save_protocol(filepath=str(tmp_path / 'saved'))

        assert (panel._protocol.period(), panel._protocol.duration()) == (PERIOD, DURATION)
        text = (tmp_path / 'saved.tsv').read_text()
        assert 'Period\t7.0\n' in text and 'Duration\t3.0\n' in text
        assert ctx.settings['protocol']['period'] == STORED['period']

    def test_new_starts_from_the_schedule_on_screen(self, ctx):
        built = _protocol()
        ctx.session.new_protocol.return_value = built
        on_screen = _protocol()
        on_screen.modify_time_params(
            period=datetime.timedelta(minutes=2), duration=datetime.timedelta(hours=9)
        )
        panel = _Panel(on_screen)

        panel.new_protocol()

        kwargs = ctx.session.new_protocol.call_args.kwargs
        assert (kwargs['period'], kwargs['duration']) == (
            datetime.timedelta(minutes=2),
            datetime.timedelta(hours=9),
        )

    def test_a_new_protocol_is_shown_with_its_own_schedule(self, ctx):
        ctx.session.new_protocol.return_value = _protocol()
        panel = _Panel(_protocol())
        panel.ids['capture_period'].text = '99'

        panel.new_protocol()

        assert float(panel.ids['capture_period'].text) == 7
        assert float(panel.ids['capture_dur'].text) == 3


class TestAClickAfterARefusedEdit:
    """The same input: the field's focus-loss commit, then the button's on_release."""

    @pytest.fixture
    def same_input(self, monkeypatch):
        monkeypatch.setattr(ui_helpers, '_input_frame', lambda: 41)

    def _refuse_an_edit(self, panel):
        panel.ids['capture_period'].text = '0.005'
        panel.update_period()

    def test_run_does_not_start(self, ctx, reported, same_input):
        panel = _Panel(_protocol())
        starts = []
        panel._submit_panel_request = lambda trigger, call, stop=False: starts.append(trigger)

        self._refuse_an_edit(panel)
        panel.run_protocol_from_ui()

        assert starts == []

    def test_scan_does_not_start(self, ctx, reported, same_input):
        panel = _Panel(_protocol())
        starts = []
        panel._submit_panel_request = lambda trigger, call, stop=False: starts.append(trigger)

        self._refuse_an_edit(panel)
        panel.run_scan_from_ui()

        assert starts == []

    def test_save_does_not_save(self, ctx, reported, same_input, monkeypatch, tmp_path):
        panel = _Panel(_protocol())
        monkeypatch.setattr(_Panel, '_gather_layer_settings_for_save', lambda self: {})

        self._refuse_an_edit(panel)
        panel.save_protocol(filepath=str(tmp_path / 'saved'))

        assert not (tmp_path / 'saved.tsv').exists()

    def test_a_click_in_a_later_input_acts(self, ctx, reported, monkeypatch):
        panel = _Panel(_protocol())
        starts = []
        panel._submit_panel_request = lambda trigger, call, stop=False: starts.append(trigger)
        panel._protocol_start = lambda: lambda: None
        frame = iter([41, 42])
        monkeypatch.setattr(ui_helpers, '_input_frame', lambda: next(frame))

        self._refuse_an_edit(panel)
        panel.run_protocol_from_ui()

        assert starts == ['protocol']


class TestTheStoredScheduleHasTwoWriters:
    """The stored default is written by the user's file and by update_settings, nothing else."""

    WRITE = re.compile(r"\[\s*['\"](period|duration)['\"]\s*\]\s*=(?!=)")

    def test_no_gui_or_module_writes_a_stored_period_or_duration(self):
        writers = []
        for path in [
            *REPO.glob('ui/**/*.py'),
            *REPO.glob('modules/**/*.py'),
            REPO / 'lumaviewpro.py',
        ]:
            for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
                if 'protocol' in line and self.WRITE.search(line):
                    writers.append(f'{path.relative_to(REPO)}:{number}: {line.strip()}')
        assert writers == []

    def test_the_instrument_sees_a_write(self):
        assert self.WRITE.search("settings['protocol']['period'] = raw_period")

    # A read of the stored schedule in the GUI is a second store for what the
    # fields show: they show the protocol's.
    READ = re.compile(r"\[\s*['\"]protocol['\"]\s*\]\s*\[\s*['\"](period|duration)['\"]\s*\]")

    def test_no_gui_code_reads_the_stored_period_or_duration(self):
        readers = []
        for path in REPO.glob('ui/**/*.py'):
            for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
                if self.READ.search(line):
                    readers.append(f'{path.relative_to(REPO)}:{number}: {line.strip()}')
        assert readers == []

    def test_the_instrument_sees_a_read(self):
        assert self.READ.search("str(settings['protocol']['period'])")


class TestEveryProtocolThePanelTakesIsShown:
    """Each place the panel takes a protocol shows its schedule next."""

    @staticmethod
    def _takes(tree: 'ast.Module') -> list[tuple[int, bool]]:
        takes = []
        for node in ast.walk(tree):
            body = getattr(node, 'body', None)
            if not isinstance(body, list):
                continue
            for i, stmt in enumerate(body):
                if not (
                    isinstance(stmt, ast.Assign)
                    and any(
                        isinstance(t, ast.Attribute)
                        and t.attr == '_protocol'
                        and isinstance(t.value, ast.Name)
                        and t.value.id == 'self'
                        for t in stmt.targets
                    )
                ):
                    continue
                if isinstance(stmt.value, ast.Constant) and stmt.value.value is None:
                    continue
                after = body[i + 1] if i + 1 < len(body) else None
                shown = (
                    isinstance(after, ast.Expr)
                    and isinstance(after.value, ast.Call)
                    and isinstance(after.value.func, ast.Attribute)
                    and after.value.func.attr == '_show_schedule'
                )
                takes.append((stmt.lineno, shown))
        return takes

    def test_each_take_is_followed_by_showing_the_schedule(self):
        takes = self._takes(parse_module('ui/protocol_settings.py'))

        assert len(takes) >= 4, f'the panel takes a protocol in at least four places, found {takes}'
        assert [line for line, shown in takes if not shown] == []

    def test_the_instrument_sees_a_take_left_unshown(self):
        source = (
            'class P:\n'
            '    def a(self):\n'
            '        self._protocol = build()\n'
            '        self._show_schedule()\n'
            '    def b(self):\n'
            '        self._protocol = build()\n'
            '        self.update_step_ui()\n'
        )
        assert self._takes(ast.parse(source)) == [(3, True), (6, False)]
