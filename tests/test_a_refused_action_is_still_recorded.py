# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A refused pick or click is still a thing the user did.

Three handlers judged the action before recording it, so the one case a
forensic reader most wants to see -- the user asked for something and did not
get it -- was the one case that left no line at all. The bundle showed a
warning with nothing saying which control provoked it, or a stage that never
moved with nothing saying anyone had asked it to.

Attribution is the point, and it is why recording at the top beats recording
at each refusal. The notification text does not identify the control: the
file-writes gate says the same words for five different buttons, and the
protocol builder's refusal is shared by nine callers. The record names the
control; the notification that follows says why it was refused.

The two handlers whose refusal comes from the Session are driven against the
real simulated session, with an input it refuses; the stage boxes are driven
through their real handlers.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

import modules.app_context as _app_ctx
import ui.microscope_settings as ms
import ui.protocol_settings as ps
from modules.exceptions import ProtocolRunRefusedError
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

_GUI_LOG = 'LVP.gui_interactions'


@pytest.fixture
def session(tmp_path):
    """The simulated session, on the shipped settings: no layer acquires."""
    session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    yield session
    session.shutdown()


def _recorded(caplog):
    return [r.getMessage() for r in caplog.records if r.name == _GUI_LOG]


class _Microscope(ms.MicroscopeSettings):
    """The real class, with only the widget tree stubbed."""

    def __init__(self, binning_label):
        # a stand-in by design: Kivy is stubbed in the test process; the subject is the handler's record
        self.ids = {
            'binning_spinner': SimpleNamespace(text=binning_label),
            'frame_width_id': SimpleNamespace(text='', focus=False),
            'frame_height_id': SimpleNamespace(text='', focus=False),
        }
        self.redrawn = 0

    def _redraw_framing(self):
        # a stand-in by design: the redraw draws the widget tree Kivy's stub cannot hold
        self.redrawn += 1


class _Protocols(ps.ProtocolSettings):
    """The real class, with only the widget tree stubbed."""

    def __init__(self):
        # a stand-in by design: Kivy is stubbed in the test process; the subject is the handler's record
        self.ids = {
            'tiling_size_spinner': SimpleNamespace(text='1x1'),
            'acquire_zstack_id': SimpleNamespace(active=False),
            'step_name_input': SimpleNamespace(text=''),
        }
        self._protocol = SimpleNamespace(
            period_s=lambda: 20 * 60.0,
            duration_s=lambda: 48 * 3600.0,
        )

    def update_step_ui(self):
        pass


def test_a_binning_the_session_refuses_is_still_recorded(session, monkeypatch, caplog):
    """A binning this camera does not offer is refused by the Session's
    preview, reported once, and no apply is sent; the pick that asked for it
    is in the record all the same."""
    from modules.notification_center import notifications

    reported = []
    submitted = []
    monkeypatch.setattr(notifications, 'report_outcome', lambda exc, **_kw: reported.append(exc))
    monkeypatch.setattr(ms, 'submit_reported', lambda *args, **kwargs: submitted.append(args))
    monkeypatch.setattr(
        _app_ctx,
        'ctx',
        SimpleNamespace(
            scope=session.scope,
            settings=session.settings,
            initializing=False,
            camera_executor=None,
        ),
    )
    offered = session.scope.capabilities.camera_binning_sizes
    unoffered = max(offered) + 1
    microscope = _Microscope(f'{unoffered}x{unoffered}')

    from ui.pick_spinner import PickSpinner

    spinner = microscope.ids['binning_spinner']
    # a stand-in by design: Kivy is stubbed in the test process; the pick's
    # own path records, then the text it shows runs the handler as on_text does
    spinner.log_name, spinner.text_follows_pick, spinner.is_open = 'BINNING', True, True
    spinner.dispatch = lambda event, value: None
    with caplog.at_level(logging.INFO, logger=_GUI_LOG):
        PickSpinner._on_dropdown_select(spinner, None, f'{unoffered}x{unoffered}')
        microscope.select_binning_size()

    assert [getattr(e, 'reason', e) for e in reported] == ['binning_unsupported']
    assert submitted == [] and microscope.redrawn == 1
    assert f'SELECT BINNING {unoffered}x{unoffered}' in _recorded(caplog)


def test_a_new_protocol_the_session_refuses_is_still_recorded(session, monkeypatch, caplog):
    """With no layer set to acquire, the Session refuses the build; the press
    that asked for it is in the record, and nothing was built."""
    with pytest.raises(ProtocolRunRefusedError) as refused:
        session.new_protocol()
    assert refused.value.reason == 'no_acquiring_layer'
    monkeypatch.setattr(_app_ctx, 'ctx', SimpleNamespace(session=session))
    caplog.clear()

    with caplog.at_level(logging.INFO, logger=_GUI_LOG):
        _Protocols().new_protocol()

    recorded = _recorded(caplog)
    assert 'BUTTON NEW_PROTOCOL ' in recorded, recorded
    assert not [line for line in recorded if line.startswith('PROTOCOL NEW')], recorded


def test_change_on_an_empty_protocol_is_recorded_and_refused_by_the_session(
    session, monkeypatch, caplog
):
    """Change on an empty protocol returned in the panel before its record,
    so the press left no line and no reason, where Delete was refused "No Such
    Step" by the Session."""
    from modules.notification_center import notifications

    reported = []
    monkeypatch.setattr(notifications, 'report_outcome', lambda exc, **_kw: reported.append(exc))
    # a stand-in by design: the stage map is a Kivy widget the redraw after the refusal draws on
    stage = SimpleNamespace(set_protocol_steps=lambda protocol: None)
    monkeypatch.setattr(_app_ctx, 'ctx', SimpleNamespace(session=session, stage=stage))
    panel = _Protocols()
    panel._protocol = session.create_empty_protocol()
    panel.curr_step = -1

    with caplog.at_level(logging.INFO, logger=_GUI_LOG):
        panel.modify_step()

    assert any(line.startswith('PROTOCOL MODIFY_STEP') for line in _recorded(caplog)), _recorded(
        caplog
    )
    assert [(type(e).__name__, e.title) for e in reported] == [
        ('StepNotFoundError', 'No Such Step')
    ]


@pytest.mark.parametrize(
    ('handler', 'box', 'record'),
    (
        ('set_xposition', 'x_pos_id', 'SET_X_POSITION'),
        ('set_yposition', 'y_pos_id', 'SET_Y_POSITION'),
    ),
)
def test_an_unparseable_stage_entry_is_recorded_as_a_refusal_and_the_box_goes_back(
    monkeypatch, handler, box, record
):
    """What the kv float filter lets through but is not a number moves
    nothing. The refusal leaves a line saying so -- the name alone would read
    as a successful move -- and then the box shows the target again, and
    that is recorded too. One entry: which entries are not numbers ('',
    '-', '.', '-.') is typed_number's to decide, pinned in
    test_a_typed_non_number_puts_its_box_back.py; here every one reaches the
    same branch."""
    import ui.motion_settings as motion
    from modules import gui_logger

    typed = '-'

    lines = []
    monkeypatch.setattr(gui_logger, 'button', lambda name, detail='': lines.append((name, detail)))
    monkeypatch.setattr(gui_logger, 'text_input', lambda name, value: lines.append((name, value)))
    moves = []
    monkeypatch.setattr(motion, 'move_to_plate', lambda *a, **k: moves.append((a, k)))
    monkeypatch.setattr(
        _app_ctx, 'ctx', SimpleNamespace(session=SimpleNamespace(controls_locked=False))
    )
    stand = SimpleNamespace(ids={box: SimpleNamespace(text=typed)})
    stand.update_gui = lambda: setattr(stand.ids[box], 'text', '12.50')

    getattr(motion.XYStageControl, handler)(stand, typed)

    assert moves == []
    assert stand.ids[box].text == '12.50'
    assert lines == [(record, f'refused: {typed!r}'), (f'{record}_APPLIED', '12.50')]
