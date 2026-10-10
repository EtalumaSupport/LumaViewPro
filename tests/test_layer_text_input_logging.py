"""A typed commit in a layer panel is recorded once, as typed, under its channel's name.

Every numeric text box in the layer panel commits through one shared helper,
``LayerControl._validate_and_apply_text_input``, and the helper -- not each of
its callers -- writes the record, so a new box records by routing through it.
The record name is derived from the box's id and the panel's channel
(``exp_text`` on the Blue panel records ``EXP_Blue``); no caller passes one.

The record is ``gui_logger.text_input``, synchronous and unconditional, with
``<NAME>_APPLIED`` beside it only when what is stored differs from what was
typed: the writer refused it, or it was not a number and the box went back.

Each test drives the helper on a ``LayerControl`` built without Kivy, with the
real range check behind the settings write (``settings_writer``), and reads
what landed on the ``LVP.gui_interactions`` logger.
"""

import logging
from types import SimpleNamespace

import modules.app_context as _app_ctx
from tests.settings_fixtures import settings_writer
from ui.layer_control import LayerControl

_GUI_LOG = 'LVP.gui_interactions'


def _records(caplog):
    return [r.getMessage() for r in caplog.records if r.name == _GUI_LOG]


class _Slider:
    def __init__(self, low, high, value):
        self.min = low
        self.max = high
        self.value = value


def _blue_panel(monkeypatch, caplog, typed, *, stored=500.0):
    """The Blue panel's exposure box holding ``typed``, over a store holding ``stored``.

    Returns the panel, the store, and one entry per settings write: the
    records already in the log when the write happened.
    """
    caplog.set_level(logging.INFO, logger=_GUI_LOG)
    settings = {'Blue': {'exposure_ms': stored}, 'fx2_debug_wire_enabled': False}
    write = settings_writer(settings)
    writes = []

    def update_settings(path, value):
        writes.append(_records(caplog))
        write(path, value)

    monkeypatch.setattr(
        _app_ctx, 'ctx', SimpleNamespace(settings=settings, update_settings=update_settings)
    )
    # a stand-in by design: Kivy is stubbed in the test process; the subject is the handler's record
    panel = LayerControl.__new__(LayerControl)
    panel.layer = 'Blue'
    panel._initializing = False
    panel.ids = {
        'exp_slider': _Slider(0.01, 1000.0, stored),
        'exp_text': SimpleNamespace(text=typed),
        'logHistogram_id': SimpleNamespace(active=True),
    }
    return panel, settings, writes


def _commit(panel):
    """The helper as every caller calls it: the box, its slider and the key, no record name."""
    return panel._validate_and_apply_text_input('exp_text', 'exp_slider', 'exposure_ms')


def test_layer_owned_records_carry_the_channel_suffix(monkeypatch, caplog):
    """A layer box and the log-histogram box each name the channel they belong to.

    LayerControl is built once per channel with identical child ids, so a record
    name without the channel would leave a bundle unable to say which channel
    was edited. The helper is called with no record name, as its callers call
    it, so a name that went unpassed would leave no line here either.

    Fails if the helper's record name (``record_name``) or
    ``log_histogram_scale``'s toggle name drops ``self.layer``.
    """
    panel, _settings, _writes = _blue_panel(monkeypatch, caplog, '250')

    _commit(panel)
    panel.log_histogram_scale()

    assert _records(caplog) == ['TEXT_INPUT EXP_Blue 250', 'TOGGLE LOG_HISTOGRAM_Blue ON']


def test_the_record_carries_the_text_as_typed_not_the_number_stored(monkeypatch, caplog):
    """'2.50' is stored as 2.5 and recorded as '2.50': the record is what the person typed.

    Fails if the bare record is written with the parsed or the stored value
    (``raw`` or ``stored``) instead of ``typed_text``.
    """
    panel, settings, _writes = _blue_panel(monkeypatch, caplog, '2.50')

    _commit(panel)

    assert settings['Blue']['exposure_ms'] == 2.5
    assert _records(caplog) == ['TEXT_INPUT EXP_Blue 2.50']


def test_a_refused_entry_reports_the_value_kept_under_its_own_name(monkeypatch, caplog):
    """An entry the writer refuses records the attempt, then ``_APPLIED`` with what is kept.

    Fails if the ``_APPLIED`` emit after the write is removed, or written under
    the bare name, which would leave a reader unable to tell the entry from
    the correction.
    """
    panel, settings, _writes = _blue_panel(monkeypatch, caplog, '-5')

    _commit(panel)

    assert settings['Blue']['exposure_ms'] == 500.0
    typed_records = [m for m in _records(caplog) if m.startswith('TEXT_INPUT')]
    assert typed_records == ['TEXT_INPUT EXP_Blue -5', 'TEXT_INPUT EXP_Blue_APPLIED 500.0']


def test_an_accepted_whole_number_in_a_float_box_reports_no_correction(monkeypatch, caplog):
    """'5' into a float box is stored as 5.0, the same number, so no ``_APPLIED`` line.

    Fails if the correction guard compares the strings (``typed_text !=
    str(stored)``) instead of the parsed numbers, or is dropped.
    """
    panel, settings, _writes = _blue_panel(monkeypatch, caplog, '5')

    _commit(panel)

    assert settings['Blue']['exposure_ms'] == 5.0
    assert _records(caplog) == ['TEXT_INPUT EXP_Blue 5']


def test_an_unparseable_entry_records_the_attempt_and_the_value_kept(monkeypatch, caplog):
    """'-' passes the kv float filter; nothing is written, and both lines are recorded.

    Fails if the ``raw is None`` branch returns without its two records.
    """
    panel, _settings, writes = _blue_panel(monkeypatch, caplog, '-')

    _commit(panel)

    assert writes == []
    assert panel.ids['exp_text'].text == '500.0'
    assert _records(caplog) == ['TEXT_INPUT EXP_Blue -', 'TEXT_INPUT EXP_Blue_APPLIED 500.0']


def test_the_helper_does_not_emit_through_the_slider_verb(monkeypatch, caplog):
    """A typed commit and a drag on the same setting stay distinguishable.

    The slider twin records ``SLIDER EXP_Blue``; a typed commit records
    ``TEXT_INPUT`` and never ``SLIDER``, or the bundle could not tell a
    keystroke from a drag.

    Fails if the helper records through ``gui_logger.slider``.
    """
    panel, _settings, _writes = _blue_panel(monkeypatch, caplog, '250')

    _commit(panel)

    assert not [m for m in _records(caplog) if m.startswith('SLIDER')], _records(caplog)


def test_the_typed_record_is_written_after_the_settings_write(monkeypatch, caplog):
    """Pins today's record-after-write order in the layer helper; flips with its fix,
    the row's contract question 3.

    ``gui_logger.text_input``'s contract is that the record lands ahead of
    whatever the entry goes on to do. This helper writes the store first and
    records after, so a refusal's notification reaches the log ahead of the
    entry that caused it. The test asserts today's order so that it passes
    against today's code; the fix moves the record ahead of the write and
    turns this assertion into ``writes == [['TEXT_INPUT EXP_Blue 250']]``.

    Fails if the record moves ahead of the write (the fix), which is when
    this test is rewritten to the contract's side.
    """
    panel, _settings, writes = _blue_panel(monkeypatch, caplog, '250')

    _commit(panel)

    assert writes == [[]], f'records already in the log at the settings write: {writes}'
    assert _records(caplog) == ['TEXT_INPUT EXP_Blue 250']
