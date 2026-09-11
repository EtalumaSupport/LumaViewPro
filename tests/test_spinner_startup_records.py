"""A panel populating its own spinner must not record a selection nobody made.

Measured on the LS850T, bundle SN12075-2026-09-11-105735, build b68b5349: four
launches produced six ``SELECT ZPROJECTION_METHOD`` records and NONE of them was
a user action. They arrive as a ``Min``/``Max`` pair in the same second, once per
launch, because ``_init_ui`` assigns ``.values`` (which snaps the text to the
first option) and then ``.text``. ``SELECT ZSTACK_REFERENCE_POSITION`` showed
the same shape once per launch, from settings restore.

A spinner cannot tell an assignment from a pick -- both dispatch the same event.
So the writer declares what it is about to write, and the emitter drops the
record that matches. These tests drive the emitter directly.
"""

import logging

from modules import gui_logger


def test_a_declared_write_produces_no_record(caplog):
    """The startup case: declare, then the assignment's event arrives."""
    gui_logger._write_backs.clear()
    with caplog.at_level(logging.INFO, logger='LVP.gui_interactions'):
        gui_logger.note_write_back('ZPROJECTION_METHOD', 'Min')
        gui_logger.select('ZPROJECTION_METHOD', 'Min')

    assert caplog.records == [], (
        f'a panel populating itself still records a selection: '
        f'{[r.getMessage() for r in caplog.records]}'
    )


def test_the_measured_startup_pair_produces_nothing(caplog):
    """Both assignments in _init_ui, in order, exactly as the bundle showed."""
    gui_logger._write_backs.clear()
    methods = ('Min', 'Max', 'Mean')
    with caplog.at_level(logging.INFO, logger='LVP.gui_interactions'):
        gui_logger.note_write_back('ZPROJECTION_METHOD', methods[0])
        gui_logger.select('ZPROJECTION_METHOD', methods[0])  # .values dispatch
        gui_logger.note_write_back('ZPROJECTION_METHOD', methods[1])
        gui_logger.select('ZPROJECTION_METHOD', methods[1])  # .text dispatch

    assert caplog.records == [], (
        'the Min/Max startup pair measured on hardware is still being recorded'
    )


def test_a_real_selection_after_startup_is_recorded(caplog):
    """The suppression absorbs exactly one write, not the user's next pick."""
    gui_logger._write_backs.clear()
    with caplog.at_level(logging.INFO, logger='LVP.gui_interactions'):
        gui_logger.note_write_back('ZPROJECTION_METHOD', 'Max')
        gui_logger.select('ZPROJECTION_METHOD', 'Max')  # startup, dropped
        gui_logger.select('ZPROJECTION_METHOD', 'Max')  # the user picks it, kept

    assert [r.getMessage() for r in caplog.records] == ['SELECT ZPROJECTION_METHOD Max']


def test_an_undeclared_selection_is_always_recorded(caplog):
    """The guard must never swallow a pick nobody declared."""
    gui_logger._write_backs.clear()
    with caplog.at_level(logging.INFO, logger='LVP.gui_interactions'):
        gui_logger.select('ZSTACK_REFERENCE_POSITION', 'Current Position at Top')

    assert [r.getMessage() for r in caplog.records] == [
        'SELECT ZSTACK_REFERENCE_POSITION Current Position at Top'
    ]


def test_a_declaration_does_not_survive_a_different_value(caplog):
    """A stale declaration must not linger and eat a later unrelated record."""
    gui_logger._write_backs.clear()
    with caplog.at_level(logging.INFO, logger='LVP.gui_interactions'):
        gui_logger.note_write_back('ZSTACK_REFERENCE_POSITION', 'Center')
        gui_logger.select('ZSTACK_REFERENCE_POSITION', 'Top')  # not the declared one
        gui_logger.select('ZSTACK_REFERENCE_POSITION', 'Center')  # must still record

    assert [r.getMessage() for r in caplog.records] == [
        'SELECT ZSTACK_REFERENCE_POSITION Top',
        'SELECT ZSTACK_REFERENCE_POSITION Center',
    ]
