# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Autofocus All Steps writes the focus of a field with nothing to focus on.

The stated cost of an autofocus that never refuses a curve's shape
(``AutofocusRunner.run``): a field with nothing to focus on focuses where
its final pass scored highest and succeeds, so Autofocus All Steps writes
that focus into the person's protocol, moving the step's Z within the
objective's autofocus range of where it was. Telling an empty field from a
faint sample waits on a measure from labelled bench data.
"""

from __future__ import annotations

from tests.test_zstack_group_autofocus import (
    LAYER_FOCUS_UM,
    WAIT_S,
    focusing_session,  # noqa: F401 -- the session the scan runs in
)

# The 10x objective's autofocus range either side of the step's Z.
_RANGE_UM = 75.0


def test_autofocus_all_steps_on_a_field_with_nothing_in_it_writes_that_focus(
    focusing_session, monkeypatch
):
    """Every score within about 1% of the others in no order, as the LS720's
    sweep with no sample scored (2026-10-14)."""
    scope = focusing_session.scope
    monkeypatch.setattr(
        '_lumascope.modules.autofocus_functions.autofocus_score',
        lambda image: 1.75e7 + 2e5 * ((scope.motion.get_current_position('Z') * 7.31) % 1.0),
    )
    focusing_session.set_channel_settings('BF', autofocus=False)
    protocol = focusing_session.new_protocol()

    outcome = (
        focusing_session.protocol_runner()
        .start_autofocus_all_steps(protocol)
        .result(timeout=WAIT_S)
    )

    assert outcome.status == 'completed', outcome
    assert outcome.focus_written is True, outcome
    written = protocol.step_row(step_index=0)['Z']
    assert written != LAYER_FOCUS_UM
    assert abs(written - LAYER_FOCUS_UM) <= _RANGE_UM, written
