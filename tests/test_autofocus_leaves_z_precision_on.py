# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Autofocus leaves Z's precision on at every exit.

Autofocus turns Z's precision off for its coarse passes, for search speed,
and on for its fine pass. Each way out -- a reset, an abort, an exception,
success -- turns it back on, so the protocol's next Z move stops at the
tight threshold. Before this, reset() turned it off, and the protocol's
step runner calls reset() after every step, so Z ran loose for every move
after the first autofocus.
"""

import threading
from unittest.mock import patch

import pytest

from modules.autofocus_runner import AutofocusRunner
from modules.exceptions import AutofocusAborted
from tests.af_drives import park_z
from tests.protocol_drives import held_run_claim
from tests.scope_fakes import bind_settings_like_a_session, build_scope, home_sim_scope


def _build_af():
    scope = build_scope(simulate=True)
    bind_settings_like_a_session(scope)
    home_sim_scope(scope)
    park_z(scope, 5000.0)
    return AutofocusRunner(scope=scope), scope


def test_reset_restores_precision_on():
    af, scope = _build_af()
    with patch.object(scope.motion, 'set_precision_mode') as mock_set:
        af.reset()
        mock_set.assert_called_with(True)


def test_abort_path_restores_precision_on():
    # The run's finally turns precision on for an abort as it does for
    # success and an exception, so Z is never left loose by any way out.
    af, scope = _build_af()
    run_lease = scope.illumination.acquire_led_lease('protocol', claim=held_run_claim())
    abort_event = threading.Event()
    abort_event.set()  # pre-set so the run unwinds through its abort
    with (
        patch.object(scope.motion, 'set_precision_mode') as mock_set,
        patch.object(af, '_move_absolute_position'),
        patch.object(scope.camera, 'restore_camera_state'),
    ):
        with pytest.raises(AutofocusAborted):
            af.run(
                objective_id='4x Oly',
                center_um=scope.motion.get_current_position('Z'),
                abort_event=abort_event,
                led_color=None,
                led_illumination=0.0,
                led_lease=run_lease,
            )
        calls = [tuple(c.args) for c in mock_set.call_args_list]
        assert (True,) in calls, f'abort path must restore Z precision_mode=True; got calls {calls}'
