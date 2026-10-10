# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A scope, camera, board, session or sub-API in a test is the simulator, not a stand-in.

A ``MagicMock`` scope answers every question with a truthy MagicMock, so a
production branch that probes a name the real scope lacks stays green until
the bench (``led_on_fast`` and ``camera`` were found that way); a scope
built through ``__new__`` has no constructor-set state, so the member under
test runs over attributes the test chose rather than the ones bring-up
sets; a ``SimpleNamespace`` of sub-APIs answers only the calls the author
foresaw. The simulated scope is the cheap owner of every scope-level claim
(about 110 ms a session on the fast tier, measured 2026-10-09), and Eric's
word is *"i'd like to be using as much of our code as possible"*.

This walk over every test module refuses four forms:

* a ``MagicMock(...)``/``Mock(...)`` literal assigned to, or passed as a
  keyword under, a scope-shaped name (``scope``, ``camera``, ``session``,
  ``board``, ``driver``, as the name's last word, or a sub-API's name);
* ``spec_scope(...)`` and ``create_autospec(...)``;
* a ``SimpleNamespace(...)`` with a sub-API or ``camera_connected`` or
  ``objective_helper`` keyword;
* ``<class>.__new__(...)`` on a scope, session, board, camera, driver or
  API class.

The allowlist below was seeded from the census at the trunk tip the day
the guard landed and only shrinks: a site not listed is refused, an entry
with no site left is deleted, and the entry count is pinned, lowered by
the commit that removes entries. A stand-in that is right by design is
marked where it is built, on its statement's line or the line above::

    # a stand-in by design: <why>

The two reasons such a mark can give (``/writing-tests`` question 4): the
subject is a catcher and the stand-in is its interchangeable collaborator,
or the stand-in produces a fault the simulator cannot yet. A marked site is
exempt and counted; the count is announced at the end of every run.
"""

from __future__ import annotations

import ast

from tests import ratchets
from tests.ast_seams import REPO_ROOT, iter_package_modules

MARK = '# a stand-in by design:'
SELF = 'tests/guards/test_a_scope_is_the_simulator_not_a_stand_in.py'

NOUNS = frozenset({'scope', 'camera', 'session', 'board', 'driver'})
SUB_APIS = frozenset({'imaging', 'motion', 'illumination', 'diagnostics', 'protocols'})
CLASS_TAILS = ('scope', 'session', 'board', 'camera', 'driver', 'api')
NAMESPACE_KEYWORDS = frozenset(
    {'imaging', 'motion', 'illumination', 'camera_connected', 'objective_helper'}
)


def _is_scope_shaped(name: str) -> bool:
    words = [word for word in name.lower().split('_') if word]
    return bool(words) and (words[-1] in NOUNS or name.strip('_') in SUB_APIS)


def _is_scope_shaped_class(dotted: str) -> bool:
    return dotted.rsplit('.', 1)[-1].lower().endswith(CLASS_TAILS)


def _call_name(node: ast.AST) -> str:
    if not isinstance(node, ast.Call):
        return ''
    func = node.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, 'id', '')


def _is_mock_literal(node: ast.AST) -> bool:
    return _call_name(node) in ('MagicMock', 'Mock')


def _bound_name(target: ast.AST) -> str:
    return target.attr if isinstance(target, ast.Attribute) else getattr(target, 'id', '')


class _Finder(ast.NodeVisitor):
    """Every site of the four forms, each with its enclosing def and statement line."""

    def __init__(self):
        self.scope: list[str] = []
        self.statement_line = 0
        self.found: list[tuple[str, int, str]] = []

    def _scoped(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _scoped

    def visit(self, node):
        if isinstance(node, ast.stmt):
            outer, self.statement_line = self.statement_line, node.lineno
            super().visit(node)
            self.statement_line = outer
        else:
            super().visit(node)

    def _record(self, what: str):
        self.found.append(('.'.join(self.scope) or '<module>', self.statement_line, what))

    def visit_Assign(self, node):
        if _is_mock_literal(node.value):
            for target in node.targets:
                name = _bound_name(target)
                if _is_scope_shaped(name):
                    self._record(f'{name} = {_call_name(node.value)}()')
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if node.value is not None and _is_mock_literal(node.value):
            name = _bound_name(node.target)
            if _is_scope_shaped(name):
                self._record(f'{name} = {_call_name(node.value)}()')
        self.generic_visit(node)

    def visit_Call(self, node):
        name = _call_name(node)
        for keyword in node.keywords:
            if keyword.arg and _is_scope_shaped(keyword.arg) and _is_mock_literal(keyword.value):
                self._record(f'{keyword.arg}={_call_name(keyword.value)}()')
        if name in ('spec_scope', 'create_autospec'):
            self._record(f'{name}(...)')
        elif name == 'SimpleNamespace':
            keys = sorted({kw.arg for kw in node.keywords if kw.arg in NAMESPACE_KEYWORDS})
            if keys:
                self._record(f'SimpleNamespace({", ".join(keys)}=...)')
        elif name == '__new__' and isinstance(node.func, ast.Attribute):
            cls = ast.unparse(node.func.value)
            if _is_scope_shaped_class(cls):
                self._record(f'{cls}.__new__(...)')
        self.generic_visit(node)


def _is_marked(lines: list[str], statement_line: int) -> bool:
    here = lines[statement_line - 1] if statement_line >= 1 else ''
    above = lines[statement_line - 2] if statement_line >= 2 else ''
    return MARK in here or MARK in above


def sites(source: str):
    """Yield ``(qualname, line, what, marked)`` for every site in ``source``."""
    finder = _Finder()
    finder.visit(ast.parse(source))
    lines = source.splitlines()
    for qualname, line, what in finder.found:
        yield qualname, line, what, _is_marked(lines, line)


def _found_in_tests():
    for rel_path, _tree in iter_package_modules(('tests',)):
        if rel_path == SELF:
            continue
        source = (REPO_ROOT / rel_path).read_text(encoding='utf-8')
        for qualname, line, what, marked in sites(source):
            yield f'{rel_path}::{qualname}', line, what, marked


def _entries_with_a_site() -> set[str]:
    return {entry for entry, _line, _what, marked in _found_in_tests() if not marked}


def _by_design() -> list[str]:
    return [f'{entry}:{line} {what}' for entry, line, what, marked in _found_in_tests() if marked]


# Seeded at LVP dev/4.0.0 b17419cc (2026-10-09) with 237 entries. Only shrinks: a
# commit that takes a stand-in to the simulator deletes its entry and lowers this.
ALLOWED_ENTRIES = 233
ALLOWED = frozenset(
    {
        'tests/af_drives.py::af_runner_and_scope',
        'tests/camera_fakes.py::bare_fx2_camera',
        'tests/camera_fakes.py::bare_ids_camera',
        'tests/camera_fakes.py::bare_pylon_camera',
        'tests/guards/test_live_display_hold_is_a_callback.py::_writer',
        'tests/guards/test_session_construction.py::TestCreateTakesTheHostInjections.test_create_takes_no_lanes',
        'tests/guards/test_session_construction.py::TestScopeOwnershipIsConstructorState.test_a_direct_construction_is_not_owned',
        'tests/guards/test_session_construction.py::TestScopeOwnershipIsConstructorState.test_a_passed_scope_is_not_owned',
        'tests/guards/test_session_construction.py::TestScopeOwnershipIsConstructorState.test_a_session_needs_a_bundle',
        'tests/guards/test_session_metrics_lifecycle.py::_make_session',
        'tests/plugin_test_harness.py::_make_ctx',
        'tests/scope_fakes.py::scope_delivering_nothing',
        'tests/test_a_basler_camera_publishes_the_formats_it_offers.py::_capabilities',
        'tests/test_a_camera_command_with_no_camera_is_refused.py::test_the_gui_camera_controls_follow_whether_a_camera_is_connected',
        'tests/test_a_camera_refusal_is_reported_once.py::TestTheRunReportsAndCarriesOn.test_the_autofocus_sweep_reports_a_refused_target_once',
        'tests/test_a_camera_refusal_is_reported_once.py::sim_imaging',
        'tests/test_a_camera_report_never_rewrites_a_settings_box.py::_bridge_with_an_open_blue_layer',
        'tests/test_a_camera_states_its_analog_gain_maximum.py::_capabilities',
        'tests/test_a_camera_states_its_analog_gain_maximum.py::test_no_camera_states_none',
        'tests/test_a_camera_without_auto_gain_is_not_asked.py::TestTheAnswerTheGuiDisplays.test_a_stored_preference_stands_on_a_camera_with_auto_gain',
        'tests/test_a_camera_without_auto_gain_is_not_asked.py::ag_less_imaging',
        'tests/test_a_failed_save_is_reported_once.py::test_an_sdk_setter_raises_the_driver_error_once_and_posts_nothing',
        'tests/test_a_failed_temperature_read_raises.py::test_the_classic_camera_has_no_sensor',
        'tests/test_a_failed_temperature_read_raises.py::test_the_support_report_writes_a_temperature_failure_as_one',
        'tests/test_a_gesture_is_one_lane_task.py::env',
        'tests/test_a_protocol_edit_is_the_apis_answer.py::ctx',
        'tests/test_a_recorded_frame_says_where_it_was_taken.py::_fake_scope',
        'tests/test_a_refusal_names_the_run_holding_the_scope.py::TestARawTokenNeverReachesAPerson.test_a_diagnostic_refused_by_a_run_names_the_run',
        'tests/test_a_refused_auto_mode_is_not_recorded.py::sim_imaging',
        'tests/test_a_refused_frame_listener_is_raised.py::TestAFailingHandlerIsBounded.test_a_handler_raising_k_frames_running_is_removed_with_one_traceback',
        'tests/test_a_refused_frame_listener_is_raised.py::TestAFailingHandlerIsBounded.test_a_handler_that_recovers_before_k_is_kept',
        'tests/test_a_refused_frame_listener_is_raised.py::TestARemovalIsReportedAndStopsCalls.test_an_over_budget_removal_is_one_warning_and_no_traceback',
        'tests/test_a_refused_step_navigation_changes_nothing.py::nav_env',
        'tests/test_a_run_reads_its_settings_once.py::_runner_over_a_store_edited_after_the_first_read',
        'tests/test_a_runs_writes_run_on_its_lanes.py::test_a_step_darken_the_lane_refuses_falls_to_the_safety_off',
        'tests/test_a_stalled_writer_is_reported_by_the_api.py::session',
        'tests/test_a_stalled_writer_refusal_names_its_remedy.py::session',
        'tests/test_a_step_list_edit_outruns_the_move.py::env',
        'tests/test_a_video_step_announces_its_finish.py::_scope',
        'tests/test_a_zstack_with_no_range_is_refused.py::TestTheHeadlessCallerGetsIt.test_run_zstack_refuses_instead_of_capturing_one_plane',
        'tests/test_an_led_toggle_is_recorded_only_when_pressed.py::layer',
        'tests/test_an_out_of_range_camera_setting_is_refused.py::sim_imaging',
        'tests/test_an_unknown_objective_is_shown_as_a_refusal.py::TestNewProtocol.panel',
        'tests/test_audit_fixes.py::TestAcquisitionStopModeSetter._make_scope_with_fake_camera',
        'tests/test_audit_fixes.py::TestAcquisitionStopModeSetter.test_ids_driver_stub_returns_false',
        'tests/test_audit_fixes.py::TestAcquisitionStopModeSetter.test_no_camera_returns_false',
        'tests/test_audit_fixes.py::TestDeviceLinkThroughputLimitSetter._make_scope_with_fake_camera',
        'tests/test_audit_fixes.py::TestDeviceLinkThroughputLimitSetter.test_no_camera_returns_false',
        'tests/test_audit_fixes.py::TestDisconnectRaisesAfterTheTeardown.test_disconnect_camera_failure_raises',
        'tests/test_audit_fixes.py::TestEnterEngineeringModeRaises._make_ledboard',
        'tests/test_audit_fixes.py::TestFrameValidityIsL2Stable.test_frame_validity_is_publicly_named',
        'tests/test_audit_fixes.py::TestG4_MotorLogSuppression._make_failing_board',
        'tests/test_audit_fixes.py::TestG4_MotorLogSuppression.test_connect_error_logging_resumes_after_success.ok_open',
        'tests/test_audit_fixes.py::TestGigeSetters._make_scope_with_fake_camera',
        'tests/test_audit_fixes.py::TestGigeSetters.test_no_camera_returns_false_for_all',
        'tests/test_audit_fixes.py::TestPIW3_FalseColor16bitCachedAtRunStart.test_save_image_threads_param_to_write_tiff',
        'tests/test_audit_fixes.py::TestProtocolCleanupLedRestoreKey.test_restore_uses_illumination_ma_key',
        'tests/test_audit_fixes.py::TestPylonAcquisitionIdleWait.test_idle_wait_returns_false_when_node_absent',
        'tests/test_audit_fixes.py::TestPylonAcquisitionIdleWait.test_idle_wait_returns_true_when_already_idle',
        'tests/test_audit_fixes.py::TestPylonAcquisitionIdleWait.test_idle_wait_returns_true_when_inactive',
        'tests/test_audit_fixes.py::TestPylonAcquisitionIdleWait.test_idle_wait_times_out_when_stuck_active',
        'tests/test_audit_fixes.py::TestPylonDeviceNotFoundClassification.test_device_not_found_marks_disconnected_immediately',
        'tests/test_audit_fixes.py::TestPylonDeviceNotFoundClassification.test_device_not_found_skips_stage_b_and_failure_counter',
        'tests/test_audit_fixes.py::TestPylonDeviceNotFoundClassification.test_other_grab_failures_hand_off_to_stage_b',
        'tests/test_audit_fixes.py::TestPylonDiagnosticProbe._make_scope_with_fake_camera',
        'tests/test_audit_fixes.py::TestPylonOnImageGrabbedExceptionContext.test_on_image_grabbed_outer_except_uses_contextual_message',
        'tests/test_audit_fixes.py::TestPylonOnImageGrabbedOwningCopy._grab_with_copy_recorder',
        'tests/test_audit_fixes.py::TestPylonStateMutationViaMarkDisconnected.test_on_camera_device_removed_uses_mark_disconnected',
        'tests/test_audit_fixes.py::TestPylonStateMutationViaMarkDisconnected.test_on_image_grabbed_inactive_branch_uses_mark_disconnected',
        'tests/test_audit_fixes.py::TestPylonStreamGrabberStatusLog.test_log_helper_handles_missing_node_gracefully',
        'tests/test_audit_fixes.py::TestPylonStreamGrabberStatusLog.test_log_helper_handles_runtime_exception',
        'tests/test_audit_fixes.py::TestPylonStreamGrabberStatusLog.test_log_helper_no_op_when_active_none',
        'tests/test_audit_fixes.py::TestSequencedCaptureRunnerRunDirCollision._make_executor',
        'tests/test_audit_fixes.py::TestStreamGrabberSetters._make_scope_with_fake_camera',
        'tests/test_audit_fixes.py::TestStreamGrabberSetters.test_no_camera_returns_false_for_both',
        'tests/test_audit_fixes.py::_sim_backed_imaging',
        'tests/test_auto_gain_lock.py::_build',
        'tests/test_auto_gain_lock.py::_build_inert',
        'tests/test_autofocus_locks_auto_gain_once.py::_runner_with_step_targets',
        'tests/test_autofocus_result_run_scope.py::gui',
        'tests/test_autogain_support_capability_gate.py::TestCapabilityMapping._caps_for',
        'tests/test_camera_capabilities.py::TestScopeCapabilitiesIntegration._stub_motion',
        'tests/test_camera_getter_sentinel_containment.py::_build_imaging',
        'tests/test_camera_getter_sentinel_containment.py::_metadata_scope_with_real_imaging',
        'tests/test_camera_getter_sentinel_containment.py::test_writer_saves_capture_time_depth_not_save_time_rederivation',
        'tests/test_camera_profile_info_reports_its_failure.py::_scope_with',
        'tests/test_camera_rejection_reaches_the_caller.py::sim_imaging',
        'tests/test_camera_write_authority.py::_build_imaging',
        'tests/test_capture_collision_policy.py::test_video_step_row_records_writers_actual_path',
        'tests/test_capture_diag_debug_gate.py::_drive_capture',
        'tests/test_capture_evidence_saturation_depth.py::_make_writer',
        'tests/test_composite_run_outcome.py::_runner',
        'tests/test_composite_starter.py::app_ctx',
        'tests/test_diagnostic_claim.py::_make_session',
        'tests/test_diagnostic_query_warning_suppression.py::TestDiagnosticQueryCapabilityProbe._make_board',
        'tests/test_diagnostic_query_warning_suppression.py::TestDiagnosticQueryCapabilityProbe.test_diagnostic_query_returns_none_on_error_response',
        'tests/test_diagnostic_query_warning_suppression.py::TestDiagnosticQueryCapabilityProbe.test_diagnostic_query_returns_response_when_supported',
        'tests/test_exchange_multiline_logs_content.py::_make_board',
        'tests/test_exchange_multiline_logs_content.py::test_the_port_timeout_is_not_changed_while_the_reply_arrives',
        'tests/test_exposure_chunk_target_is_the_applied_value.py::_imaging_with',
        'tests/test_fatal_abort_led_safety.py::_run_cleanup_capture_led_ctx',
        'tests/test_firmware_updater.py::TestBackupConfigs._make_board',
        'tests/test_firmware_updater.py::TestRestoreConfigs._make_board',
        'tests/test_firmware_updater.py::TestRestoreConfigs.test_empty_config_data_returns_true',
        'tests/test_firmware_updater.py::TestSendFwupdateCommand.test_disconnects_on_error',
        'tests/test_firmware_updater.py::TestSendFwupdateCommand.test_disconnects_on_success',
        'tests/test_firmware_updater.py::TestSendFwupdateCommand.test_fallback_to_raw_repl_on_not_found',
        'tests/test_firmware_updater.py::TestSendFwupdateCommand.test_led_sends_fwupdate',
        'tests/test_firmware_updater.py::TestSendFwupdateCommand.test_motor_sends_fwupdate',
        'tests/test_firmware_updater.py::TestSendFwupdateCommand.test_raises_on_exchange_exception',
        'tests/test_firmware_updater.py::TestUpdateFirmwareSameVersion.test_same_version_returns_success',
        'tests/test_frame_carries_depth.py::TestGetImageFromBufferUsesFrameDepth.test_downconvert_uses_frame_depth_not_live_driver',
        'tests/test_frame_metadata_from_chunk.py::_imaging',
        'tests/test_frame_validity.py::TestCaptureTimeChunkVerification._make_imaging',
        'tests/test_frame_validity_invariant.py::sim_imaging',
        'tests/test_fresh_process_reads_homed_state.py::TestAxisStateSeeding.test_homed_hardware_seeds_idle_so_the_gate_permits_motion',
        'tests/test_fresh_process_reads_homed_state.py::TestAxisStateSeeding.test_partially_homed_hardware_seeds_each_axis_on_its_own_answer',
        'tests/test_fresh_process_reads_homed_state.py::TestAxisStateSeeding.test_seeding_requires_the_hardware_answer',
        'tests/test_fresh_process_reads_homed_state.py::TestAxisStateSeeding.test_unhomed_hardware_still_seeds_unknown',
        'tests/test_issue_239_first_interval_anchor.py::_make_parent',
        'tests/test_issue_653_skip_guard.py::handler_and_log',
        'tests/test_issue_653_skip_guard.py::test_live_skip_is_logged',
        'tests/test_issue_653_skip_guard.py::test_removal_skip_burst_is_silent',
        'tests/test_issue_653_skip_guard.py::test_zero_count_skip_is_silent',
        'tests/test_issue_655_ag_ae_exposure_cap.py::_video_session_autogain_call',
        'tests/test_issue_655_ag_ae_exposure_cap.py::test_api_set_auto_gain_forwards_cap_from_settings_dict',
        'tests/test_issue_679_exposure_cache_after_auto.py::sim_imaging',
        'tests/test_issue_690_jpg_orientation.py::test_tiff_and_jpg_save_identical_orientation',
        'tests/test_issue_697_nav_led_sweep.py::_reconcile_harness',
        'tests/test_issue_733_step_nav_preview_led_button.py::stepnav_env',
        'tests/test_jpg_export.py::_scope_with_depth',
        'tests/test_layer_control_ag_exposure_floor.py::_leave',
        'tests/test_leaving_auto_gain_is_a_session_member.py::_session',
        'tests/test_leaving_auto_gain_is_a_session_member.py::test_the_toggle_submits_the_member_on_the_camera_lane_and_redraws',
        'tests/test_led_ack_and_tsr_filename.py::TestLedDriverRejectsNonAckResponses._make_led',
        'tests/test_led_ack_and_tsr_filename.py::TestLedExitEngineeringRecoversFromWedge._make_led',
        'tests/test_led_ack_and_tsr_filename.py::TestLedOnBlockAckShape._make_led',
        'tests/test_led_ack_and_tsr_filename.py::TestLogsOnlySerialNumberLookupChain._build_report_stub',
        'tests/test_led_ack_and_tsr_filename.py::TestTechSupportReportPassesTimeoutS.test_cmd_and_read_multiline_pass_timeout_s_to_diagnostics',
        'tests/test_led_ack_and_tsr_filename.py::TestTsrLedEngineeringRoutesThroughDriver.test_enter_engineering_answers_the_refusals_words_when_led_absent',
        'tests/test_led_ack_and_tsr_filename.py::TestTsrLedEngineeringRoutesThroughDriver.test_enter_engineering_calls_sub_api',
        'tests/test_led_ack_and_tsr_filename.py::TestTsrLedEngineeringRoutesThroughDriver.test_exit_engineering_calls_sub_api',
        'tests/test_led_button_run_end_reconcile.py::_build_env',
        'tests/test_led_cap_published_by_driver.py::_caps_with',
        'tests/test_led_supports_firmware_stim.py::TestSupportsFirmwareStimProbe._make_led',
        'tests/test_led_supports_firmware_stim.py::TestSupportsFirmwareStimProbe.test_driver_none_returns_false_without_probing',
        'tests/test_lumascope_api.py::TestLEDChannelDiscovery.test_ledboard_available_channels_from_color_map',
        'tests/test_lumascope_api.py::TestMotorBoardGetMicroscopeModelDisconnect.test_returns_none_when_model_key_missing',
        'tests/test_lumascope_api.py::TestProtocolConformance.test_ledboard_satisfies_protocol',
        'tests/test_lumascope_api.py::TestProtocolConformance.test_motorboard_satisfies_protocol',
        'tests/test_manual_capture_overlay_save.py::capture_ctx',
        'tests/test_metrics_tick_cost.py::_make_session',
        'tests/test_motor_capability_probes.py::_make_board',
        'tests/test_motorboard_fullinfo_legacy_firmware.py::_make_board',
        'tests/test_motorconfig_provenance.py::TestLoaderRecordsProvenance._board_with_config',
        'tests/test_motorconfig_provenance.py::bare_board',
        'tests/test_notification_supersession.py::TestARefusalThatNamesItsRemedyIsAnOffer.test_the_offer_s_confirm_applies_the_remedy_through_the_session',
        'tests/test_objective_info_boundary_is_honest.py::_turretless_state',
        'tests/test_preview_lut_buffer_reuse.py::test_get_image_from_buffer_reuses_caller_buffer',
        'tests/test_protocol_engine_keeps_backlash_compensation.py::_runner_capturing_moves',
        'tests/test_protocol_modules.py::TestProtocolImageWriterWriteCapture._make_writer',
        'tests/test_protocol_modules.py::TestRunCleanup.test_the_camera_restore_is_the_public_member',
        'tests/test_protocol_move_io_ordering.py::_step_runner',
        'tests/test_protocol_overwrite_guard.py::test_protocol_image_writer_uses_if_collision',
        'tests/test_protocol_run_loop_iterate_scan_count.py::_make_two_scan_parent',
        'tests/test_pylon_gain_reports_its_rejection.py::pylon_imaging',
        'tests/test_pylon_hardware.py::pylon_imaging',
        'tests/test_pylon_setter_short_circuit.py::_mock_camera',
        'tests/test_regression_p2.py::_make_serial_board',
        'tests/test_run_autofocus_entry_point.py::_runner',
        'tests/test_run_encoding_ssot.py::TestOneRunOneEncoding._writer',
        'tests/test_run_leaves_layer_settings_alone.py::_scope_with_auto_gain',
        'tests/test_run_outcome_reports_autofocus_data.py::TestTheSweepDoesNotReturnBeforeItsWriteLands._bare_runner',
        'tests/test_run_zstack_entry_point.py::_runner',
        'tests/test_saving_an_unknown_position_is_refused_once.py::unknown',
        'tests/test_scope_api.py::TestGetCurrentPlatePosition.test_a_manual_scope_still_answers_the_origin',
        'tests/test_scope_api.py::TestGetCurrentPlatePosition.test_an_expected_motor_board_that_is_absent_is_refused',
        'tests/test_scope_api.py::_make_mock_scope',
        'tests/test_scope_bringup.py::_make_spec_session',
        'tests/test_scroll_wheel_routing.py::TestWheelDirectionIsConsistent._scroll_live_image',
        'tests/test_serial_logging_coverage.py::test_stim_probe_logs_to_serial_log',
        'tests/test_serial_safety.py::TestExchangeCommandStopOnEmpty._make_led_board',
        'tests/test_serial_safety.py::TestLEDBoardCommands._make_board',
        'tests/test_serial_safety.py::TestLEDBoardConversion.test_ch2color_all',
        'tests/test_serial_safety.py::TestLEDBoardConversion.test_ch2color_unknown_is_none',
        'tests/test_serial_safety.py::TestLEDBoardConversion.test_color2ch_all',
        'tests/test_serial_safety.py::TestLEDBoardConversion.test_color2ch_unknown_is_none',
        'tests/test_serial_safety.py::TestLEDBoardConversion.test_roundtrip_all_channels',
        'tests/test_serial_safety.py::TestLEDBoardLocking._make_board',
        'tests/test_serial_safety.py::TestLEDBoardStateLock._make_board',
        'tests/test_serial_safety.py::TestLEDFirmwareVersion._make_board',
        'tests/test_serial_safety.py::TestLEDNoneHandling._make_board',
        'tests/test_serial_safety.py::TestMotorBoardCommands._make_board',
        'tests/test_serial_safety.py::TestMotorBoardConversions._make_board',
        'tests/test_serial_safety.py::TestMotorBoardFullinfo._make_board',
        'tests/test_serial_safety.py::TestMotorBoardHoming._make_board',
        'tests/test_serial_safety.py::TestMotorBoardMovement._make_board',
        'tests/test_serial_safety.py::TestMotorBoardSafety._make_board',
        'tests/test_serial_safety.py::TestMotorBoardStateLock._make_board',
        'tests/test_serial_safety.py::TestMotorFirmwareVersion._make_board',
        'tests/test_serial_safety.py::TestSerialDesyncRecovery._make_board',
        'tests/test_serial_safety.py::TestSilentBoardHandling._make_silent_board',
        'tests/test_session_run_state.py::_make_session',
        'tests/test_session_teardown.py::TestACallersScopeIsLeftAlone._caller_session',
        'tests/test_session_teardown.py::TestACallersScopeIsLeftAlone.test_a_directly_constructed_session_is_the_same_row',
        'tests/test_set_exposure_ms_warning_threshold.py::_warnings_for',
        'tests/test_stage_crosshair_after_home.py::stage_and_motion',
        'tests/test_still_pending_writes.py::_writer',
        'tests/test_stored_camera_setting_survives_a_smaller_camera.py::imaging',
        'tests/test_tech_support_report_holds_the_scope.py::_make_session',
        'tests/test_the_autofocus_button_runs_through_run_autofocus.py::pressed',
        'tests/test_the_camera_holds_a_layer_from_the_session.py::TestBringUp.test_a_scope_whose_layers_are_unresolved_still_comes_up',
        'tests/test_the_camera_holds_a_layer_from_the_session.py::TestBringUp.test_a_scope_with_no_camera_is_not_applied_and_says_so',
        'tests/test_the_capture_line_logs_the_frames_settings.py::test_a_camera_with_chunks_logs_the_frames_own_values_unmarked',
        'tests/test_the_claim_names_the_holding_run.py::TestTheRunnerReadsTheClaim.test_a_runner_cannot_be_built_without_a_claim',
        'tests/test_the_gui_boundary.py::test_a_burst_of_scroll_ticks_is_one_move_of_the_last_ticks_step',
        'tests/test_the_gui_displays_the_turret.py::stand',
        'tests/test_the_protocol_panels_schedule_is_the_protocols.py::ctx',
        'tests/test_the_report_passes_nothing_it_could_not_measure.py::test_a_fullinfo_that_timed_out_is_not_a_serial_number',
        'tests/test_the_run_turns_the_turret_itself.py::test_a_refused_move_raises_instead_of_being_skipped',
        'tests/test_the_z_display_follows_the_api.py::control',
        'tests/test_the_zstack_button_runs_through_run_zstack.py::clicked',
        'tests/test_tsr_cluster_fix.py::TestMotorAndLedConnectionProbes.test_a_connected_motor_board_via_live_property',
        'tests/test_tsr_cluster_fix.py::TestMotorAndLedConnectionProbes.test_a_missing_motor_board_via_live_property',
        'tests/test_tsr_cluster_fix.py::TestMotorAndLedConnectionProbes.test_led_ok_uses_post_wave7_illumination_or_live',
        'tests/test_tsr_cluster_fix.py::TestMotorAndLedConnectionProbes.test_no_motor_board_on_a_model_built_without_one',
        'tests/test_tsr_cluster_fix.py::TestMotorAndLedConnectionProbes.test_target_str_resolves_illumination_not_legacy_led',
        'tests/test_turret_position_disambiguation.py::_make_scope_with_turret',
        'tests/test_two_diagnostics_runs_never_share_a_file.py::_scope_with',
        'tests/test_video_camera_lost_outcome.py::_make_recorder',
        'tests/test_video_timestamp_overlay.py::_video_step',
        'tests/test_video_writer.py::TestProtocolVideoDropNotification._run_one_frame_step',
        'tests/test_z_box_write_guard.py::control',
    }
)


def test_every_scope_shaped_stand_in_is_allowlisted_or_by_design():
    unlisted = sorted(
        f'{entry}:{line} {what}'
        for entry, line, what, marked in _found_in_tests()
        if not marked and entry not in ALLOWED
    )
    assert unlisted == [], (
        'a scope, camera, board, session or sub-API in a test is the simulator '
        '(sim_scope, ScopeSession.create(simulate=True)); a stand-in that is right '
        f'by design is marked at its site, "{MARK} <why>": {unlisted}'
    )


def test_every_allowlist_entry_still_names_a_site():
    stale = sorted(ALLOWED - _entries_with_a_site())
    assert stale == [], f'delete these allowlist entries; their sites are gone: {stale}'


def test_the_allowlist_only_shrinks():
    assert len(ALLOWED) == ALLOWED_ENTRIES, (
        f'{len(ALLOWED)} entries against the pinned {ALLOWED_ENTRIES}: a commit that removes '
        'entries lowers the pin with them; a new test uses the simulator or marks its '
        'stand-in by design'
    )


def _forms(source: str) -> list[str]:
    return [what for _q, _l, what, marked in sites(source) if not marked]


def test_the_guard_sees_every_form_it_refuses():
    refused = [
        'scope = MagicMock()',
        'self.scope = Mock()',
        'fake_scope: Lumascope = MagicMock(spec=Lumascope)',
        'motor_board = MagicMock()',
        'runner = Runner(session=MagicMock())',
        'ctx = Ctx(imaging=MagicMock(), led_connected=True)',
        'scope = spec_scope(camera_connected=True)',
        'double = create_autospec(real, instance=True)',
        'scope = SimpleNamespace(motion=motion, capabilities=SimpleNamespace(has_xy_stage=True))',
        'scope = types.SimpleNamespace(objective_helper=ObjectiveLoader())',
        'scope = Lumascope.__new__(Lumascope)',
        'imaging = ImagingAPI.__new__(ImagingAPI)',
        'board = motorboard_mod.MotorBoard.__new__(motorboard_mod.MotorBoard)',
        'parent = PylonCamera.__new__(PylonCamera)',
    ]
    assert [len(_forms(source)) for source in refused] == [1] * len(refused)


def test_a_mark_on_the_statement_or_the_line_above_exempts_every_site_in_it():
    marked = [
        'scope = MagicMock()  # a stand-in by design: the subject is the double',
        '# a stand-in by design: the lane is the subject\n'
        'scope = SimpleNamespace(\n    illumination=MagicMock(),\n    imaging=MagicMock(),\n)',
    ]
    assert [_forms(source) for source in marked] == [[], []]
    assert [sum(m for *_rest, m in sites(source)) for source in marked] == [1, 3]


def test_what_is_not_a_stand_in_for_a_scope_passes():
    allowed = [
        'log = MagicMock()',
        'microscope_settings = MagicMock()',
        'stop_motion = MagicMock()',
        'update_camera_config = MagicMock()',
        "box = SimpleNamespace(text='12')",
        'report = TechSupportReport.__new__(TechSupportReport)',
        'panel = MicroscopeSettings.__new__(MicroscopeSettings)',
        'widget = ScopeDisplay.__new__(ScopeDisplay)',
        'scope = sim_scope',
        'motion = sim_scope.motion',
    ]
    assert [_forms(source) for source in allowed] == [[]] * len(allowed)


# Announced at the end of every run (tests/ratchets.py).
ratchets.register(
    'tests: scope-shaped stand-in sites allowlisted',
    lambda: len(ALLOWED),
    ALLOWED_ENTRIES,
    rule='equal',
)
ratchets.register('tests: stand-ins by design', lambda: len(_by_design()), 0, rule='announce')
