# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The ratchet: a control added to the GUI cannot silently log nothing, and a
control cannot start writing another control's name.

Commit 3 of the gap-fill (``GUI_LOGGING_GAP_FILL_PLAN_2026-09-08.md``, D1).
Commits 1 and 2 filled thirteen silent controls and ten misattributed ones;
this is what stops the next one.

**The predicate is NAMING, not presence.** A census asking only "does this
control reach an emitter?" is exactly what passed all ten controls commit 2
had to fix -- each reached an emitter, and what came out named something else.
So the pin below is per-control: this control reaches THESE record names. A
handler rewired to a neighbour's emitter changes the pairing and fails here,
even though the unlogged set stays empty.

**Why the count could not be pinned before now.** Every census behind commits 1
and 2 read ``ui/lumaviewpro.kv`` alone, and Kivy also takes rules from
``Builder.load_string`` blocks in ``.py`` modules. D8 recorded one confirmed gap
in that blind spot. Sweeping it (this commit's first task) found the blind spot
held BOTH remaining unlogged controls and that the kv itself is clean -- which
is what makes a pinned number defensible at last.
``tests/gui_logging_census.py`` enumerates both sources so it cannot shrink back
to one file.

**What this does not assert.** A derived name (a layer's channel suffix, a jog
button's direction) is pinned by the HELPER that computes it, not by the string:
the value needs a running widget, and the suite mocks Kivy. Commit 2's pass
verified those derivations against the real parse tree; this guards the wiring
that reaches them.

Updating the pins is meant to be deliberate. A failure here is the question
"what record does this new control write, and does it name itself?" -- answer it
before editing the roster.
"""

from __future__ import annotations

from tests.gui_logging_census import LOAD_STRING_MODULES, census
from tests.ast_seams import REPO_ROOT


def _D(helper):
    """A record whose NAME this helper computes from the control's own
    identity -- a layer's channel suffix, a jog button's direction. The
    string needs a running widget, so what is pinned is the helper that
    builds it."""
    return f'via:{helper}'


# Controls that reach no gui_logger record at all. Each needs a reason; a new
# entry is a decision, not a formality.
_UNLOGGED = {
    # D8. Typing an acceleration limit records nothing while the twin slider
    # emits SLIDER ACCELERATION -- so the log shows a limit changing with no
    # line saying a user set it. Fix is commit 4 (Eric 2026-09-11: gaps the
    # sweep finds are filled in their own commit, not folded in here).
    'AdvancedSettings.acceleration_pct_text': 'gap -- fill in commit 4',
    # Found by this commit's sweep, same class as D8 and invisible to every
    # census before it. Cancelling a stalled support-report / zip-logs /
    # stitching run leaves no record, although each of those runs logs its
    # START (GENERATE_SUPPORT_REPORT, ZIP_LOGS, RUN_STITCHER). A bundle can
    # therefore show an operation beginning and never finishing with nothing
    # saying the user stopped it. Fix is commit 4.
    'CustomPopup.Button(cancel)': 'gap -- fill in commit 4',
    # NOT a gap: dismissing a modal is not a control action. No close or
    # dismiss button anywhere in the app is censused, and the notification
    # popup's equivalent is carried by gui_logger.popup_response instead.
    'AdvancedSettings.Button(close)': 'excluded -- modal dismiss is not an action',
}
_ROSTER = {
    'AdvancedSettings.acceleration_pct_slider': ('ACCELERATION',),
    'AdvancedSettings.high_conversion_gain': ('HIGH_CONVERSION_GAIN',),
    'AdvancedSettings.keep_led_between_steps_btn': ('KEEP_LED_BETWEEN_STEPS',),
    'AdvancedSettings.line_noise_reduction': ('LINE_NOISE_REDUCTION',),
    'AdvancedSettings.live_view_fps_slider': ('FPS',),
    'AdvancedSettings.protocol_led_on_btn': ('PROTOCOL_LED_ON',),
    'AdvancedSettings.scope_spinner': ('SCOPE',),
    'AdvancedSettings.separate_folder_per_channel_id': ('SEPARATE_FOLDERS',),
    'AdvancedSettings.show_step_locations_id': ('SHOW_STEP_LOCATIONS',),
    'AdvancedSettings.stimulation_settings_btn': ('STIMULATION_ENABLED',),
    'AdvancedSettings.tiling_overlap_spinner': ('TILING_OVERLAP',),
    'AdvancedSettings.video_max_duration_input': (
        'VIDEO_MAX_DURATION_S',
        'VIDEO_MAX_DURATION_S_APPLIED',
    ),
    'AdvancedSettings.video_max_fps_input': ('VIDEO_MAX_FPS', 'VIDEO_MAX_FPS_APPLIED'),
    'AdvancedSettings.video_timestamp_overlay_id': ('VIDEO_TIMESTAMP_OVERLAY',),
    'CellCountControls.FileChooseBTN(choose:load_cell_count_input_image)': (
        'FILE_CHOOSE',
        'FILE_CHOOSE_OPEN',
    ),
    'CellCountControls.FileChooseBTN(choose:load_cell_count_method)': (
        'FILE_CHOOSE',
        'FILE_CHOOSE_OPEN',
    ),
    'CellCountControls.FileSaveBTN(choose:saveas_cell_count_method)': (
        'FILE_SAVE',
        'FILE_SAVE_OPEN',
    ),
    'CellCountControls.FolderChooseBTN(choose:apply_cell_count_method_to_folder)': (
        'FOLDER_CHOOSE',
        'FOLDER_CHOOSE_OPEN',
    ),
    'CellCountControls.RoundedButton(apply_method_to_preview_image)': ('APPLY_METHOD_TO_PREVIEW',),
    'CellCountControls.cell_count_fluorescent_mode_id': ('CELL_COUNT_FLUORESCENT_MODE',),
    'CellCountControls.slider_cell_count_threshold_id': ('CELL_COUNT_THRESHOLD',),
    'CellCountControls.text_cell_count_pixels_per_um_id': (
        'CELL_COUNT_AREA_RANGE',
        'CELL_COUNT_PERIMETER_RANGE',
        'CELL_COUNT_PIXELS_PER_UM',
    ),
    'CompositeGenControls.FolderChooseBTN(choose:apply_composite_gen_to_folder)': (
        'FOLDER_CHOOSE',
        'FOLDER_CHOOSE_OPEN',
    ),
    'GraphingControls.FileChooseBTN(choose:load_graphing_data)': (
        'FILE_CHOOSE',
        'FILE_CHOOSE_OPEN',
    ),
    'GraphingControls.FileSaveBTN(choose:save_graph)': (
        'FILE_SAVE',
        'FILE_SAVE_OPEN',
    ),
    'GraphingControls.graph_title_input': (_D('log_text_commit'),),
    'GraphingControls.graphing_x_axis_spinner': (
        'GRAPHING_X_AXIS',
        'GRAPHING_Y_AXIS',
        'TRENDLINE',
    ),
    'GraphingControls.graphing_y_axis_spinner': (
        'GRAPHING_X_AXIS',
        'GRAPHING_Y_AXIS',
        'TRENDLINE',
    ),
    'GraphingControls.trendline_spinner': (
        'GRAPHING_X_AXIS',
        'GRAPHING_Y_AXIS',
        'TRENDLINE',
    ),
    'GraphingControls.x_axis_label_input': (_D('log_text_commit'),),
    'GraphingControls.y_axis_label_input': (_D('log_text_commit'),),
    'ImageSettings.toggle_imagesettings': ('IMAGE_SETTINGS_PANEL',),
    'LayerControl.RoundedButton(apply_focus_to_channel_steps)': (
        _D('apply_focus_to_channel_steps'),
    ),
    'LayerControl.RoundedButton(goto_focus)': (_D('goto_focus'),),
    'LayerControl.RoundedButton(save_focus)': (_D('save_focus'),),
    'LayerControl.acquire_image': (_D('update_acquire'),),
    'LayerControl.acquire_none': (_D('update_acquire'),),
    'LayerControl.acquire_video': (_D('update_acquire'),),
    'LayerControl.auto_gain': (_D('update_auto_gain'),),
    'LayerControl.autofocus': (_D('update_autofocus'),),
    'LayerControl.composite_threshold_slider': (_D('composite_threshold_slider'),),
    'LayerControl.composite_threshold_text': (_D('composite_threshold_text'),),
    'LayerControl.enable_led_btn': (_D('update_led_state'),),
    'LayerControl.exp_slider': (_D('exp_slider'),),
    'LayerControl.exp_text': (_D('exp_text'),),
    'LayerControl.false_color': (_D('false_color'),),
    'LayerControl.gain_slider': (_D('gain_slider'),),
    'LayerControl.gain_text': (_D('gain_text'),),
    'LayerControl.ill_slider': (_D('ill_slider'),),
    'LayerControl.ill_text': (_D('ill_text'),),
    'LayerControl.logHistogram_id': (_D('log_histogram_scale'),),
    'LayerControl.stim_disable_btn': (_D('update_stim_enable'),),
    'LayerControl.stim_enable_btn': (_D('update_stim_enable'),),
    'LayerControl.stim_freq_slider': (_D('stim_freq_slider'),),
    'LayerControl.stim_freq_text': (_D('stim_freq_text'),),
    'LayerControl.stim_ill_slider': (_D('stim_ill_slider'),),
    'LayerControl.stim_ill_text': (_D('stim_ill_text'),),
    'LayerControl.stim_pulse_count_slider': (_D('stim_pulse_count_slider'),),
    'LayerControl.stim_pulse_count_text': (_D('stim_pulse_count_text'),),
    'LayerControl.stim_pulse_width_slider': (_D('stim_pulse_width_slider'),),
    'LayerControl.stim_pulse_width_text': (_D('stim_pulse_width_text'),),
    'LayerControl.sum_slider': (_D('sum_slider'),),
    'LayerControl.sum_text': (_D('sum_text'),),
    'LayerControl.video_duration_slider': (_D('video_duration_slider'),),
    'LayerControl.video_duration_text': (_D('video_duration_text'),),
    'MainDisplay.capture_btn': ('LIVE_CAPTURE',),
    'MainDisplay.composite_btn': ('COMPOSITE_CAPTURE',),
    'MainDisplay.fit_btn': ('FIT_IMAGE',),
    'MainDisplay.live_btn': (
        'CAM_PLAY',
        'CAM_TOGGLE',
    ),
    'MainDisplay.live_folder_btn': (
        'FOLDER_CHOOSE',
        'FOLDER_CHOOSE_OPEN',
    ),
    'MainDisplay.one2one_btn': ('ONE_TO_ONE_IMAGE',),
    'MainDisplay.open_last_save_folder': ('OPEN_SAVE_FOLDER',),
    'MainDisplay.record_btn': ('RECORD',),
    'MicroscopeSettings.binning_spinner': (
        'BINNING',
        _D('select_binning_size'),
    ),
    'MicroscopeSettings.btn_advanced_settings': ('OPEN_ADVANCED_SETTINGS',),
    'MicroscopeSettings.btn_support_report': ('GENERATE_SUPPORT_REPORT',),
    'MicroscopeSettings.btn_zip_logs': ('ZIP_LOGS',),
    'MicroscopeSettings.enable_bullseye_btn_id': ('BULLSEYE',),
    'MicroscopeSettings.enable_crosshairs_btn': ('CROSSHAIRS',),
    'MicroscopeSettings.enable_live_image_histogram_equalization_btn': (
        'LIVE_HISTOGRAM_EQUALIZATION',
    ),
    'MicroscopeSettings.enable_scale_bar_btn': ('SCALE_BAR',),
    'MicroscopeSettings.frame_height_id': (_D('frame_size'),),
    'MicroscopeSettings.frame_width_id': (_D('frame_size'),),
    'MicroscopeSettings.image_mode_spinner': ('IMAGE_MODE',),
    'MicroscopeSettings.jpg_quality_slider': ('JPG_QUALITY',),
    'MicroscopeSettings.live_image_output_format_spinner': ('LIVE_IMAGE_OUTPUT_FORMAT',),
    'MicroscopeSettings.sequenced_image_output_format_spinner': ('SEQUENCED_IMAGE_OUTPUT_FORMAT',),
    'MicroscopeSettings.show_tooltips_btn': ('SHOW_TOOLTIPS',),
    'MicroscopeSettings.video_recording_format_spinner': ('VIDEO_RECORDING_FORMAT',),
    'MotionSettings.toggle_motionsettings': ('MOTION_SETTINGS_PANEL',),
    'PostProcessingAccordion.RoundedButton(open_cell_count)': ('OPEN_CELL_COUNT',),
    'PostProcessingAccordion.RoundedButton(open_graphing)': ('OPEN_GRAPHING',),
    'ProtocolSettings.FileChooseBTN(choose:load_protocol)': (
        'FILE_CHOOSE',
        'FILE_CHOOSE_OPEN',
    ),
    'ProtocolSettings.FileSaveBTN(choose:saveas_protocol)': (
        'FILE_SAVE',
        'FILE_SAVE_OPEN',
    ),
    'ProtocolSettings.RoundedButton(new_protocol)': ('NEW',),
    'ProtocolSettings.RoundedButton(save_protocol)': ('SAVE',),
    'ProtocolSettings.acquire_zstack_id': ('ACQUIRE_ZSTACK',),
    'ProtocolSettings.add_step_btn': ('INSERT_STEP',),
    'ProtocolSettings.bf_af_for_fluorescence_btn': ('BF_AF_FOR_FLUORESCENCE',),
    'ProtocolSettings.capture_dur': ('PROTOCOL_DURATION',),
    'ProtocolSettings.capture_period': ('PROTOCOL_PERIOD',),
    'ProtocolSettings.capture_root': ('CAPTURE_ROOT', 'CAPTURE_ROOT_APPLIED'),
    'ProtocolSettings.change_step_btn': ('MODIFY_STEP',),
    'ProtocolSettings.delete_step_btn': ('DELETE_STEP',),
    'ProtocolSettings.labware_spinner': ('LABWARE',),
    'ProtocolSettings.next_step_btn': ('NEXT_STEP',),
    'ProtocolSettings.prev_step_btn': ('PREV_STEP',),
    'ProtocolSettings.protocol_disable_image_saving_id': ('PROTOCOL_DISABLE_IMAGE_SAVING',),
    'ProtocolSettings.protocol_zstacking_apply_id': ('APPLY_ZSTACKING',),
    'ProtocolSettings.run_autofocus_btn': ('AF_SCAN_START',),
    'ProtocolSettings.run_protocol_btn': (
        'ABORT_PROTOCOL',
        'RUN',
    ),
    'ProtocolSettings.run_scan_btn': (
        'ABORT_SCAN',
        'SCAN',
    ),
    'ProtocolSettings.step_name_input': ('RENAME_STEP',),
    'ProtocolSettings.step_number_input': (
        'STEP_NUMBER',
        'STEP_NUMBER_APPLIED',
    ),
    'ProtocolSettings.tiling_size_apply_id': ('APPLY_TILING',),
    'ProtocolSettings.tiling_size_spinner': ('TILING',),
    'QuickEnhanceControls.FileOrFolderChooseBTN(choose:choose_quick_enhance_target)': (
        'FILE_OR_FOLDER_CHOOSE',
        'FILE_OR_FOLDER_CHOOSE_OPEN',
    ),
    'VerticalControl.RoundedButton(goto_bookmark)': ('GOTO_Z_BOOKMARK',),
    'VerticalControl.RoundedButton(set_all_bookmarks)': ('SET_ALL_BOOKMARKS',),
    'VerticalControl.RoundedButton(set_bookmark)': ('SET_Z_BOOKMARK',),
    'VerticalControl.autofocus_id': ('AUTOFOCUS',),
    'VerticalControl.fast_down': (_D('coarse_down'),),
    'VerticalControl.fast_up': (_D('coarse_up'),),
    'VerticalControl.home_id': ('HOME_Z',),
    'VerticalControl.obj_position': ('Z_POSITION',),
    'VerticalControl.objective_spinner2': ('OBJECTIVE',),
    'VerticalControl.reset_turret_objective_btn': (
        'OBJECTIVE',
        'RESET_TURRET_OBJECTIVE',
        'TURRET_OBJECTIVE',
    ),
    'VerticalControl.set_turret_objective_btn': ('TURRET_OBJECTIVE',),
    'VerticalControl.slow_down': (_D('fine_down'),),
    'VerticalControl.slow_up': (_D('fine_up'),),
    'VerticalControl.turret_pos_1_btn': (
        'OBJECTIVE',
        'TURRET_OBJECTIVE',
        _D('turret_select'),
    ),
    'VerticalControl.turret_pos_2_btn': (
        'OBJECTIVE',
        'TURRET_OBJECTIVE',
        _D('turret_select'),
    ),
    'VerticalControl.turret_pos_3_btn': (
        'OBJECTIVE',
        'TURRET_OBJECTIVE',
        _D('turret_select'),
    ),
    'VerticalControl.turret_pos_4_btn': (
        'OBJECTIVE',
        'TURRET_OBJECTIVE',
        _D('turret_select'),
    ),
    'VerticalControl.z_position_id': ('Z_POSITION',),
    'VideoCreationControls.FolderChooseBTN(choose:apply_video_gen_to_folder)': (
        'FOLDER_CHOOSE',
        'FOLDER_CHOOSE_OPEN',
    ),
    'VideoCreationControls.enable_timestamp_overlay_btn': ('VIDEO_TIMESTAMP_OVERLAY_BTN',),
    'VideoCreationControls.video_gen_fps_id': ('VIDEO_GEN_FPS',),
    'XYStageControl.Button:(coarse_fwd)': (_D('coarse_fwd'),),
    'XYStageControl.Button:(coarse_left)': (_D('coarse_left'),),
    'XYStageControl.Button:(fine_back)': (_D('fine_back'),),
    'XYStageControl.Button:(fine_fwd)': (_D('fine_fwd'),),
    'XYStageControl.RoundedButton(goto_xbookmark)': ('GOTO_X_BOOKMARK',),
    'XYStageControl.RoundedButton(goto_ybookmark)': ('GOTO_Y_BOOKMARK',),
    'XYStageControl.RoundedButton(set_xbookmark)': ('SET_X_BOOKMARK',),
    'XYStageControl.RoundedButton(set_ybookmark)': ('SET_Y_BOOKMARK',),
    'XYStageControl.fast_down': (
        _D('coarse_back'),
        _D('coarse_right'),
    ),
    'XYStageControl.home_id': ('HOME_XY',),
    'XYStageControl.slow_down': (_D('fine_right'),),
    'XYStageControl.slow_up': (_D('fine_left'),),
    'XYStageControl.x_pos_id': ('SET_X_POSITION',),
    'XYStageControl.y_pos_id': ('SET_Y_POSITION',),
    'ZProjectionControls.FolderChooseBTN(choose:apply_zprojection_to_folder)': (
        'FOLDER_CHOOSE',
        'FOLDER_CHOOSE_OPEN',
    ),
    'ZProjectionControls.zprojection_method_spinner': ('ZPROJECTION_METHOD',),
    'ZStack.zstack_aqr_btn': ('ZSTACK',),
    'ZStack.zstack_range_id': (
        _D('log_step_field'),
        _D('set_steps'),
    ),
    'ZStack.zstack_spinner': ('ZSTACK_REFERENCE_POSITION',),
    'ZStack.zstack_stepsize_id': (
        _D('log_step_field'),
        _D('set_steps'),
    ),
}


def _observed(found):
    """What a control reaches: literal record names, plus ``via:<helper>`` for
    each helper that computes a name from the control's own identity."""
    return set(found['static']) | {f'via:{d}' for d in found['derived']}


def test_every_kv_handler_resolves_to_a_method():
    """A binding naming a method no class defines is silently dead.

    kv resolves ``root.foo()`` at dispatch time, so a handler on the wrong
    class leaves the binding present and the control mute with no import
    error and no failing test anywhere else.
    """
    surface = census()
    dead = {k: v['unresolved'] for k, v in surface.items() if v['unresolved']}
    assert not dead, (
        f'{len(dead)} control(s) bind a handler that no class in the MRO '
        f'defines, so operating them does nothing: {dead}'
    )


def test_the_unlogged_controls_are_exactly_the_pinned_set():
    """The ratchet. A new control that logs nothing lands here and fails."""
    surface = census()
    unlogged = {k for k, v in surface.items() if not v['static'] and not v['derived']}
    new = unlogged - set(_UNLOGGED)
    assert not new, (
        f'{len(new)} control(s) reach no gui_logger record: {sorted(new)}. '
        'Operating one leaves nothing in gui_interactions.log naming it. '
        'Add the emitter, or add it to _UNLOGGED with a reason.'
    )
    fixed = set(_UNLOGGED) - unlogged
    assert not fixed, (
        f'{sorted(fixed)} now log and can come out of _UNLOGGED -- the list '
        'must not carry entries that are no longer true.'
    )


def test_every_control_still_writes_its_own_record_name():
    """The naming predicate: this control reaches THESE names, not a neighbour's.

    Presence alone is what let ten misattributed controls through commit 1's
    census. Pinning the pairing is what catches the eleventh.
    """
    surface = census()
    logged = {k: v for k, v in surface.items() if k not in _UNLOGGED}

    unpinned = sorted(set(logged) - set(_ROSTER))
    assert not unpinned, (
        f'{len(unpinned)} control(s) are not in the roster: {unpinned}. '
        'Name the record each one writes and pin it.'
    )

    gone = sorted(set(_ROSTER) - set(logged))
    assert not gone, (
        f'{len(gone)} rostered control(s) no longer bind a user event: {gone}. '
        'Remove them from the roster if the control is gone.'
    )

    changed = {
        k: {'pinned': sorted(set(_ROSTER[k])), 'now': sorted(_observed(v))}
        for k, v in logged.items()
        if set(_ROSTER[k]) != _observed(v)
    }
    assert not changed, (
        f'{len(changed)} control(s) changed the record they write: {changed}. '
        'A control writing a name that is not its own is the defect commit 2 '
        'existed to fix.'
    )


def test_no_new_kv_source_escapes_the_census():
    """Keeps the blind spot closed.

    Every census before this commit read the kv file alone, and both remaining
    unlogged controls were sitting in ``Builder.load_string`` blocks it could
    not see. A new block added without registering it would reopen exactly that
    hole, and nothing else in the suite would notice.
    """
    declared = set(LOAD_STRING_MODULES)
    actual = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in sorted(REPO_ROOT.glob('ui/*.py'))
        if 'Builder.load_string' in path.read_text()
    }
    unregistered = sorted(actual - declared)
    assert not unregistered, (
        f'{unregistered} define kv rules via Builder.load_string but are not '
        'in LOAD_STRING_MODULES, so their controls are invisible to the '
        'census. Add them.'
    )
    stale = sorted(declared - actual)
    assert not stale, f'{stale} no longer hold a load_string block.'
