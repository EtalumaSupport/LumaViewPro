# Copyright Etaluma, Inc.
import datetime
import logging
import os
import threading

from kivy.clock import Clock
from kivy.properties import BooleanProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout

import modules.app_context as _app_ctx
import modules.binning as binning
import modules.common_utils as common_utils
import modules.config_ui_getters as config_ui_getters
from modules import gui_logger, path_utils
from modules.config_helpers import (
    camera_max_exposure_for_ui,
    camera_max_gain_for_ui,
)
from modules.config_ui_getters import (
    firmware_stim_supported,
)
from modules.memory_profiler import MemoryLeakProfiler
from modules.notification_center import notifications
import modules.image_mode as image_mode
from modules.zstack_config import ZStackConfig
from ui.ui_helpers import run_reported, submit_reported

logger = logging.getLogger('LVP.ui.microscope_settings')


# Which frame box committed -> the record it owns and the stored dimension
# that corrects it. Both boxes bind one handler, so the handler is told which
# one the user left; the record name and the axis travel together because a
# correction has to report the dimension the record is about.
_FRAME_BOXES = {
    'frame_width_id': ('FRAME_WIDTH', 'width'),
    'frame_height_id': ('FRAME_HEIGHT', 'height'),
}


class _CoalescingApplier:
    """One-at-a-time worker that keeps only the LATEST pending value.

    Used by MicroscopeSettings.frame_size so rapid frame edits do not stack
    up slow resizes on the camera lane (issue #624). On large frames each
    Pylon stop_grabbing/start_grabbing cycle blocks the camera worker for
    ~11s; naive queueing of rapid user edits (committing width, then
    height, while the first apply still runs) produced multi-minute
    backlogs that made the UI feel frozen.

    Pattern:
      - submit(value) stashes value in a single pending slot and returns
        True only when the caller should enqueue the worker task (no task
        already in flight).
      - apply_pending(fn) drains the pending slot and calls fn(value) for
        each value. Loops until pending is empty so late-arriving updates
        during an apply() are picked up in the SAME task rather than
        spawning a new one.

    Whether a value is already in force is not decided here: the apply
    itself skips a size the camera already delivers, against the camera's
    own record. A record kept here went stale whenever something else
    framed the camera (a binning change applies its own frame) and then
    swallowed a real edit as a repeat.
    """

    def __init__(self, name='coalescing_applier'):
        self._name = name
        self._pending = None
        self._in_flight = False
        self._lock = threading.Lock()

    def submit(self, value):
        with self._lock:
            self._pending = value
            if self._in_flight:
                return False
            self._in_flight = True
            return True

    def apply_pending(self, fn):
        """Apply each pending value in turn; raise the first failure.

        A failure does not stop the drain: an edit that arrived while a
        refused one was applying is still the person's latest request. The
        first exception is raised once the slot is empty, so the caller
        reports it; the gate is open again by then. Each later failure is
        logged through the reporter and not shown: one popup per drain.
        """
        failure = None
        while True:
            with self._lock:
                val = self._pending
                self._pending = None
                if val is None:
                    self._in_flight = False
                    break
            try:
                fn(val)
            except Exception as e:
                if failure is None:
                    failure = e
                else:
                    notifications.report_outcome(
                        e, solicited=True, category=f'UI:{self._name}', log_only=True
                    )
        if failure is not None:
            raise failure


class MicroscopeSettings(BoxLayout):
    # The model the scope runs as, and a model saved for the next start,
    # shown read-only in the panel. The selector lives in Advanced Settings;
    # both are the API's answers, set in show_scope_model.
    current_scope_model = StringProperty('')
    next_start_model = StringProperty('')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.debug('[LVP Main  ] MicroscopeSettings.__init__()')
        # Coalesce rapid set_frame_size requests. See
        # _CoalescingApplier + issue #624.
        self._frame_size_applier = _CoalescingApplier(name='FRAME_SIZE')

        # try:
        #     os.chdir(source_path)
        #     with open('./data/objectives.json', "r") as read_file:
        #         self.objectives = json.load(read_file)
        # except Exception:
        #     logger.exception('[LVP Main  ] Unable to open objectives.json.')
        #     raise

    # def get_objective_info(self, objective_id: str) -> dict:
    #     return self.objectives[objective_id]

    # Fill the panel from the settings store
    def load_settings(self):
        logger.info('[LVP Main  ] MicroscopeSettings.load_settings()')
        ctx = _app_ctx.ctx

        lumaview = ctx.lumaview
        settings = ctx.settings

        # Settings are imported at the very beginning of file

        if settings['profiling']['enabled']:
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')  # noqa: F841 -- deferred
            # Joined to the data directory, never CWD-relative: an
            # installed build cannot write beside its executable.
            profiling_save_path = os.path.join(ctx.source_path, 'logs/profiling')
            MemoryLeakProfiler.start(root_log_dir=profiling_save_path)
            logger.info('[LVP Main  ] Memory Profiler started.')

        # Handle / object-type leak diagnostic. Same opt-in pattern as
        # the memory profiler above; settings-driven so customers and
        # bench operators can enable without rebuilding.
        if settings['profiling']['handle_trace_enabled']:
            from lib import handle_trace as _handle_trace

            _handle_trace.enable(
                obj_sample_every=int(settings['profiling']['handle_trace_obj_sample_every'])
            )

        # update GUI values from JSON data:

        # The Session adopted the model the hardware reports at
        # bring-up; render it (control visibility + read-only model
        # label + stage redraw, in that order).
        self.reconfigure_for_scope()

        # Image mode selector: every mode on every camera, showing the stored
        # one, which bring-up ran as saved.
        # Setting the spinner text fires select_image_mode (on_text), which
        # caches the mode and applies the pixel format.
        self.load_image_modes()
        mode = image_mode.resolve_settings_image_mode(settings)
        self.ids['image_mode_spinner'].text = image_mode.IMAGE_MODE_LABELS[mode]

        self.ids['live_image_output_format_spinner'].text = settings['image_output_format']['live']
        # JPG quality slider reflects the saved preference; its row shows
        # and enables with the live format declaratively in the kv.
        jpg_quality = int(settings['jpg_quality'])
        self.ids['jpg_quality_slider'].value = jpg_quality
        self.ids['jpg_quality_value_label'].text = str(jpg_quality)

        self.ids['sequenced_image_output_format_spinner'].text = settings['image_output_format'][
            'sequenced'
        ]

        # The exposure/gain slider caps from the live camera (the resolver
        # applies the documented no-camera fallback; #616). The gain cap
        # keeps the slider honest per-camera -- a universal 48 dB let LS620
        # users overdrive past the usable range and black out the image.
        max_exposure = camera_max_exposure_for_ui(lumaview.scope.imaging)
        ctx.max_exposure = max_exposure
        max_gain = camera_max_gain_for_ui(lumaview.scope.imaging)
        ctx.max_gain = max_gain

        if not settings['video_as_frames']:
            self.ids['video_recording_format_spinner'].text = 'mp4'
        else:
            self.ids['video_recording_format_spinner'].text = 'Frames'

        self.select_video_recording_format()

        ctx.live_view_fps = settings['live_view_fps']

        fps_label = 'Max (uncapped)' if ctx.live_view_fps == 0 else str(ctx.live_view_fps)
        logger.info(f'[LVP Main  ] Live view FPS set to {fps_label}')

        # Set Frame Size UI
        binning_size_str = settings['binning']['size']

        # settings['frame'] holds the DISPLAYED (post-binning) size, and the
        # box shows that size unscaled -- the unbinned ROI is carried
        # separately as frame['native_width'/'native_height']. The framing
        # redraw agrees: it writes the box from the same stored number, which
        # the Session stores from what the camera delivered. Multiplying by the binning
        # factor here contradicted all of that and would show a 2x2 user twice
        # the size the camera delivers.
        self._write_frame_text(settings['frame']['width'], settings['frame']['height'])

        # Pixel Binning -- UI recalculation only, scope.imaging.set_binning_size()
        # was applied by the Session's bring-up
        # The twin of the labware restore below: writing the spinner
        # dispatches its text event, and the explicit call emits again, so
        # cold start recorded two binning selections nobody made. Declared
        # twice because only one declaration is pending per name -- whichever
        # emission happens consumes one, and the second replaces any the
        # first left unconsumed.
        gui_logger.note_write_back('BINNING', binning_size_str)
        self.ids['binning_spinner'].text = binning_size_str
        gui_logger.note_write_back('BINNING', binning_size_str)
        self.select_binning_size()

        # The settings-to-scope bring-up ran in the Session before this
        # widget existed: the labware selected, scope.initialize()
        # applied. The turret, its assignments and the objective shown
        # are the API's answers -- on a turreted scope the objective is
        # unknown until the turret is in a known slot. The startup
        # sequence owns the objective question, so this display asks
        # nothing.
        ctx.motion_settings.ids['verticalcontrol_id'].show_turret_state(prompt=False)

        if settings['scale_bar']['enabled']:
            self.ids['enable_scale_bar_btn'].state = 'down'
        else:
            self.ids['enable_scale_bar_btn'].state = 'normal'

        protocol_settings = ctx.motion_settings.ids['protocol_settings_id']
        # Restoring the stored labware dispatches the spinner's event, and
        # the explicit call below emits again -- neither is a user pick.
        # Declared twice because only one declaration is pending per name:
        # whichever of the two emissions happens consumes one, and the
        # second declaration replaces any the first left unconsumed.
        gui_logger.note_write_back('LABWARE', settings['protocol']['labware'])
        protocol_settings.ids['labware_spinner'].text = settings['protocol']['labware']
        gui_logger.note_write_back('LABWARE', settings['protocol']['labware'])
        protocol_settings.select_labware()
        # Apply the persisted step-location view at startup; the toggle
        # that edits this now lives in Advanced Settings.
        ctx.stage.show_protocol_steps(enable=settings['show_step_locations'])

        zstack_settings = ctx.motion_settings.ids['verticalcontrol_id'].ids['zstack_id']
        # Restoring the stored position dispatches the spinner's event, which
        # would read as the user choosing it during startup.
        gui_logger.note_write_back('ZSTACK_REFERENCE_POSITION', settings['zstack']['position'])
        zstack_settings.ids['zstack_spinner'].text = settings['zstack']['position']
        zstack_settings.ids['zstack_stepsize_id'].text = str(settings['zstack']['step_size'])
        zstack_settings.ids['zstack_range_id'].text = str(settings['zstack']['range'])

        z_reference = common_utils.convert_zstack_reference_position_setting_to_config(
            text_label=settings['zstack']['position']
        )

        zstack_config = ZStackConfig(
            range=settings['zstack']['range'],
            step_size=settings['zstack']['step_size'],
            current_z_reference=z_reference,
            current_z_value=None,
        )

        zstack_settings.ids['zstack_steps_id'].text = str(zstack_config.number_of_steps())

        if settings['show_tooltips']:
            self.ids['show_tooltips_btn'].state = 'down'
            ctx.show_tooltips = True
        else:
            self.ids['show_tooltips_btn'].state = 'normal'
            ctx.show_tooltips = False

        # Stimulation is firmware-gated. The enable toggle lives in
        # Advanced Settings now; startup just establishes the setting and
        # pushes the persisted state down to every layer via the single
        # owner (which forces it off on unsupported firmware).
        self.apply_stimulation_support()

        for layer in common_utils.get_layers():
            layer_obj = ctx.image_settings.layer_lookup(layer=layer)

            # Size the sliders to the camera caps BEFORE the values land
            # (the Kivy slider clamps the displayed value to its max). A
            # stored value above the cap stays in the store and is pinned
            # on the slider; the box keeps the real number.
            layer_obj.ids['gain_slider'].max = max_gain
            layer_obj.ids['exp_slider'].max = max_exposure

        # Render and re-apply any layer the camera cannot fully reach --
        # the single owner, shared with the capability resync. Ordering:
        # this runs BEFORE the widgets are filled, so the pinned slider and
        # the stored value agree the first time they are drawn. Its
        # explicit apply is a no-op here (the layers are still initializing
        # from construction); the startup push to the camera is bring-up's
        # (ScopeSession.configure_scope applies BF).
        ctx.image_settings.reconcile_layers_to_camera_caps()

        for layer in common_utils.get_layers():
            ctx.image_settings.layer_lookup(layer=layer).sync_widgets_from_settings()

        self.set_ui_features_for_scope()

    def update_bullseye_state(self):
        gui_logger.toggle('BULLSEYE', self.ids['enable_bullseye_btn_id'].state == 'down')
        if self.ids['enable_bullseye_btn_id'].state == 'down':
            _app_ctx.ctx.viewer.update_shader(false_color='BF')
            _app_ctx.ctx.scope_display.use_bullseye = True
        else:
            layer = common_utils.get_opened_layer(_app_ctx.ctx.image_settings)
            if layer is not None:
                layer_obj = _app_ctx.ctx.image_settings.layer_lookup(layer=layer)
                if layer_obj.ids['false_color'].active:
                    _app_ctx.ctx.viewer.update_shader(false_color=layer)

            _app_ctx.ctx.scope_display.use_bullseye = False

    def load_image_modes(self):
        """Populate the image-mode spinner: every mode, on every camera."""
        self.ids['image_mode_spinner'].values = image_mode.available_mode_labels()

    # Drives the 8-bit binning depth-loss hint row; the row height follows the
    # label's wrapped texture so the multi-line warning is not clipped.
    binning_depth_hint_active = BooleanProperty(False)
    # Disables the frame size and binning controls on a scope with no camera
    # connected; set from scope.camera_connected when the camera's
    # capabilities are synced.
    camera_connected = BooleanProperty(True)

    def _refresh_binning_depth_hint(self):
        """Show the depth-loss hint below the binning control only when binning
        is active in an 8-bit mode (the binned range is truncated on save).
        """
        if 'binning_depth_hint_row' not in self.ids:
            return
        scope_display = getattr(_app_ctx.ctx, 'scope_display', None)
        if scope_display is None:
            return
        binning_size = self._ui_binning_size()
        self.binning_depth_hint_active = image_mode.depth_truncation_warning_active(
            binning_size, scope_display.image_mode
        )

    # Drives the JPG depth hint row: the API's predicate, rendered.
    jpg_depth_hint_active = BooleanProperty(False)

    def _refresh_jpg_depth_hint(self):
        """Show the JPG depth hint when a full-depth mode is paired with a JPG output."""
        settings = _app_ctx.ctx.settings
        self.jpg_depth_hint_active = image_mode.jpg_depth_warning_active(
            image_mode.resolve_settings_image_mode(settings),
            (
                settings['image_output_format']['live'],
                settings['image_output_format']['sequenced'],
            ),
        )

    def select_image_mode(self):
        ctx = _app_ctx.ctx

        label = self.ids['image_mode_spinner'].text
        mode = image_mode.LABEL_TO_IMAGE_MODE.get(label)
        if mode is None:
            return  # 'Select' placeholder or an unknown label -- ignore
        gui_logger.select('IMAGE_MODE', mode)

        # During app init, bring-up applies the stored format while the
        # camera start gate is still closed, and the spinner is only being
        # set from the store; a second apply from here would race it.
        if ctx.initializing:
            self._redraw_image_mode()
            return

        session = ctx.session
        submit_reported(
            lambda: session.set_image_mode(mode),
            self._redraw_image_mode,
            'IMAGE_MODE',
            lane=ctx.camera_executor,
        )

    def _redraw_image_mode(self):
        """Show the stored image mode: the display mode, the selector, the depth hint."""
        ctx = _app_ctx.ctx
        mode = image_mode.resolve_settings_image_mode(ctx.settings)
        ctx.scope_display.image_mode = mode
        label = image_mode.IMAGE_MODE_LABELS[mode]
        if self.ids['image_mode_spinner'].text != label:
            # The selector going back to the stored mode is the app's write,
            # not a pick.
            gui_logger.note_write_back('IMAGE_MODE', mode)
            self.ids['image_mode_spinner'].text = label
        self._refresh_binning_depth_hint()
        self._refresh_jpg_depth_hint()

    def select_live_image_output_format(self):
        fmt = self.ids['live_image_output_format_spinner'].text
        # The settings load sets the spinner from the store, which fires this
        # too; that is not a pick, so nothing is logged or written.
        if fmt == _app_ctx.ctx.settings['image_output_format']['live']:
            return
        gui_logger.select('LIVE_IMAGE_OUTPUT_FORMAT', fmt)
        run_reported(
            lambda: _app_ctx.ctx.update_settings('image_output_format.live', fmt),
            None,
            'LIVE_IMAGE_OUTPUT_FORMAT',
        )
        self._refresh_jpg_depth_hint()
        # The JPG-quality row's visibility (and disabled state) follows the
        # selected format declaratively in lumaviewpro.kv (jpg_quality_row binds
        # to live_image_output_format_spinner.text), so no toggle is needed here.

    def update_jpg_quality(self, value):
        quality = int(value)
        _app_ctx.ctx.update_settings('jpg_quality', quality)
        if 'jpg_quality_value_label' in self.ids:
            self.ids['jpg_quality_value_label'].text = str(quality)
        gui_logger.slider('JPG_QUALITY', quality)

    def select_sequenced_image_output_format(self):
        fmt = self.ids['sequenced_image_output_format_spinner'].text
        # As the live format: a spinner set from the store is not a pick.
        if fmt == _app_ctx.ctx.settings['image_output_format']['sequenced']:
            return
        gui_logger.select('SEQUENCED_IMAGE_OUTPUT_FORMAT', fmt)
        run_reported(
            lambda: _app_ctx.ctx.update_settings('image_output_format.sequenced', fmt),
            None,
            'SEQUENCED_IMAGE_OUTPUT_FORMAT',
        )
        self._refresh_jpg_depth_hint()

    def select_video_recording_format(self) -> None:
        gui_logger.select('VIDEO_RECORDING_FORMAT', self.ids['video_recording_format_spinner'].text)
        as_frames = self.ids['video_recording_format_spinner'].text != 'mp4'
        _app_ctx.ctx.update_settings('video_as_frames', as_frames)

    def update_scale_bar_state(self):
        enabled = self.ids['enable_scale_bar_btn'].state == 'down'
        gui_logger.toggle('SCALE_BAR', enabled)
        session = _app_ctx.ctx.session
        run_reported(lambda: session.set_scale_bar(enabled), None, 'SCALE_BAR')

    def update_crosshairs_state(self):
        enabled = self.ids['enable_crosshairs_btn'].state == 'down'
        gui_logger.toggle('CROSSHAIRS', enabled)
        scope_display = _app_ctx.ctx.scope_display
        if self.ids['enable_crosshairs_btn'].state == 'down':
            scope_display.use_crosshairs = True
            scope_display.show_crosshairs(True)
        else:
            scope_display.use_crosshairs = False
            scope_display.show_crosshairs(False)

    def update_live_image_histogram_equalization(self):
        ctx = _app_ctx.ctx
        enabled = self.ids['enable_live_image_histogram_equalization_btn'].state == 'down'
        gui_logger.toggle('LIVE_HISTOGRAM_EQUALIZATION', enabled)
        ctx.scope_display.use_live_image_histogram_equalization = enabled
        ctx.live_histo_setting = enabled

    def update_show_tooltips(self):
        ctx = _app_ctx.ctx
        enabled = self.ids['show_tooltips_btn'].state == 'down'
        gui_logger.toggle('SHOW_TOOLTIPS', enabled)
        ctx.show_tooltips = enabled
        ctx.update_settings('show_tooltips', enabled)

    def apply_stimulation_support(self):
        """Push the persisted global stimulation enable to every channel.

        Single owner of the per-layer stimulation sync. Reads
        ``settings['stimulation_enabled']`` (the source of truth, populated
        by the settings load) rather than a widget, so the startup load and
        the Advanced Settings toggle both drive the same path. Firmware
        without stim support can never enable it, even if a stale setting
        says otherwise.
        """
        settings = _app_ctx.ctx.settings
        stimulation_enabled = firmware_stim_supported() and settings['stimulation_enabled']
        settings['stimulation_enabled'] = stimulation_enabled

        # Update all layer controls
        for layer in common_utils.get_layers():
            if layer in common_utils.get_fluorescence_layers():
                layer_obj = _app_ctx.ctx.image_settings.layer_lookup(layer=layer)
                if layer_obj:
                    if stimulation_enabled:
                        # Enable stimulation features
                        layer_obj.stimulation_support = True
                        # Don't automatically show stim controls, just enable support
                    else:
                        # Disable stimulation features
                        layer_obj.stimulation_support = False
                        layer_obj.show_stim_controls = False
                        layer_obj.show_camera_controls = True
                        # Set stim to disabled
                        if 'stim_disable_btn' in layer_obj.ids:
                            layer_obj.ids['stim_disable_btn'].active = True
                        # Disable stim_config
                        if 'stim_config' in settings[layer]:
                            settings[layer]['stim_config']['enabled'] = False

    def load_binning_sizes(self):
        sizes = _app_ctx.ctx.lumaview.scope.capabilities.camera_binning_sizes
        self.ids['binning_spinner'].values = [f'{s}x{s}' for s in sizes]

    def _ui_binning_size(self) -> int:
        """The binning factor the store holds, which the panel shows."""
        settings = _app_ctx.ctx.settings
        return binning.binning_size_str_to_int(settings['binning']['size'])

    def select_binning_size(self):
        ctx = _app_ctx.ctx
        label = self.ids['binning_spinner'].text
        gui_logger.select('BINNING', label)

        # During app init, bring-up applies the stored binning and frame;
        # the spinner is only being set from the store.
        if ctx.initializing:
            self._redraw_framing()
            return

        size = binning.binning_size_str_to_int(label)
        session = ctx.session
        # The boxes show the frame the new binning will give while the apply
        # runs, so an edit typed meanwhile is read against the binning the
        # spinner shows; the redraw then shows what the camera delivered.
        preview = session.frame_at_binning(size)
        self._write_frame_text(preview['width'], preview['height'])
        submit_reported(
            lambda: session.set_binning_size(size),
            self._framing_applied,
            'BINNING',
            lane=ctx.camera_executor,
        )

    def _framing_applied(self) -> None:
        """Record the framing a person's binning pick or frame edit left, then show it."""
        settings = _app_ctx.ctx.settings
        gui_logger.frame_size(
            settings['frame']['width'], settings['frame']['height'], self._ui_binning_size()
        )
        self._redraw_framing()

    def _redraw_framing(self) -> None:
        """Show the stored binning and frame: the selector, the boxes, the hint, the field of view."""
        settings = _app_ctx.ctx.settings
        label = settings['binning']['size']
        if self.ids['binning_spinner'].text != label:
            # The selector going back to the stored binning is the app's
            # write, not a pick.
            gui_logger.note_write_back('BINNING', label)
            self.ids['binning_spinner'].text = label
        self._write_frame_text(settings['frame']['width'], settings['frame']['height'])
        self._refresh_binning_depth_hint()
        self.refresh_fov_labels()

    def reconfigure_for_scope(self) -> None:
        """Apply the current scope to the UI in the canonical order.

        set_ui_features_for_scope first (control visibility + the read-only
        model label), then a stage redraw for the scope's geometry. Runs at
        startup, once the scope is up; a model selected later is saved for
        the next start and changes nothing here.
        """
        ctx = _app_ctx.ctx
        self.set_ui_features_for_scope()
        ctx.stage.full_redraw()

    def show_scope_model(self) -> None:
        """Show the model the scope runs as, and one saved for the next start."""
        ctx = _app_ctx.ctx
        self.current_scope_model = ctx.lumaview.scope.layer_identity.model or ''
        self.next_start_model = ctx.session.model_at_next_start or ''

    def set_ui_features_for_scope(self) -> None:
        ctx = _app_ctx.ctx

        microscope_settings = ctx.motion_settings.ids['microscope_settings_id']

        microscope_settings.show_scope_model()

        # Which motion hardware exists is asked of the drivers, never of the
        # selected model. scopes.json describes the model picked in Advanced
        # Settings, and that selection is editable while the app runs -- gate
        # the controls on it and the UI offers an XY stage on a scope that has
        # none, after which a protocol images a single position while
        # labelling every file with a different well.
        caps = ctx.lumaview.scope.capabilities

        motion_settings = ctx.motion_settings
        motion_settings.set_turret_control_visibility(visible=caps.has_turret)
        motion_settings.set_xystage_control_visibility(visible=caps.has_xy_stage)
        motion_settings.set_tiling_control_visibility(visible=caps.has_xy_stage)
        motion_settings.set_focus_control_visibility(visible=caps.has_focus)

        image_settings = ctx.image_settings
        # Which layers exist comes from the scope's resolved identity --
        # the one path that also serves headless callers -- refreshed by
        # reconfigure_for_scope before this runs. Visibility and titles
        # are per-layer from the record, so a filterset carrying only
        # some channels (a Green-only unit) shows exactly what the unit
        # has, with each drawer titled by what its layer IS on this unit.
        identity = ctx.lumaview.scope.layer_identity
        present = {layer.key_name for layer in identity.layers}
        image_settings.set_df_layer_control_visibility(visible='DF' in present)
        image_settings.set_lumi_layer_control_visibility(visible='Lumi' in present)
        for color in common_utils.get_fluorescence_layers():
            image_settings.set_fluorescence_layer_control_visibility(
                color, visible=color in present
            )
        image_settings.set_phasecontrast_layer_control_visibility(visible='PC' in present)
        image_settings.apply_layer_titles(identity.layers)
        image_settings.set_layer_focus_visibility(visible=caps.has_focus)

        protocol_settings = ctx.motion_settings.ids['protocol_settings_id']
        protocol_settings.set_labware_selection_visibility(visible=caps.has_xy_stage)
        protocol_settings.set_focus_control_visibility(visible=caps.has_focus)

        ctx.motion_settings.ids['post_processing_id'].ids[
            'stitch_controls_id'
        ].set_button_enabled_state(state=caps.has_xy_stage)

        if not caps.has_xy_stage:
            # Stage-less scopes (Lumi, LS820) keep a single-plate
            # ("Center Plate") graphic in the protocol tab so the crosshair
            # position is visible; bring-up has already put the scope on it.
            # Only the XY motion capability is disabled (set below). Stitch
            # is hidden -- it needs tiling.
            ctx.motion_settings.ids['post_processing_id'].hide_stitch()

        # Nothing to write: session.motion_enabled and the stage crosshair
        # each ask the drivers for the XY fact at read time, so there is no
        # copy here to keep in step. Run state is still republished -- the
        # derivation's consumers are edge-driven, and a scope change can
        # move motion_enabled.
        ctx.session.notify_run_state()

        # Size the protocol-tab stage holder to its width-based aspect for
        # every scope. The plate graphic now renders on XYStage=False scopes
        # too (single Center Plate), so the holder is no longer collapsed.
        # The kv-defined ``protocol_stage_holder_id`` FloatLayout has
        # ``height: self.width * 2 / 3``; set it explicitly and bind to width
        # so the holder follows resize (bind once, tracked on the widget so
        # repeated scope toggles don't stack handlers).
        protocol_stage_holder = protocol_settings.ids.get('protocol_stage_holder_id')
        if protocol_stage_holder is not None:
            protocol_stage_holder.size_hint_y = None
            protocol_stage_holder.height = max(1, int(protocol_stage_holder.width * 2 / 3))
            if not getattr(protocol_stage_holder, '_lvp_height_bound', False):
                protocol_stage_holder.bind(
                    width=lambda inst, w: setattr(inst, 'height', max(1, int(w * 2 / 3)))
                )
                protocol_stage_holder._lvp_height_bound = True

        # UI-1 follow-up (2026-05-03): cheap "reset on switch" -- explicit
        # resort of both accordions after any scope-config change so
        # successive LS850 <-> LS820 <-> LS620 transitions can't leave the
        # children list in a non-canonical state. Eric 2026-05-03:
        # "maybe it could do a fully reset when you switch" -- this is
        # that approach.
        ctx.motion_settings._resort_accordion()
        image_settings._resort_accordion()

    def _typed_frame_dimensions(self) -> dict:
        """The size currently TYPED into the frame fields.

        Only the handler applying the edit wants this. Every other
        consumer wants the size the camera delivered, which lives in
        settings['frame'] -- the fields are an editor, and until the
        apply lands they can hold a size no camera is at.

        Raises:
            ValueError: the fields do not hold a pair of integers.
        """
        return {
            'width': int(self.ids['frame_width_id'].text),
            'height': int(self.ids['frame_height_id'].text),
        }

    def _write_frame_text(self, width, height) -> None:
        """Write a frame read-back into the boxes, unless the user is typing.

        The boxes commit on focus loss (`on_focus: if not self.focus:
        root.frame_size(...)`), so a size written underneath a part-typed
        entry is not merely displayed -- it is committed as a framing change
        when the user clicks away. Each box is guarded on its OWN focus: they
        are edited one at a time, and skipping both because one is focused
        would leave the other showing a size no camera is at.

        Every writer of these boxes goes through here so the guard cannot be
        present at three sites and missing at the fourth.
        """
        for widget_id, value in (
            ('frame_width_id', width),
            ('frame_height_id', height),
        ):
            box = self.ids[widget_id]
            if box.focus:
                continue
            new_text = str(value)
            # Rewriting the same string churns the enclosing ScrollView.
            if box.text != new_text:
                box.text = new_text

    def frame_size(self, committed_id: str):
        """Apply a user edit of the frame width/height fields.

        ``committed_id`` names the box whose commit invoked this. Both boxes
        bind this one handler, and a record that cannot say which box the user
        left is not a record of what they did; it is required rather than
        defaulted because a caller that cannot answer cannot log the edit
        either.

        The typed value is a displayed (post-binning) size; the Session works
        out the region it implies at the stored binning, applies it, and the
        boxes then show what the camera delivered.
        """
        logger.info('[LVP Main  ] MicroscopeSettings.frame_size()')
        ctx = _app_ctx.ctx

        record, axis = _FRAME_BOXES[committed_id]
        # First act: the user typed it whether or not a camera is there to
        # hear about it.
        gui_logger.text_input(record, self.ids[committed_id].text)

        try:
            typed = self._typed_frame_dimensions()
        except ValueError:
            # An entry that is not a pair of integers is a CORRECTION, not a
            # request. Substituting the stored size and applying it reported a
            # framing the user never asked for -- emptying a box logged the
            # size already in force, so the bundle claimed an edit that never
            # happened while the box sat blank. Put both boxes back and stop.
            frame = ctx.settings['frame']
            gui_logger.text_input(f'{record}_APPLIED', frame[axis])
            self._write_frame_text(frame['width'], frame['height'])
            return

        # Rapid edits (width, then height, each committing on focus loss)
        # fold into one apply while one is in flight -- see _CoalescingApplier.
        if self._frame_size_applier.submit((typed['width'], typed['height'])):
            session = ctx.session
            applier = self._frame_size_applier
            submit_reported(
                lambda: applier.apply_pending(lambda wh: session.set_frame_size(*wh)),
                self._framing_applied,
                'FRAME_SIZE',
                lane=ctx.camera_executor,
            )

    def refresh_fov_labels(self) -> None:
        """Recompute the FOV readout from the current delivered-sourced
        frame settings and the UI binning. With no known objective there is
        no field of view to show, so the readout is blank."""
        ctx = _app_ctx.ctx
        settings = ctx.settings
        objective = ctx.scope.runtime_state.get_current_objective()
        if objective is None:
            self.ids['field_of_view_width_id'].text = ''
            self.ids['field_of_view_height_id'].text = ''
            return
        fov_size = config_ui_getters.get_field_of_view(
            focal_length=objective['focal_length'],
            frame_size=settings['frame'],
            binning_size=ctx.session.get_binning_size(),
        )
        fov_w_text, fov_h_text = common_utils.format_field_of_view(fov_size)
        self.ids['field_of_view_width_id'].text = fov_w_text
        self.ids['field_of_view_height_id'].text = fov_h_text

    def open_advanced_settings(self):
        """Open the Advanced Settings modal (power-user / rarely-touched rows)."""
        gui_logger.button('OPEN_ADVANCED_SETTINGS')
        from ui.advanced_settings import AdvancedSettings

        self._advanced_settings_popup = AdvancedSettings()
        self._advanced_settings_popup.open()

    def generate_support_report(self):
        """Show confirmation dialog, then generate a tech support report."""
        gui_logger.button('GENERATE_SUPPORT_REPORT')
        from ui.notification_popup import show_confirmation_popup

        # The report homes only the axes the scope has, so a scope with
        # none is not told its stage will move.
        moves = (
            'The stage will be homed and moved during testing.\n'
            'Please remove any samples from the stage.\n\n'
            if _app_ctx.ctx.scope.capabilities.axes
            else ''
        )
        show_confirmation_popup(
            title='Tech Support Report',
            message=(
                'This will create a diagnostic report to send to\n'
                'Etaluma Tech Support.\n\n'
                f'{moves}'
                'This may take a few minutes.'
            ),
            confirm_text='Generate',
            cancel_text='Cancel',
            on_confirm=self._start_support_report,
        )

    def _start_support_report(self):
        session = _app_ctx.ctx.session
        self._make_zip(
            'Generating Support Report...',
            'GENERATE_SUPPORT_REPORT',
            lambda progress: session.make_support_report(
                output_dir=path_utils.desktop_folder(), on_progress=progress
            ),
            budget_of=session.make_support_report,
        )

    def zip_logs_only(self):
        """Quick zip of logs + data + recent protocols. No hardware tests."""
        gui_logger.button('ZIP_LOGS')
        self._make_zip(
            'Zipping Logs...',
            'ZIP_LOGS',
            lambda progress: _app_ctx.ctx.session.make_logs_zip(
                output_dir=path_utils.desktop_folder(), on_progress=progress
            ),
        )

    def _make_zip(self, title, label, make, budget_of=None):
        """Run one of the Session's support zips under a progress popup, then show where it went.

        The zip runs on the diagnostics executor, so a Stop never waits
        behind it. A zip that was not saved is reported by the GUI boundary
        in the report's own words; the popup then just closes.
        """
        from ui.notification_popup import show_notification_popup
        from ui.progress_popup import CustomPopup

        popup = CustomPopup(title=title, auto_dismiss=False)
        popup.open()
        produced = {}

        def _progress(pct, msg):
            def _show_progress(dt):
                popup.progress = pct
                popup.text = msg

            Clock.schedule_once(_show_progress, 0)

        def _make():
            produced['saved'] = make(_progress)

        def _show():
            popup.dismiss()
            saved = produced.get('saved')
            if saved is not None:
                show_notification_popup(title=saved.title, message=saved.message)

        submit_reported(
            _make,
            _show,
            label,
            lane=_app_ctx.ctx.session.executor_bundle.diagnostics_executor,
            budget_of=budget_of,
        )
