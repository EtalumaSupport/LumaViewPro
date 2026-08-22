# Copyright Etaluma, Inc.
import logging

from kivy.clock import Clock
from kivy.metrics import dp
from kivy.uix.accordion import AccordionItem
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView

import modules.app_context as _app_ctx
import modules.common_utils as common_utils
from modules import gui_logger

logger = logging.getLogger('LVP.ui.image_settings')


# ============================================================================
# Accordion Item Widgets (Layer/Channel Selection)
# ============================================================================


class AccordionItemXyStageControl(AccordionItem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_gui(self, full_redraw: bool = False):
        self.ids['xy_stagecontrol_id'].update_gui(full_redraw=full_redraw)


class AccordionItemImageSettingsBase(AccordionItem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def accordion_collapse(self):
        _app_ctx.ctx.image_settings.accordion_collapse()


class AccordionItemImageSettingsLumiControl(AccordionItemImageSettingsBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class AccordionItemImageSettingsDfControl(AccordionItemImageSettingsBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class AccordionItemImageSettingsRedControl(AccordionItemImageSettingsBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class AccordionItemImageSettingsGreenControl(AccordionItemImageSettingsBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class AccordionItemImageSettingsBlueControl(AccordionItemImageSettingsBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# ============================================================================
# ImageSettings -- Right Sidebar Panel (Channel Controls, LED, Exposure)
# ============================================================================


class ImageSettings(BoxLayout):
    settings_width = dp(300)
    tab_width = dp(30)

    # Canonical top-to-bottom display order for the right-side accordion.
    # Used by _resort_accordion() so live scope-model transitions
    # (LS620 -> LS850 etc.) place re-added layer accordions in the right
    # spot instead of pinning them to the bottom (UI-1, 2026-05-02).
    _LAYER_DISPLAY_ORDER = ('BF', 'PC', 'DF', 'Blue', 'Green', 'Red', 'Lumi')

    # Ownership boundary for the accordion LED/camera reconcile: the
    # reconcile belongs to genuine USER drawer clicks. A PROGRAMMATIC
    # expansion (manual step navigation) owns its entire LED + camera
    # outcome through the LED authority, so while this is True the
    # reconcile defers to that owner. Same flag idiom as
    # LayerControl._suppressing_led_log.
    _suppress_reconcile_for_programmatic_expand = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.debug('[LVP Main  ] ImageSettings.__init__()')
        self._accordion_item_df_control_visible = False
        self._accordion_item_df_control = AccordionItemImageSettingsDfControl()
        self._accordion_item_lumi_control_visible = False
        self._accordion_item_lumi_control = AccordionItemImageSettingsLumiControl()
        self._accordion_item_fluorescence_control_visible = False
        self._accordion_item_red_control = AccordionItemImageSettingsRedControl()
        self._accordion_item_green_control = AccordionItemImageSettingsGreenControl()
        self._accordion_item_blue_control = AccordionItemImageSettingsBlueControl()
        # PC accordion item is defined inline in lumaviewpro.kv (unlike
        # DF/Lumi/R/G/B which are Python class instances). Populated in
        # _init_ui once self.ids is available. Visible-by-default here
        # matches the kv starting state; set_phasecontrast_layer_control_visibility
        # hides it for scopes that declare PhaseContrast=false
        # (LS560/LS620 -- PC on those scopes is BF with a mechanical
        # phase slider, not a separate LED channel).
        self._accordion_item_pc_control = None
        self._accordion_item_pc_control_visible = True
        self._init_ui_retries = 0
        # Debounce accordion_collapse -- Kivy fires multiple collapse events
        # when switching tabs (one per item). Trigger collapses them into one.
        self._accordion_collapse_trigger = Clock.create_trigger(
            lambda dt: self._do_accordion_collapse(), 0
        )
        Clock.schedule_once(self._init_ui, 0)

    def layer_lookup(self, layer: str):
        LAYER_MAP = {
            'DF': self._accordion_item_df_control,
            'Lumi': self._accordion_item_lumi_control,
            'Blue': self._accordion_item_blue_control,
            'Red': self._accordion_item_red_control,
            'Green': self._accordion_item_green_control,
        }

        if layer in LAYER_MAP:
            return LAYER_MAP[layer].ids[layer]
        else:
            return self.ids[layer]

    def accordion_item_lookup(self, layer: str):
        LAYER_MAP = {
            'DF': self._accordion_item_df_control,
            'Lumi': self._accordion_item_lumi_control,
            'Blue': self._accordion_item_blue_control,
            'Red': self._accordion_item_red_control,
            'Green': self._accordion_item_green_control,
        }

        if layer in LAYER_MAP:
            return LAYER_MAP[layer]
        else:
            return self.ids[f'{layer}_accordion']

    def set_expanded_layer(self, layer: str, *largs) -> None:
        """
        Expand the specified layer accordion and collapse all others.
        Cleans up ScrollView viewport textures on collapse to prevent memory accumulation.
        Accordion toggling is always disabled during protocol execution to prevent memory leaks.
        """

        # Skip accordion toggling during protocol execution to prevent memory leaks
        if _app_ctx.ctx.session.run_lockout:
            return

        gui_logger.select('IMAGE_LAYER', layer)

        # Ordering invariant: the guard is set BEFORE the mutation loop
        # (the collapse events it fires prime the reconcile trigger) and
        # its CLEAR is scheduled in the finally AFTER the mutations --
        # Kivy runs same-frame timeout-0 events in scheduling order, so a
        # clear scheduled before priming would fire first and re-expose
        # the reconcile race. try/finally: an exception mid-loop must not
        # wedge the guard (that would permanently disable the user-click
        # drawer reconcile). The next-tick clear also absorbs a user click
        # landing on this same frame -- that click's reconcile is skipped
        # once; the next real click re-fires.
        self._suppress_reconcile_for_programmatic_expand = True
        try:
            for a_layer in common_utils.get_layers():
                accordion_item_obj = self.accordion_item_lookup(layer=a_layer)

                # Check if we need to collapse this accordion
                target_collapsed = layer != a_layer

                if layer == a_layer:
                    accordion_item_obj.collapse = False
                else:
                    # Before collapsing, clean up ScrollView to prevent memory leak
                    if not accordion_item_obj.collapse and target_collapsed:
                        layer_control = self.layer_lookup(layer=a_layer)
                        # Find and clean up ScrollView in this layer control
                        for child in layer_control.walk():
                            if isinstance(child, ScrollView):
                                # Schedule cleanup after collapse animation completes
                                from ui.ui_helpers import cleanup_scrollview_viewport

                                Clock.schedule_once(
                                    lambda dt, sv=child: cleanup_scrollview_viewport(sv), 0.2
                                )

                    accordion_item_obj.collapse = True
        finally:
            Clock.schedule_once(
                lambda dt: setattr(self, '_suppress_reconcile_for_programmatic_expand', False),
                0,
            )

    def set_lumi_layer_control_visibility(self, visible: bool) -> None:
        if visible:
            self._show_lumi_layer_control()
        else:
            self._hide_lumi_layer_control()

    def _show_lumi_layer_control(self):
        if not self._accordion_item_lumi_control_visible:
            self._accordion_item_lumi_control_visible = True
            self.ids['accordion_id'].add_widget(self._accordion_item_lumi_control, 0)
            self._resort_accordion()

    def _hide_lumi_layer_control(self):
        settings = _app_ctx.ctx.settings
        if settings:
            settings['Lumi']['acquire'] = None
        if self._accordion_item_lumi_control_visible:
            self._accordion_item_lumi_control.collapse = True
            self._accordion_item_lumi_control_visible = False
            self.ids['accordion_id'].remove_widget(self._accordion_item_lumi_control)

    def set_df_layer_control_visibility(self, visible: bool) -> None:
        if visible:
            self._show_df_layer_control()
        else:
            self._hide_df_layer_control()

    def _show_df_layer_control(self):
        if not self._accordion_item_df_control_visible:
            self._accordion_item_df_control_visible = True
            self.ids['accordion_id'].add_widget(self._accordion_item_df_control, 0)
            self._resort_accordion()

    def _hide_df_layer_control(self):
        settings = _app_ctx.ctx.settings
        if settings:
            settings['DF']['acquire'] = None
        if self._accordion_item_df_control_visible:
            self._accordion_item_df_control.collapse = True
            self._accordion_item_df_control_visible = False
            self.ids['accordion_id'].remove_widget(self._accordion_item_df_control)

    def set_phasecontrast_layer_control_visibility(self, visible: bool) -> None:
        if visible:
            self._show_pc_layer_control()
        else:
            self._hide_pc_layer_control()

    def _resolve_pc_accordion(self):
        """Return the PC_accordion widget, resolving from self.ids on
        first use. `set_ui_features_for_scope()` runs during
        load_settings before the Clock-scheduled _init_ui fires, so a
        ref captured only in _init_ui arrives too late for the initial
        hide. Lazy resolution keeps both code paths correct.
        """
        if self._accordion_item_pc_control is None:
            # Cache the REAL widget (`.__self__`), never the WeakProxy
            # self.ids hands out: on scopes that hide this item,
            # remove_widget drops the tree's only strong reference, and a
            # proxy-cached item is garbage-collected with all its layer
            # widgets -- every later deref of those ids then raises
            # ReferenceError. The Python-constructed accordion items
            # survive hiding precisely because their instance attributes
            # are strong references; this cache must match.
            proxy = self.ids.get('PC_accordion')
            self._accordion_item_pc_control = proxy.__self__ if proxy is not None else None
        return self._accordion_item_pc_control

    def _show_pc_layer_control(self):
        widget = self._resolve_pc_accordion()
        if widget is not None and not self._accordion_item_pc_control_visible:
            self._accordion_item_pc_control_visible = True
            self.ids['accordion_id'].add_widget(widget)
            self._resort_accordion()

    def _hide_pc_layer_control(self):
        settings = _app_ctx.ctx.settings if _app_ctx.ctx else None
        if settings and 'PC' in settings:
            settings['PC']['acquire'] = None
        widget = self._resolve_pc_accordion()
        if widget is not None and self._accordion_item_pc_control_visible:
            widget.collapse = True
            self._accordion_item_pc_control_visible = False
            self.ids['accordion_id'].remove_widget(widget)

    def set_fluoresence_layer_controls_visibility(self, visible: bool) -> None:
        if visible:
            self._show_fluorescence_layer_controls()
        else:
            self._hide_fluorescence_layer_controls()

    def _show_fluorescence_layer_controls(self):
        if not self._accordion_item_fluorescence_control_visible:
            self._accordion_item_fluorescence_control_visible = True
            self.ids['accordion_id'].add_widget(self._accordion_item_blue_control, 0)
            self.ids['accordion_id'].add_widget(self._accordion_item_green_control, 0)
            self.ids['accordion_id'].add_widget(self._accordion_item_red_control, 0)
            self._resort_accordion()

    def _hide_fluorescence_layer_controls(self):
        settings = _app_ctx.ctx.settings
        if settings:
            settings['Red']['acquire'] = None
            settings['Green']['acquire'] = None
            settings['Blue']['acquire'] = None
        if self._accordion_item_fluorescence_control_visible:
            self._accordion_item_blue_control.collapse = True
            self._accordion_item_green_control.collapse = True
            self._accordion_item_red_control.collapse = True

            self._accordion_item_fluorescence_control_visible = False
            self.ids['accordion_id'].remove_widget(self._accordion_item_blue_control)
            self.ids['accordion_id'].remove_widget(self._accordion_item_green_control)
            self.ids['accordion_id'].remove_widget(self._accordion_item_red_control)

    def _resort_accordion(self):
        """Rebuild the accordion children list in canonical layer order.

        Live scope-model transitions (LS620 -> LS850, etc.) re-add
        previously hidden layer-control widgets via add_widget(...,0),
        which appends to the children list and ends up at the BOTTOM of
        the visible accordion regardless of canonical order. After every
        ``_show_*`` call we re-sort so the order matches
        ``_LAYER_DISPLAY_ORDER`` regardless of insertion sequence.

        Kivy renders the children list bottom-to-top, so we walk the
        canonical order in REVERSE and re-add each currently-visible
        widget. AccordionItem state (``collapse``, internal anim) lives
        on the widget instance, so remove + re-add preserves it.
        """
        accordion = self.ids.get('accordion_id') if hasattr(self, 'ids') else None
        if accordion is None:
            return

        widget_for_layer = {
            'BF': self.ids.get('BF_accordion'),
            'PC': self._resolve_pc_accordion(),
            'DF': self._accordion_item_df_control,
            'Blue': self._accordion_item_blue_control,
            'Green': self._accordion_item_green_control,
            'Red': self._accordion_item_red_control,
            'Lumi': self._accordion_item_lumi_control,
        }
        flu_visible = self._accordion_item_fluorescence_control_visible
        visible_for_layer = {
            'BF': True,
            'PC': self._accordion_item_pc_control_visible,
            'DF': self._accordion_item_df_control_visible,
            'Blue': flu_visible,
            'Green': flu_visible,
            'Red': flu_visible,
            'Lumi': self._accordion_item_lumi_control_visible,
        }

        # Walk the live children list directly and remove any widget we
        # track. ``widget.parent is accordion`` was unreliable here --
        # Kivy's parent attribute can lag the children list during
        # add_widget calls inside the same event tick. Membership in
        # ``accordion.children`` is the ground truth.
        #
        # Compare via ``widget.uid`` rather than Python ``id()`` because
        # ``self.ids.get(...)`` returns a Kivy WeakProxy whose Python id
        # differs from the underlying widget's id. Today all the
        # right-side widgets are python instance refs (no kv ids in
        # ``widget_for_layer``) so id() happens to work, but the left-
        # side resort hit this exact trap 2026-05-03 -- using uid is the
        # defensive choice.
        tracked_uids = {w.uid for w in widget_for_layer.values() if w is not None}
        present = [w for w in list(accordion.children) if w.uid in tracked_uids]
        for widget in present:
            accordion.remove_widget(widget)

        # Walk forward through canonical order. ``add_widget`` with no
        # index prepends to the children list; the accordion renders
        # children in reverse order (children[0] is drawn last -> bottom),
        # so the FIRST canonical layer added ends up at the bottom of
        # the children list and at the TOP of the visual accordion. The
        # final iteration (Lumi) lands at children[0] -> bottom of display.
        for layer in self._LAYER_DISPLAY_ORDER:
            if not visible_for_layer.get(layer, False):
                continue
            widget = widget_for_layer.get(layer)
            if widget is None:
                continue
            # Defensive: if a widget still has a parent (e.g. transient
            # state during animation), detach it before adding so kivy
            # doesn't raise "already has a parent".
            if widget.parent is not None:
                try:
                    widget.parent.remove_widget(widget)
                except Exception:
                    pass
            accordion.add_widget(widget)

    def _init_ui(self, dt=0):
        ctx = _app_ctx.ctx
        if ctx is None:
            self._init_ui_retries += 1
            if self._init_ui_retries > 50:
                logger.error(
                    '[LVP Main  ] ImageSettings._init_ui: ctx still None after 50 retries, giving up'
                )
                return
            Clock.schedule_once(self._init_ui, 0.1)
            return
        self.assign_led_button_down_images()
        # Skip accordion_collapse during app initialization to prevent premature apply_settings
        if not ctx.initializing:
            self.accordion_collapse()
        self.sync_camera_capability_ranges()
        self.enable_image_stats_if_needed()

    def enable_image_stats_if_needed(self):
        if _app_ctx.ctx.engineering_mode:
            for layer in common_utils.get_layers():
                layer_obj = self.layer_lookup(layer=layer)
                layer_obj.ids['image_stats_mean_id'].height = '30dp'
                layer_obj.ids['image_stats_stddev_id'].height = '30dp'
                layer_obj.ids['image_af_score_id'].height = '30dp'

    def set_layer_exposure_ranges(self):
        ctx = _app_ctx.ctx
        for layer in common_utils.get_fluorescence_layers():
            layer_obj = self.layer_lookup(layer=layer)
            layer_obj.ids[
                'exp_slider'
            ].min = 1.0  # 1ms floor -- sub-ms never realistic for fluorescence
            layer_obj.ids['exp_slider'].max = ctx.max_exposure
            layer_obj.ids['exp_slider'].step = 1.0  # Integer steps only

        for layer in common_utils.get_transmitted_layers():
            layer_obj = self.layer_lookup(layer=layer)

            if layer == 'BF':
                # M25: Cap at 50ms but don't exceed camera capability.
                layer_obj.ids['exp_slider'].max = min(50, ctx.max_exposure)
            else:
                # M25: Cap at 200ms but don't exceed camera capability.
                layer_obj.ids['exp_slider'].max = min(200, ctx.max_exposure)

        for layer in common_utils.get_luminescence_layers():
            layer_obj = self.layer_lookup(layer=layer)
            layer_obj.ids['exp_slider'].min = 1.0  # 1ms floor
            layer_obj.ids['exp_slider'].max = ctx.max_exposure
            layer_obj.ids['exp_slider'].step = 1.0  # Integer steps only

    def set_layer_gain_ranges(self):
        """Size each layer's gain slider to the connected camera's cap.

        Parallel to set_layer_exposure_ranges. Pre-fix the gain slider
        was hardcoded 0-48 dB in the kv regardless of camera -- that
        let LS620 users drag past the usable range and black out the
        image. `ctx.max_gain` is populated from
        Lumascope.max_gain_db_cached in load_settings (and the same
        default-fallback pattern as exposure).
        """
        ctx = _app_ctx.ctx
        for layer in common_utils.get_layers():
            layer_obj = self.layer_lookup(layer=layer)
            layer_obj.ids['gain_slider'].max = ctx.max_gain

    def set_layer_autogain_support(self):
        """Gate the Auto Gain/Exp control on the camera's hardware AG/AE support.

        The checkbox drives hardware auto-gain/exposure (imaging.set_auto_gain
        -> driver.auto_gain), so on a camera without either (IDS U3-34Lx, FX2
        LS620) the control is a no-op and is hidden. Parallel to
        set_layer_gain_ranges: runs at every capability-sync point (connect /
        scope-change). The per-layer static rule (Lumi has no autogain) is
        preserved -- this only drives the orthogonal camera_autogain_support
        gate, AND-ed with autogain_support in the kv. Capability resolution +
        fail-safe live in camera_autogain_supported() (the single gate).

        Only the visibility gate is set here; the persisted per-layer auto_gain
        is NOT mutated. The effective enable (preference AND capability) is
        derived non-destructively at the consumption point
        (LayerControl.effective_auto_gain), so a capable camera's saved
        preference survives a swap to an AG-less body and back.
        """
        from modules.config_ui_getters import camera_autogain_supported

        supported = camera_autogain_supported()
        for layer in common_utils.get_layers():
            layer_obj = self.layer_lookup(layer=layer)
            layer_obj.camera_autogain_support = supported

    def sync_camera_capability_ranges(self):
        """Resync every per-layer camera control from the live camera caps.

        The single grouping of the per-layer capability setters, so the connect
        path (_init_ui) and the reconnect path apply the SAME set and can't
        drift to a subset (reconnect previously refreshed only exposure ranges).
        Callers that change the camera (reconnect) must refresh ctx.max_gain /
        ctx.max_exposure first -- these setters read those caps.
        """
        self.set_layer_exposure_ranges()
        self.set_layer_gain_ranges()
        self.set_layer_autogain_support()
        self.clamp_layer_settings_to_caps()

    def clamp_layer_settings_to_caps(self):
        """Bring every layer's stored gain/exposure within the live camera caps.

        A camera swap can leave a layer's persisted gain_db/exp_ms above the new
        body's physical maximum; applying that value blacks the channel out (and
        it would persist to current.json on the next save). Reconcile the stored
        value -- and its slider -- down to the cap for every layer, the same
        reconciliation load_settings performs, so connect and reconnect agree.
        An over-cap value cannot be honored by the hardware regardless.
        """
        ctx = _app_ctx.ctx
        settings = ctx.settings
        for layer in common_utils.get_layers():
            layer_obj = self.layer_lookup(layer=layer)
            if settings[layer]['gain_db'] > ctx.max_gain:
                settings[layer]['gain_db'] = ctx.max_gain
                layer_obj.ids['gain_slider'].value = ctx.max_gain
            if settings[layer]['exp_ms'] > ctx.max_exposure:
                settings[layer]['exp_ms'] = ctx.max_exposure
                layer_obj.ids['exp_slider'].value = ctx.max_exposure

    def open_or_default_layer(self):
        """The layer whose accordion is expanded, or 'BF' when none is open.

        Lets the reconnect path re-apply the VISIBLE layer's controls against
        the new camera (so e.g. its gain/exposure sliders reflect it), instead
        of only ever refreshing a hardcoded channel. Delegates the open-layer
        scan to common_utils.get_opened_layer (which guards the accordion
        lookup); BF is the default channel shown when every accordion is
        collapsed.
        """
        layer = common_utils.get_opened_layer(self)
        if layer is not None:
            return layer
        return 'BF'

    def assign_led_button_down_images(self):
        led_button_down_background_map = {
            'Red': './data/icons/ToggleRR.png',
            'Green': './data/icons/ToggleRG.png',
            'Blue': './data/icons/ToggleRB.png',
            'Lumi': './data/icons/ToggleRB.png',
        }

        for layer in common_utils.get_layers_with_led():
            button_down_image = led_button_down_background_map.get(
                layer, './data/icons/ToggleRW.png'
            )
            layer_obj = self.layer_lookup(layer=layer)
            layer_obj.ids['enable_led_btn'].background_down = button_down_image

    # Hide (and unhide) main settings
    def toggle_settings(self):
        if not _app_ctx.ctx.session.run_lockout:
            self.update_transmitted()
        # State after toggle reflects target visibility -- 'normal' = settings
        # tab going invisible (panel collapsing to side), 'down' = expanding.
        state_down = self.ids['toggle_imagesettings'].state == 'down'
        gui_logger.toggle('IMAGE_SETTINGS_PANEL', state_down)
        logger.info('[LVP Main  ] ImageSettings.toggle_settings()')
        ctx = _app_ctx.ctx
        lumaview = ctx.lumaview

        # move position of settings and stop histogram if main settings are collapsed
        if self.ids['toggle_imagesettings'].state == 'normal':
            self.pos = lumaview.width - self.tab_width, 0

            for layer in common_utils.get_layers():
                layer_obj = ctx.image_settings.layer_lookup(layer=layer)
                Clock.unschedule(layer_obj.ids['histo_id'].histogram)
                logger.info('[LVP Main  ] Clock.unschedule(lumaview...histogram)')
        else:
            self.pos = lumaview.width - self.settings_width, 0

        # if scope_display.play == True:
        #     scope_display.start()

    def update_transmitted(self):
        for layer in common_utils.get_transmitted_layers():
            layer_obj = self.layer_lookup(layer=layer)

            # Remove 'Colorize' option in transmitted channels control
            # -----------------------------------------------------
            # Remove CBT from transmitted channel control
            layer_obj.show_cbt = False
            label = layer_obj.ids['composite_threshold_label']
            slider = layer_obj.ids['composite_threshold_slider']
            text = layer_obj.ids['composite_threshold_text']
            label.text = ''
            label.visible = False
            label.opacity = 0
            slider.disabled = True
            slider.visible = False
            slider.cursor_size = '0dp', '0dp'
            slider.opacity = 0
            slider.value_track_color = (0.0,) * 4
            text.disabled = True
            text.visible = False
            text.width = '0dp'
            text.text = ''
            text.opacity = 0
            layer_obj.ids['false_color_label'].text = ''
            layer_obj.ids['false_color'].color = (0.0,) * 4

            # Adjust 'Illumination' range
            layer_obj.ids['ill_slider'].step = 1
            layer_obj.ids['ill_slider'].max = 50

    def accordion_collapse(self):
        """Called by Kivy on every accordion item collapse/expand.

        Kivy fires this multiple times when switching tabs (once per item).
        Debounced via Clock.create_trigger so the actual work runs once per frame.
        """
        self._accordion_collapse_trigger()

    def _do_accordion_collapse(self):
        logger.info('[LVP Main  ] ImageSettings.accordion_collapse()')

        ctx = _app_ctx.ctx

        # Skip during app initialization - will be called explicitly after init completes
        if ctx.initializing:
            return

        # Programmatic expansion (manual step navigation) owns its LED +
        # camera outcome through the authority; the reconcile below is for
        # genuine user drawer clicks. Checked at FIRE time so a trigger
        # primed before the guard was set still no-ops.
        if self._suppress_reconcile_for_programmatic_expand:
            return

        # Skip during protocol run: user clicking a different panel
        # shouldn't override the layer the protocol is actively driving,
        # and the "Protocol LEDs On" feature relies on step_navigation
        # having just turned the step's LED on -- killing it here would
        # negate that. The original guard checked protocol_led_on alone,
        # but that setting persists across protocol runs; combined with
        # the prior shape it incorrectly skipped LED cleanup during pure
        # Live-mode accordion switches whenever the user had previously
        # enabled Protocol LEDs On. Live-mode users then saw a
        # previously-enabled channel's LED stay lit until they enabled
        # the new channel (issue #659). Gating on protocol_running.is_set
        # collapses both checks and matches the original intent.
        # set_expanded_layer() already bails for programmatic paths; this
        # covers the user-click path so a mid-capture click doesn't kill
        # the running-step LED or apply a different layer's settings.
        if ctx.session.run_lockout:
            return

        # Issue #637: opening/closing the drawer must not send anything to
        # the camera or LEDs. Kivy's accordion auto-expands a different
        # item when the active one collapses (default Accordion behavior),
        # so a drawer-close event would otherwise trigger apply_settings
        # for whichever item Kivy auto-expanded -- applying that layer's
        # exposure/gain to the camera while the user's LED was still on a
        # different channel. Saturated-image symptom in #637.
        if self.ids['toggle_imagesettings'].state == 'normal':
            return

        # Reconcile each layer's LED to the open drawer: off the collapsed
        # layers, apply the open one. Switching the others off individually
        # (not a nuclear leds_off) avoids clearing the LED-state cache, which
        # would force the open layer's own channel -- e.g. one a step just lit
        # -- to re-fire and blink off then on. led_off self-skips a channel
        # already off, and offing the collapsed layers here still clears a
        # previously-lit channel when the drawer switches to a layer whose LED
        # is off (which the open layer's apply_settings alone would not do).
        for layer in common_utils.get_layers():
            layer_accordion = self.accordion_item_lookup(layer=layer)
            if layer_accordion.collapse:
                try:
                    state = ctx.scope.illumination.get_led_state(color=layer)
                    if state.get('enabled', False):
                        ctx.scope.illumination.led_off_async(layer)
                except Exception as e:
                    logger.warning(
                        f'[LVP Main  ] get_led_state({layer}) failed during '
                        f'accordion-collapse LED cleanup: {e}'
                    )
            else:
                self.layer_lookup(layer=layer).apply_settings()

    def check_settings(self, *args):
        logger.info('[LVP Main  ] ImageSettings.check_settings()')
        lumaview = _app_ctx.ctx.lumaview
        if self.ids['toggle_imagesettings'].state == 'normal':
            self.pos = lumaview.width - self.tab_width, 0
        else:
            self.pos = lumaview.width - self.settings_width, 0


def set_histogram_layer(active_layer):
    for layer in common_utils.get_layers():
        layer_ref = _app_ctx.ctx.image_settings.layer_lookup(layer=layer)
        Clock.unschedule(layer_ref.ids['histo_id'].histogram)

        if layer == active_layer:
            Clock.schedule_interval(layer_ref.ids['histo_id'].histogram, 0.5)
            logger.info(
                f'[LVP Main  ] Clock.schedule_interval(...[{active_layer}]...histogram, 0.5)'
            )
