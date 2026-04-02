# Copyright Etaluma, Inc.
import logging

from kivy.clock import Clock
from kivy.metrics import dp
from kivy.uix.accordion import AccordionItem
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView

import modules.app_context as _app_ctx
import modules.common_utils as common_utils

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
# ImageSettings — Right Sidebar Panel (Channel Controls, LED, Exposure)
# ============================================================================

class ImageSettings(BoxLayout):
    settings_width = dp(300)
    tab_width = dp(30)

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
        self._init_ui_retries = 0
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
            return self.ids[f"{layer}_accordion"]


    def set_expanded_layer(self, layer: str, *largs) -> None:
        """
        Expand the specified layer accordion and collapse all others.
        Cleans up ScrollView viewport textures on collapse to prevent memory accumulation.
        Accordion toggling is always disabled during protocol execution to prevent memory leaks.
        """

        # Skip accordion toggling during protocol execution to prevent memory leaks
        if _app_ctx.ctx.protocol_running.is_set():
            return

        for a_layer in common_utils.get_layers():
            accordion_item_obj = self.accordion_item_lookup(layer=a_layer)

            # Check if we need to collapse this accordion
            target_collapsed = (layer != a_layer)

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
                            Clock.schedule_once(lambda dt, sv=child: cleanup_scrollview_viewport(sv), 0.2)

                accordion_item_obj.collapse = True


    def set_lumi_layer_control_visibility(self, visible: bool) -> None:
        if visible:
            self._show_lumi_layer_control()
        else:
            self._hide_lumi_layer_control()


    def _show_lumi_layer_control(self):
        if not self._accordion_item_lumi_control_visible:
            self._accordion_item_lumi_control_visible = True
            self.ids['accordion_id'].add_widget(self._accordion_item_lumi_control, 0)


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


    def _hide_df_layer_control(self):
        settings = _app_ctx.ctx.settings
        if settings:
            settings['DF']['acquire'] = None
        if self._accordion_item_df_control_visible:
            self._accordion_item_df_control.collapse = True
            self._accordion_item_df_control_visible = False
            self.ids['accordion_id'].remove_widget(self._accordion_item_df_control)


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


    def _init_ui(self, dt=0):
        ctx = _app_ctx.ctx
        if ctx is None:
            self._init_ui_retries += 1
            if self._init_ui_retries > 50:
                logger.error('[LVP Main  ] ImageSettings._init_ui: ctx still None after 50 retries, giving up')
                return
            Clock.schedule_once(self._init_ui, 0.1)
            return
        self.assign_led_button_down_images()
        # Skip accordion_collapse during app initialization to prevent premature apply_settings
        if not ctx.initializing:
            self.accordion_collapse()
        self.set_layer_exposure_ranges()
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
            layer_obj.ids['exp_slider'].min = 1.0   # 1ms floor — sub-ms never realistic for fluorescence
            layer_obj.ids['exp_slider'].max = ctx.max_exposure
            layer_obj.ids['exp_slider'].step = 1.0   # Integer steps only

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
            layer_obj.ids['exp_slider'].min = 1.0   # 1ms floor
            layer_obj.ids['exp_slider'].max = ctx.max_exposure
            layer_obj.ids['exp_slider'].step = 1.0   # Integer steps only


    def assign_led_button_down_images(self):
        led_button_down_background_map = {
            'Red': './data/icons/ToggleRR.png',
            'Green': './data/icons/ToggleRG.png',
            'Blue': './data/icons/ToggleRB.png',
            'Lumi': './data/icons/ToggleRB.png',
        }

        for layer in common_utils.get_layers_with_led():
            button_down_image = led_button_down_background_map.get(layer, './data/icons/ToggleRW.png')
            layer_obj = self.layer_lookup(layer=layer)
            layer_obj.ids['enable_led_btn'].background_down = button_down_image


    # Hide (and unhide) main settings
    def toggle_settings(self):
        if not _app_ctx.ctx.protocol_running.is_set():
            self.update_transmitted()
        logger.info('[LVP Main  ] ImageSettings.toggle_settings()')
        ctx = _app_ctx.ctx
        lumaview = ctx.lumaview
        scope_display = ctx.scope_display

        # scope_display.stop()

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
            label.text = ""
            label.visible = False
            label.opacity = 0
            slider.disabled = True
            slider.visible = False
            slider.cursor_size = '0dp','0dp'
            slider.opacity = 0
            slider.value_track_color = (0., )*4
            text.disabled = True
            text.visible = False
            text.width = '0dp'
            text.text = ""
            text.opacity = 0
            layer_obj.ids['false_color_label'].text = ''
            layer_obj.ids['false_color'].color = (0., )*4

            # Adjust 'Illumination' range
            layer_obj.ids['ill_slider'].step = 1
            layer_obj.ids['ill_slider'].max = 50

    def accordion_collapse(self):
        logger.info('[LVP Main  ] ImageSettings.accordion_collapse()')
        from ui.ui_helpers import scope_leds_off
        ctx = _app_ctx.ctx

        # Skip during app initialization - will be called explicitly after init completes
        if ctx.initializing:
            return

        # Don't turn off LEDs if "Protocol LEDs On" is active and we're
        # stepping through a protocol. The accordion collapse fires when
        # go_to_step_update_ui opens the step's channel — this would kill
        # the LED that step_navigation just turned on. (fixes #605)
        if ctx.settings.get('protocol_led_on', False):
            return

        # turn off the camera update and all LEDs
        scope_display = ctx.scope_display
        # scope_display.stop()
        scope_leds_off()

        # turn off all LED toggle buttons and histograms
        for layer in common_utils.get_layers():
            layer_accordion = self.accordion_item_lookup(layer=layer)
            layer_is_collapsed = layer_accordion.collapse

            if layer_is_collapsed:
                continue

            layer_obj = self.layer_lookup(layer=layer)
            layer_obj.apply_settings()

        # Restart camera feed
        # if scope_display.play == True:
        #     scope_display.start()


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
            logger.info(f'[LVP Main  ] Clock.schedule_interval(...[{active_layer}]...histogram, 0.5)')
