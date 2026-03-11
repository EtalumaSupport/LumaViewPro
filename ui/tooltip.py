# Copyright Etaluma, Inc.
"""
Tooltip system extracted from lumaviewpro.py.

Contains the Tooltip widget class and TooltipMixin which provides
tooltip management methods for LumaViewProApp.
"""

import kivy.uix.accordion
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, InstructionGroup, Rectangle, Line
from kivy.uix.accordion import AccordionItem
from kivy.uix.label import Label

import modules.app_context as _app_ctx


class Tooltip(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.horiz_padding = 4      #4
        self.vert_padding = 4       #4

        self.opacity = 0
        self.font_size = '15sp'
        self.color = [0, 0, 0, 1]  # Black text
        self.bind(size=self._update_rect, pos=self._update_rect)
        with self.canvas.before:
            Color(1, 1, 1, 1)  # White background
            self.rect = Rectangle(size=(self.texture_size[0] + self.horiz_padding, self.texture_size[1] + self.vert_padding))

        self.opacity = 0  # Initially hidden

    def _update_rect(self, *args):
        self.rect.size = (self.texture_size[0] + self.horiz_padding, self.texture_size[1] + self.vert_padding)


class TooltipMixin:
    """Mixin providing tooltip management methods for LumaViewProApp.

    Expects the host class to have these attributes set during build():
        self.hidden, self.tooltip_attr_widgets, self.widget_to_accordion_dict,
        self.tt_widget, self.widget_being_described, self.tt_shown,
        self.tt_clock_event, self.mouse_pos
    """

    def init_tooltips(self, root_widget):
        """Initialize tooltip system. Call from build() after widget tree exists."""
        self.hidden = True
        self.tooltip_attr_widgets = self.find_widgets_with_tooltips(root_widget)
        self.widget_to_accordion_dict = self.create_widget_to_parent_dict(self.tooltip_attr_widgets)
        self.tt_widget = Tooltip()
        self.widget_being_described = None
        Window.bind(mouse_pos=self.mouse_moved)
        self.tt_shown = False
        self.tt_clock_event = None

    # Returns a list of widgets with tooltip_text attribute
    def find_widgets_with_tooltips(self, widget) -> list:
        widgets = []

        children = widget.children
        if hasattr(widget, 'tooltip_text'):
            if widget.tooltip_text != "":
                widgets.append(widget)
                return widgets
        for child in children:
            widgets += self.find_widgets_with_tooltips(child)
        return widgets

    # Helper function to find a widget's Accordion
    # Returns a list of all parents that are accordions. As list increments, accordions approach head of widget tree
    def find_accordion_parents(self, widget) -> list:
        return_list = []
        if widget.parent is None:
            return return_list
        if isinstance(widget.parent, kivy.uix.accordion.AccordionItem) or isinstance(widget.parent, AccordionItem):
            return return_list + [widget.parent] + self.find_accordion_parents(widget.parent)
        else:
            return self.find_accordion_parents(widget.parent)

    # Creates a dictionary to relate a widget to the Accordion(s) it is in
    def create_widget_to_parent_dict(self, tt_attr_widgets) -> dict:
        dict = {}
        for widget in tt_attr_widgets:
            dict[widget] = self.find_accordion_parents(widget)
        return dict


    # Called every time mouse is moved
    # Used to check if tooltip should be shown
    def mouse_moved(self, *args) -> None:
        ctx = _app_ctx.ctx

        delay_until_tooltip = 0.5   # In Seconds

        mouse_pos = args[1]
        self.mouse_pos = mouse_pos
        on_widget = False

        if ctx.show_tooltips:
            self.hidden = False

            # Hide tooltip on mouse movement if not colliding anymore (Put here to check asap after a change)
            if self.widget_being_described is not None:
                if not self.tt_collision(self.widget_being_described, mouse_pos[0], mouse_pos[1]):
                    self.hide_tooltip()
                if self.tt_clock_event is not None:
                    Clock.unschedule(self.tt_clock_event)
                    self.tt_clock_event = None

            # Checks collision and if tooltip is visible. If it isn't on any tooltip, hide the tooltip
            for widget in self.tooltip_attr_widgets:

                if widget.pos[0] > -100 and widget.pos[0] < Window.width and widget.pos[1] > 0 and widget.pos[1] < Window.height:

                    collision = self.tt_collision(widget, mouse_pos[0], mouse_pos[1])

                    if collision:
                        accordion_parents = self.widget_to_accordion_dict[widget]
                        self.widget_being_described = widget

                        # If widget is not in an Accordion, it is always visible, so show tooltip
                        if len(accordion_parents) < 1:

                            on_widget = True
                            if not self.tt_shown:
                                self.tt_widget.text = widget.tooltip_text
                                self.tt_clock_event = Clock.schedule_once(self.show_tooltip, delay_until_tooltip)
                            break

                        # If all accordions above the widget are not collapsed, show the widget
                        elif True not in [accordion.collapse for accordion in accordion_parents]:
                            on_widget = True
                            if not self.tt_shown:
                                self.tt_widget.text = widget.tooltip_text
                                self.tt_clock_event = Clock.schedule_once(self.show_tooltip, delay_until_tooltip)
                            break
                        else:
                            continue

                    else:
                        on_widget = False
                else:
                    on_widget = False

            if not on_widget:
                if self.tt_clock_event:
                    Clock.unschedule(self.tt_clock_event)
                    self.tt_clock_event = None

                self.hide_tooltip()
        else:
            # Hides tooltip one time if tooltips are turned off (else always remains on screen)
            if not self.hidden:
                self.hide_tooltip()
                if self.tt_clock_event is not None:
                    Clock.unschedule(self.tt_clock_event)
                    self.tt_clock_event = None
                self.hidden = True


    def tt_collision(self, widget, mouse_x: float, mouse_y: float) -> bool:
        # Shows hitboxes for tooltips.
        # Only seems to work for widgets not in channel control for some reason
        show_hitboxes = False

        true_widget_x = widget.to_window(*widget.pos)[0]
        true_widget_y = widget.to_window(*widget.pos)[1]

        if type(widget) is not Label:
            left = true_widget_x
            right = true_widget_x + widget.width
            bottom = true_widget_y
            top = true_widget_y + widget.height

            if show_hitboxes:
                grp = getattr(widget, '_hitbox_group', None)
                if grp is None:
                    grp = InstructionGroup()
                    widget._hitbox_group = grp
                    widget.canvas.after.add(grp)
                grp.clear()
                grp.add(Color(1,0,0,1))
                grp.add(Line(rectangle=(left, bottom, right-left, top-bottom)))

            return left <= mouse_x <= right and bottom <= mouse_y <= top

        else:
            # Widget is a Label
            # Hitbox is only on the text portion of the label, unless wrapping is present

            text_width = widget.texture_size[0]
            text_height = widget.texture_size[1]
            total_width = widget.width
            total_height = widget.height

            if text_width == total_width and text_height == total_height:
                text_width, text_height = self.calculate_label_text_size(widget)

            # Setting text_x and text_y to represent the bottom left corner of the label text

            if widget.halign == "left":
                text_x = true_widget_x
            elif widget.halign == "right":
                text_x = true_widget_x + (total_width - text_width)
            else:
                text_x = ((total_width - text_width) / 2) + true_widget_x

            if widget.valign == "top":
                text_y = true_widget_y + (total_height - text_height)
            else:
                text_y = ((total_height - text_height) / 2) + true_widget_y

            if show_hitboxes:
                grp = getattr(widget, '_hitbox_group', None)
                if grp is None:
                    grp = InstructionGroup()
                    widget._hitbox_group = grp
                    widget.canvas.after.add(grp)
                grp.clear()
                grp.add(Color(1, 0, 0, 1))
                grp.add(Line(rectangle=(text_x, text_y, text_width, text_height), width=1))

            return text_x <= mouse_x <= (text_x + text_width) and text_y <= mouse_y <= (text_y + text_height)

    _text_size_cache = {}  # {(text, font_size, max_width): (w, h)}

    # Used to calculate a label's text dimensions when the label size is preset (keeps collision only on the text)
    def calculate_label_text_size(self, widget) -> tuple:
        text = widget.text
        font_size = widget.font_size
        max_width = widget.size[0]

        cache_key = (text, font_size, max_width)
        cached = self._text_size_cache.get(cache_key)
        if cached is not None:
            return cached

        temp_label = Label(text=text, font_size=font_size,)
        temp_label.texture_update()
        text_width, text_height = temp_label.texture_size

        if text_width > max_width:
            temp_label.text_size[0] = max_width
            temp_label.texture_update()
            text_width, text_height = temp_label.texture_size

        self._text_size_cache[cache_key] = (text_width, text_height)
        return text_width, text_height


    def show_tooltip(self, *args) -> None:
        ctx = _app_ctx.ctx

        if ctx.show_tooltips:
            if self.widget_being_described is not None:
                self.tt_widget._update_rect()
                # Default offsets
                vert_offset = 15
                horiz_offset = 15

                # If mouse is low on the screen
                low_screen_vert_offset = 7

                # If mouse is far right on the screen
                right_screen_horiz_offset = 7

                # If mouse is in lower quarter of screen, show tooltip above mouse instead of below
                if self.mouse_pos[1] < Window.height / 4:
                    lower_half = True
                else:
                    lower_half = False

                if self.mouse_pos[0] > Window.width - Window.width / 4:
                    far_right = True
                else:
                    far_right = False

                if not self.tt_shown:

                    # Remove and add the widget to ensure it shows up at the front of the screen
                    ctx.lumaview.remove_widget(self.tt_widget)
                    ctx.lumaview.add_widget(self.tt_widget)
                    self.tt_widget.size = Window.size

                    if lower_half:
                        tt_widget_y = self.mouse_pos[1] - self.tt_widget.height + low_screen_vert_offset + (Window.height / 2)
                        tt_widget_rect_y = self.mouse_pos[1] + low_screen_vert_offset/2 + (self.tt_widget.vert_padding / 2) - self.tt_widget.texture_size[1]/2 - self.tt_widget.vert_padding/2 + 1
                    else:
                        # Upper Half
                        tt_widget_y = self.mouse_pos[1] - self.tt_widget.height - vert_offset + (Window.height / 2)
                        tt_widget_rect_y = self.mouse_pos[1] - vert_offset/2 + (self.tt_widget.vert_padding / 2) - self.tt_widget.rect.size[1] - 2*self.tt_widget.vert_padding + self.tt_widget.texture_size[1]/2

                    if far_right:
                        tt_widget_x = self.mouse_pos[0] - right_screen_horiz_offset - (Window.width / 2) - (self.tt_widget.texture_size[0]/2)
                        tt_widget_rect_x = self.mouse_pos[0] - right_screen_horiz_offset - (self.tt_widget.horiz_padding / 2) - (self.tt_widget.texture_size[0])
                    else:
                        # Left Side
                        tt_widget_x = self.mouse_pos[0] + horiz_offset - (Window.width / 2) + (self.tt_widget.texture_size[0]/2)
                        tt_widget_rect_x = self.mouse_pos[0] + horiz_offset - (self.tt_widget.horiz_padding / 2)

                    self.tt_widget.pos = (tt_widget_x, tt_widget_y)
                    self.tt_widget.rect.pos = (tt_widget_rect_x, tt_widget_rect_y)

                    self.tt_widget.opacity = 1
                    self.tt_shown = True

    def hide_tooltip(self, *args) -> None:
        self.widget_being_described = None
        if self.tt_shown:
            self.tt_widget.opacity = 0
            self.tt_shown = False
