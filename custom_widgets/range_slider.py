# Source code for Range Slider from: https://groups.google.com/g/kivy-users/c/oMFx0YKW5oA/m/r4uj07ENBgAJ

#: -*- encoding: utf-8 -*-
from kivy.lang import Builder
from kivy.metrics import sp
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import (NumericProperty, AliasProperty, OptionProperty,
                             ReferenceListProperty, BoundedNumericProperty, ListProperty)

Builder.load_string('''
<RangeSlider>:
    canvas:
        Color:
            rgb: 1, 1, 1
        BorderImage:
            # border: (0, 18, 0, 18) if self.orientation == 'horizontal' else (18, 0, 18, 0)
            pos: (self.x + self.padding, self.center_y - sp(18)) if self.orientation == 'horizontal' else (self.center_x - 18, self.y + self.padding)
            size: (self.width - self.padding * 2, sp(36)) if self.orientation == 'horizontal' else (sp(36), self.height - self.padding * 2)
            source: 'atlas://data/images/defaulttheme/slider{}_background{}'.format(self.orientation[0], '_disabled' if self.disabled else '')
        Color:
            rgba: self.connector_color
        BorderImage:
            # border: (0, 18, 0, 18) if self.orientation == 'horizontal' else (18, 0, 18, 0)
            pos: (self.value1_pos[0], self.center_y - sp(18)) if self.orientation == 'horizontal' else (self.center_x - sp(18), self.value1_pos[1])
            size: (self.value2_pos[0] - self.value1_pos[0], sp(36)) if self.orientation == 'horizontal' else (sp(36), self.value2_pos[1] - self.value1_pos[1])
            source: 'atlas://data/images/defaulttheme/slider{}_background{}'.format(self.orientation[0], '_disabled' if self.disabled else '')
        Color:
            rgb: 1, 1, 1
        Rectangle:
            pos: (self.value1_pos[0] - sp(16), self.center_y - sp(17)) if self.orientation == 'horizontal' else (self.center_x - sp(16), self.value1_pos[1] - sp(16))
            size: (sp(32), sp(32))
            source: 'atlas://data/images/defaulttheme/slider_cursor{}'.format('_disabled' if self.disabled else '')
        Rectangle:
            pos: (self.value2_pos[0] - sp(16), self.center_y - sp(17)) if self.orientation == 'horizontal' else (self.center_x - sp(16), self.value2_pos[1] - sp(16))
            size: (sp(32), sp(32))
            source: 'atlas://data/images/defaulttheme/slider_cursor{}'.format('_disabled' if self.disabled else '')
''')


class RangeSlider(Widget):
    """Class for creating a RangeSlider widget.

    Check module documentation for more details.
    """
    connector_color = ListProperty([.2, .7, 0.9, 1])
    '''Connector bar color, in the format (r, g, b, a).
    for disabling this bar use a = .0 '''

    def _get_value(self):
        return [self.value1, self.value2]

    def _set_value(self, value):
        self.value1, self.value2 = value

    value = AliasProperty(_get_value, _set_value, bind=('value1', 'value2'))
    '''Current value used for the both sliders.

    :attr:`value` is an :class:`~kivy.properties.AliasProperty` and defaults
    to [0, 0].'''

    value1 = NumericProperty(0.)
    '''Current value used for the first slider.

    :attr:`value` is a :class:`~kivy.properties.NumericProperty` and defaults
    to 0.'''

    value2 = NumericProperty(100.)
    '''Current value used for the second slider.

    :attr:`value` is a :class:`~kivy.properties.NumericProperty` and defaults
    to 0.'''

    min = NumericProperty(0.)
    '''Minimum value allowed for :attr:`value`.

    :attr:`min` is a :class:`~kivy.properties.NumericProperty` and defaults to
    0.'''

    max = NumericProperty(100.)
    '''Maximum value allowed for :attr:`value`.

    :attr:`max` is a :class:`~kivy.properties.NumericProperty` and defaults to
    100.'''

    padding = NumericProperty(sp(16))
    '''Padding of the slider. The padding is used for graphical representation
    and interaction. It prevents the cursor from going out of the bounds of the
    slider bounding box.

    By default, padding is sp(16). The range of the slider is reduced from
    padding \*2 on the screen. It allows drawing the default cursor of sp(32)
    width without having the cursor go out of the widget.

    :attr:`padding` is a :class:`~kivy.properties.NumericProperty` and defaults
    to sp(16).'''

    orientation = OptionProperty('horizontal', options=(
        'vertical', 'horizontal'))
    '''Orientation of the slider.

    :attr:`orientation` is an :class:`~kivy.properties.OptionProperty` and
    defaults to 'horizontal'. Can take a value of 'vertical' or 'horizontal'.
    '''

    range = ReferenceListProperty(min, max)
    '''Range of the slider in the format (minimum value, maximum value)::

        >>> slider = Slider(min=10, max=80)
        >>> slider.range
        [10, 80]
        >>> slider.range = (20, 100)
        >>> slider.min
        20
        >>> slider.max
        100

    :attr:`range` is a :class:`~kivy.properties.ReferenceListProperty` of
    (:attr:`min`, :attr:`max`) properties.
    '''

    step = BoundedNumericProperty(0, min=0)
    '''Step size of the slider.

    .. versionadded:: 1.4.0

    Determines the size of each interval or step the slider takes between
    min and max. If the value range can't be evenly divisible by step the
    last step will be capped by slider.max

    :attr:`step` is a :class:`~kivy.properties.NumericProperty` and defaults
    to 1.'''

    # The following two methods constrain the slider's value
    # to range(min,max). Otherwise it may happen that self.value < self.min
    # at init.

    def on_min(self, *largs):
        self.value1 = min(self.max, max(self.min, self.value1))
        self.value2 = min(self.max, max(self.min, self.value2))

    def on_max(self, *largs):
        self.value1 = min(self.max, max(self.min, self.value1))
        self.value2 = min(self.max, max(self.min, self.value2))

    def get_norm_value1(self):
        vmin = self.min
        d = self.max - vmin
        if d == 0:
            return 0
        return (self.value1 - vmin) / float(d)

    def get_norm_value2(self):
        vmin = self.min
        d = self.max - vmin
        if d == 0:
            return 0
        return (self.value2 - vmin) / float(d)

    def set_norm_value1(self, value):
        vmin = self.min
        step = self.step
        val = value * (self.max - vmin) + vmin
        if step == 0:
            self.value1 = val
        else:
            self.value1 = min(round((val - vmin) / step) * step + vmin,
                              self.max)

    def set_norm_value2(self, value):
        vmin = self.min
        step = self.step
        val = value * (self.max - vmin) + vmin
        if step == 0:
            self.value2 = val
        else:
            self.value2 = min(round((val - vmin) / step) * step + vmin,
                              self.max)

    value1_normalized = AliasProperty(get_norm_value1, set_norm_value1,
                                      bind=('value1', 'min', 'max', 'step'))
    value2_normalized = AliasProperty(get_norm_value2, set_norm_value2,
                                      bind=('value2', 'min', 'max', 'step'))

    '''Normalized value inside the :attr:`range` (min/max) to 0-1 range::

        >>> slider = Slider(value=50, min=0, max=100)
        >>> slider.value
        50
        >>> slider.value_normalized
        0.5
        >>> slider.value = 0
        >>> slider.value_normalized
        0
        >>> slider.value = 100
        >>> slider.value_normalized
        1

    You can also use it for setting the real value without knowing the minimum
    and maximum::

        >>> slider = Slider(min=0, max=200)
        >>> slider.value_normalized = .5
        >>> slider.value
        100
        >>> slider.value_normalized = 1.
        >>> slider.value
        200

    :attr:`value_normalized` is an :class:`~kivy.properties.AliasProperty`.
    '''

    def get_value1_pos(self):
        padding = self.padding
        x = self.x
        y = self.y
        nval = self.value1_normalized
        if self.orientation == 'horizontal':
            return (x + padding + nval * (self.width - 2 * padding), y)
        else:
            return (x, y + padding + nval * (self.height - 2 * padding))

    def get_value2_pos(self):
        padding = self.padding
        x = self.x
        y = self.y
        nval = self.value2_normalized
        if self.orientation == 'horizontal':
            return (x + padding + nval * (self.width - 2 * padding), y)
        else:
            return (x, y + padding + nval * (self.height - 2 * padding))

    def set_value1_pos(self, pos):
        padding = self.padding
        x = min(self.right - padding, max(pos[0], self.x + padding))
        y = min(self.top - padding, max(pos[1], self.y + padding))
        if self.orientation == 'horizontal':
            if self.width == 0:
                self.value1_normalized = 0
            else:
                self.value1_normalized = (x - self.x - padding
                                          ) / float(self.width - 2 * padding)
        else:
            if self.height == 0:
                self.value1_normalized = 0
            else:
                self.value1_normalized = (y - self.y - padding
                                          ) / float(self.height - 2 * padding)

    def set_value2_pos(self, pos):
        padding = self.padding
        x = min(self.right - padding, max(pos[0], self.x + padding))
        y = min(self.top - padding, max(pos[1], self.y + padding))
        if self.orientation == 'horizontal':
            if self.width == 0:
                self.value2_normalized = 0
            else:
                self.value2_normalized = (x - self.x - padding
                                          ) / float(self.width - 2 * padding)
        else:
            if self.height == 0:
                self.value2_normalized = 0
            else:
                self.value2_normalized = (y - self.y - padding
                                          ) / float(self.height - 2 * padding)

    value1_pos = AliasProperty(get_value1_pos, set_value1_pos,
                               bind=('x', 'y', 'width', 'height', 'min',
                                     'max', 'value1_normalized', 'orientation'))
    value2_pos = AliasProperty(get_value2_pos, set_value2_pos,
                               bind=('x', 'y', 'width', 'height', 'min',
                                     'max', 'value2_normalized', 'orientation'))
    '''Position of the internal cursor, based on the normalized value.

    :attr:`value_pos` is an :class:`~kivy.properties.AliasProperty`.
    '''

    def _touch_normalized_value(self, touch):
        pos = touch.pos
        padding = self.padding
        x = min(self.right - padding, max(pos[0], self.x + padding))
        y = min(self.top - padding, max(pos[1], self.y + padding))
        if self.orientation == 'horizontal':
            value = (x - self.x - padding
                     ) / float(self.width - 2 * padding)
        else:
            value = (y - self.y - padding
                     ) / float(self.height - 2 * padding)
        return value

    def on_touch_down(self, touch):
        if self.disabled or not self.collide_point(*touch.pos):
            return
        touch.grab(self)
        t_value = self._touch_normalized_value(touch)
        if abs(self.value1_normalized - t_value) < abs(self.value2_normalized - t_value):
            self.value1_pos = touch.pos
            touch.ud['cursorid'] = 1
        else:
            self.value2_pos = touch.pos
            touch.ud['cursorid'] = 2
        return True

    def on_touch_move(self, touch):
        if touch.grab_current == self:
            if 'cursorid' in touch.ud:
                if touch.ud['cursorid'] == 1:
                    self.value1_pos = touch.pos
                    if self.value1 > self.value2:
                        self.value1_pos = self.value2_pos
                elif touch.ud['cursorid'] == 2:
                    self.value2_pos = touch.pos
                    if self.value2 < self.value1:
                        self.value2_pos = self.value1_pos
                return True

    def on_touch_up(self, touch):
        if touch.grab_current == self:
            touch.ungrab(self)
            return True

    def get_current_range_str(self):
        min_value, max_value = self._get_value()
        return f"{min_value} - {max_value}"
        
