import sys
import kivy
#kivy.require('1.0.6')

from kivy.app import App
from kivy.core.window import Window
from kivy.factory import Factory
from kivy.graphics import RenderContext
from kivy.properties import StringProperty, ObjectProperty
from kivy.clock import Clock

# User Interface
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.switch import Switch
from kivy.uix.dropdown import DropDown
from kivy.uix.image import Image

# Video Related
from kivy.graphics.texture import Texture
from kivy.uix.camera import Camera
from kivy.core.camera import Camera
import numpy as np
import time
from PIL import Image

comment = '''
Based on code from the kivy example Live Shader Editor found at:
kivy.org/doc/stable/examples/gen__demo__shadereditor__main__py.html
'''

fs_header = '''
#ifdef GL_ES
    precision highp float;
#endif

/* Outputs from the vertex shader */
varying vec4 frag_color;
varying vec2 tex_coord0;

/* uniform texture samplers */
uniform sampler2D texture0;

/* fragment attributes
attribute float red_gain;
attribute float green_gain;
attribute float blue_gain; */

/* custom one */
uniform vec2 resolution;
uniform float time;
uniform vec4 black_point;
uniform vec4 white_point;
'''

vs_header = '''
#ifdef GL_ES
    precision highp float;
#endif

/* Outputs to the fragment shader */
varying vec4 frag_color;
varying vec2 tex_coord0;

/* vertex attributes */
attribute vec2     vPosition;
attribute vec2     vTexCoords0;

/* uniform variables */
uniform mat4       modelview_mat;
uniform mat4       projection_mat;
uniform vec4       color;
'''

global black_point
black_point = (0., )*4

global white_point
white_point = (1., )*4

class ShaderEditor(BoxLayout):

    fs = StringProperty('''
void main (void){
	gl_FragColor = white_point * frag_color * texture2D(texture0, tex_coord0)
	- black_point;
}
''')
    vs = StringProperty('''
void main (void) {
  frag_color = color;
  tex_coord0 = vTexCoords0;
  gl_Position = projection_mat * modelview_mat * vec4(vPosition.xy, 0.0, 1.0);
# }
''')

    viewer = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(ShaderEditor, self).__init__(**kwargs)
        self.test_canvas = RenderContext()
        s = self.test_canvas.shader
        self.trigger_compile = Clock.create_trigger(self.compile_shaders, -1)
        self.bind(fs=self.trigger_compile, vs=self.trigger_compile)

    def compile_shaders(self, *largs):
        print('try compile')
        if not self.viewer:
            print('compile fail')
            return

        # we don't use str() here because it will crash with non-ascii char
        fs = fs_header + self.fs
        vs = vs_header + self.vs

        print('-->', fs)
        self.viewer.fs = fs
        print('-->', vs)
        self.viewer.vs = vs

class ShaderViewer(BoxLayout):
    fs = StringProperty(None)
    vs = StringProperty(None)

    def __init__(self, **kwargs):
        self.canvas = RenderContext()
        super(ShaderViewer, self).__init__(**kwargs)
        Clock.schedule_interval(self.update_shader, 0)

    def update_shader(self, *args):
        #print(black_point)
        c = self.canvas
        c['projection_mat'] = Window.render_context['projection_mat']
        c['time'] = Clock.get_boottime()
        c['resolution'] = list(map(float, self.size))
        c['black_point'] = black_point
        c['white_point'] = white_point
        c.ask_update()

    def on_fs(self, instance, value):
        self.canvas.shader.fs = value

    def on_vs(self, instance, value):
        self.canvas.shader.vs = value

    def capture(self):
        camera = self.ids['microscope_camera']
        #camera = self
        timestr = time.strftime("%Y%m%d_%H%M%S")
        img = camera.export_as_image()
        img.save("capture/IMG_{}.tiff".format(timestr))

Factory.register('ShaderViewer', cls=ShaderViewer)

class ShaderSettings(BoxLayout):
    # get slider values and update global variables where needed
    def get_sliders(self):
        global black_point
        global white_point

        bf_ill = self.ids['bf_led_id'].ids['illumination_id'].value_normalized
        bf_gain = self.ids['bf_led_id'].ids['gain_id'].value_normalized
        bf_exp = self.ids['bf_led_id'].ids['exposure_id'].value

        bl_ill = self.ids['bl_led_id'].ids['illumination_id'].value_normalized
        bl_gain = self.ids['bl_led_id'].ids['gain_id'].value_normalized
        bl_exp = self.ids['bl_led_id'].ids['exposure_id'].value

        gr_ill = self.ids['gr_led_id'].ids['illumination_id'].value_normalized
        gr_gain = self.ids['gr_led_id'].ids['gain_id'].value_normalized
        gr_exp = self.ids['gr_led_id'].ids['exposure_id'].value

        rd_ill = self.ids['rd_led_id'].ids['illumination_id'].value_normalized
        rd_gain = self.ids['rd_led_id'].ids['gain_id'].value_normalized
        rd_exp = self.ids['rd_led_id'].ids['exposure_id'].value


        black_point = (rd_ill, gr_ill, bl_ill, bf_ill)
        white_point = (rd_gain, gr_gain, bl_gain, bf_gain)

class LED_Control(BoxLayout):
    bg_color = ObjectProperty(None)
    ctrl_label = StringProperty(None)

    def __init__(self, **kwargs):
        super(LED_Control, self).__init__(**kwargs)
        if self.bg_color is None:
            self.bg_color = (0.5, 0.5, 0.5, 0.5)
        if self.ctrl_label is None:
            self.ctrl_label = 'Ctrl Label'

class Protocol_Control(BoxLayout):
    bg_color = ObjectProperty(None)
    protocol_label = StringProperty(None)

    def __init__(self, **kwargs):
        super(Protocol_Control, self).__init__(**kwargs)
        if self.bg_color is None:
            self.bg_color = (0.5, 0.5, 0.5, 0.5)
        if self.protocol_label is None:
            self.protocol_label = 'Protocol Label'

# # MainDisplay is organized in lumaviewplus.kv
class MainDisplay(TabbedPanel):
    pass

class ConfigTab(BoxLayout):
    pass

class ImageTab(BoxLayout):
    pass

class MotionTab(BoxLayout):
    pass

class ProtocolTab(BoxLayout):
    pass

class AnalysisTab(BoxLayout):
    pass

class LumaViewProApp(App):
    def build(self):
        # kwargs = {}
        # if len(sys.argv) > 1:
        #     kwargs['source'] = sys.argv[1]

        return MainDisplay()

def update_filter_callback():
    pass

LumaViewProApp().run()
