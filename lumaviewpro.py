import sys
import kivy
#kivy.require('1.0.6')

from kivy.app import App
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.core.window import Window
from kivy.factory import Factory
from kivy.graphics import RenderContext
from kivy.properties import StringProperty, ObjectProperty
from kivy.clock import Clock

# new...
from kivy.uix.widget import Widget
from kivy.uix.togglebutton import ToggleButton

from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.uix.camera import Camera
from kivy.core.camera import Camera
import numpy as np
import time

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

class ShaderViewer(BoxLayout):
    fs = StringProperty(None)
    vs = StringProperty(None)

    def __init__(self, **kwargs):
        self.canvas = RenderContext()
        super(ShaderViewer, self).__init__(**kwargs)
        Clock.schedule_interval(self.update_shader, 0)

    def update_shader(self, *args):
        c = self.canvas
        c['projection_mat'] = Window.render_context['projection_mat']
        c['time'] = Clock.get_boottime()
        c['resolution'] = list(map(float, self.size))
        c['black_point'] = black_point
        c.ask_update()

    def on_fs(self, instance, value):
        self.canvas.shader.fs = value

    def on_vs(self, instance, value):
        self.canvas.shader.vs = value


Factory.register('ShaderViewer', cls=ShaderViewer)


class ShaderEditor(BoxLayout):

    #source = StringProperty('data/sample.tif')

    fs = StringProperty('''
void main (void){
	gl_FragColor = frag_color * texture2D(texture0, tex_coord0)
	+ black_point;
}
''')
    vs = StringProperty('''
void main (void) {
  frag_color = color;
  tex_coord0 = vTexCoords0;
  gl_Position = projection_mat * modelview_mat * vec4(vPosition.xy, 0.0, 1.0);
}
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

    def capture(self):
        camera = self.ids['scope']
        timestr = time.strftime("%Y%m%d_%H%M%S")
        camera.export_to_png("capture/IMG_{}.png".format(timestr))


    # get slider values and update global variables where needed
    def get_sliders(self):
        global black_point

        bf_ill = self.ids['bf_ill']
        bf_gain = self.ids['bf_gain']
        bf_exp = self.ids['bf_exp']

        bl_ill = self.ids['bl_ill']
        bl_gain = self.ids['bl_gain']
        bl_exp = self.ids['bl_exp']

        gr_ill = self.ids['gr_ill']
        gr_gain = self.ids['gr_gain']
        gr_exp = self.ids['gr_exp']

        rd_ill = self.ids['rd_ill']
        rd_gain = self.ids['rd_gain']
        rd_exp = self.ids['rd_exp']

        slider_vals = np.array([[bf_ill.value, bf_gain.value, bf_exp.value],
                            [bl_ill.value, bl_gain.value, bl_exp.value],
                            [gr_ill.value, gr_gain.value, gr_exp.value],
                            [rd_ill.value, rd_gain.value, rd_exp.value]])
        black_point = (rd_ill.value_normalized, gr_ill.value_normalized, bl_ill.value_normalized, bf_ill.value_normalized)
    #    print('Black Point:\n', black_point)
    #    print('Slider Values:\n', slider_vals)

# MainDisplay is organized in lumaviewplus.kv
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
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['source'] = sys.argv[1]
        #else:
        #    kwargs['source'] = 'data/sample.tif'
        #return ShaderEditor(**kwargs)
        return MainDisplay()

    # def on_stop(self):
    #     #without this, app will not exit even if the window is closed
    #     self.capture.release()

def update_filter_callback():
    pass

LumaViewProApp().run()
