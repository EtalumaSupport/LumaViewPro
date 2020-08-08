import sys
import kivy
kivy.require('1.0.6')

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

import time

'''
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
uniform vec4 color;
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


class ShaderViewer(BoxLayout):
    fs = StringProperty(None)
    vs = StringProperty(None)

    def __init__(self, **kwargs):
        self.canvas = RenderContext()
        super(ShaderViewer, self).__init__(**kwargs)
        Clock.schedule_interval(self.update_shader, 0)

    def update_shader(self, *args):
        canvas = self.canvas
        canvas['projection_mat'] = Window.render_context['projection_mat']
        canvas['time'] = Clock.get_boottime()
        canvas['resolution'] = list(map(float, self.size))
        canvas['color'] = (0.8,0.1,0.1,0.1)
        canvas.ask_update()

    def on_fs(self, instance, value):
        self.canvas.shader.fs = value

    def on_vs(self, instance, value):
        self.canvas.shader.vs = value


Factory.register('ShaderViewer', cls=ShaderViewer)


class ShaderEditor(BoxLayout):

    source = StringProperty('data/sample.tif')

    fs = StringProperty('''
void main (void){
	vec4 
	gl_FragColor = frag_color * texture2D(texture0, tex_coord0);
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
        camera.export_to_png("IMG_{}.png".format(timestr))



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

    def on_stop(self):
        #without this, app will not exit even if the window is closed
        self.capture.release()

LumaViewProApp().run()
