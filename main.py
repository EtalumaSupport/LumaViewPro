import sys
import kivy
kivy.require('1.0.6')

from kivy.app import App
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.uix.floatlayout import FloatLayout
from kivy.core.window import Window
from kivy.factory import Factory
from kivy.graphics import RenderContext
from kivy.properties import StringProperty, ObjectProperty
from kivy.clock import Clock
from kivy.compat import PY2

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

/* custom one */
uniform vec2 resolution;
uniform float time;
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



# MainDisplay is organized in lumaviewplus.kv
class MainDisplay(TabbedPanel):
    pass

class LumaViewPlusApp(App):
    def build(self):
        return MainDisplay()

if __name__ == '__main__':
    LumaViewPlusApp().run()
