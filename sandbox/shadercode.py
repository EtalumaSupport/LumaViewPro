#!/usr/bin/python3
'''
Minimal Kivy program to test shader code
'''

import kivy
kivy.require('2.0.0')


from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics import RenderContext
from kivy.properties import StringProperty, ObjectProperty, BooleanProperty, NumericProperty
from kivy.factory import Factory
from kivy.core.window import Window

# Video Related
from kivy.graphics.texture import Texture
import cv2
from pypylon import pylon
import random

class PylonCamera(Image):
    record = ObjectProperty(None)
    record = False

    def __init__(self, **kwargs):
        super(PylonCamera,self).__init__(**kwargs)
        self.camera = False;
        self.play = True;
        self.connect()
        self.start()


    def connect(self):
        try:
            # Create an instant camera object with the camera device found first.
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.camera.Open()
            self.camera.Width.SetValue(self.camera.Width.Max)
            self.camera.Height.SetValue(self.camera.Height.Max)
            self.camera.GainAuto.SetValue('Off')
            self.camera.ExposureAuto.SetValue('Off')
            self.camera.ReverseX.SetValue(True);
            # Grabbing Continusely (video) with minimal delay
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            print("LumaViewPro compatible camera or scope is now connected.")

        except:
            if self.camera == False:
                print("It looks like a LumaViewPro compatible camera or scope is not connected.")
                print("Error: PylonCamera.connect() exception")
            self.camera = False

    def start(self):
        self.fps = 10
        self.frame_event = Clock.schedule_interval(self.update, 1.0 / self.fps)

    def stop(self):
        if self.frame_event:
            Clock.unschedule(self.frame_event)

    def update(self, dt):
        if self.camera == False:
            self.connect()
            if self.camera == False:
                self.source = "~/Documents/Earthineering/isim_optim.jpg"
                #self.source = "~/Documents/Earthineering/DaneRecolor_optim.jpg"
                #self.source = "~/Documents/Earthineering/McNamara2_optim.jpg"
                #self.source = "./data/camera to USB.png"
                return
        try:
            if self.camera.IsGrabbing():
                grabResult = self.camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)

                if grabResult.GrabSucceeded():
                    image = grabResult.GetArray()
                    self.array = image
                    image_texture = Texture.create(size=(image.shape[1],image.shape[0]), colorfmt='luminance')
                    image_texture.blit_buffer(image.flatten(), colorfmt='luminance', bufferfmt='ubyte')                    # display image from the texture
                    self.texture = image_texture

                self.lastGrab = pylon.PylonImage()
                self.lastGrab.AttachGrabResultBuffer(grabResult)

            if self.record == True:
                global lumaview
                lumaview.capture(0)

            grabResult.Release()

        except:
            if self.camera == False:
                print("A LumaViewPro compatible camera or scope was disconnected.")
                print("Error: PylonCamera.update() exception")
            self.camera = False

    def frame_size(self, w, h):
        if self.camera == False:
            print("A LumaViewPro compatible camera or scope is not connected.")
            print("Error: PylonCamera.frame_size() self.camera == False")
            return

        width = int(min(int(w), self.camera.Width.Max)/2)*2
        height = int(min(int(h), self.camera.Height.Max)/2)*2
        offset_x = int((self.camera.Width.Max-width)/4)*2
        offset_y = int((self.camera.Height.Max-height)/4)*2

        self.camera.StopGrabbing()
        self.camera.Width.SetValue(width)
        self.camera.Height.SetValue(height)
        self.camera.OffsetX.SetValue(offset_x)
        self.camera.OffsetY.SetValue(offset_y)
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def gain(self, gain):
        if self.camera == False:
            print("A LumaViewPro compatible camera or scope is not connected.")
            print("Error: PylonCamera.gain() self.camera == False")
            return

        self.camera.StopGrabbing()
        self.camera.Gain.SetValue(gain)
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def auto_gain(self, state = True):
        if self.camera == False:
            print("A LumaViewPro compatible camera or scope is not connected.")
            print("Error: PylonCamera.gain() self.camera == False")
            return

        self.camera.StopGrabbing()
        if state == True:
            self.camera.GainAuto.SetValue('Continuous') # 'Off' 'Once' 'Continuous'
        else:
            self.camera.GainAuto.SetValue('Off')

        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def exposure_t(self, t):
        if self.camera == False:
            print("A LumaViewPro compatible camera or scope is not connected.")
            print("Error: PylonCamera.exposure_t() self.camera == False")
            return

        self.camera.StopGrabbing()
        self.camera.ExposureTime.SetValue(t*1000) # (t*1000) in microseconds; therefore t  in milliseconds
        # # DEBUG:
        # print(camera.ExposureTime.Min)
        # print(camera.ExposureTime.Max)
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def auto_exposure_t(self):
        if self.camera == False:
            print("A LumaViewPro compatible camera or scope is not connected.")
            print("Error: PylonCamera.gain() self.camera == False")
            return

        self.camera.StopGrabbing()
        self.camera.ExposureAuto.SetValue('Continuous') # 'Off' 'Once' 'Continuous'
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)


# -----------------------------------------------------------------------------
# Shader code
# Based on code from the kivy example Live Shader Editor found at:
# kivy.org/doc/stable/examples/gen__demo__shadereditor__main__py.html
# -----------------------------------------------------------------------------
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

# global illumination_vals
# illumination_vals = (0., )*4

global gain_vals
gain_vals = (1., )*4

class ShaderViewer(BoxLayout):
    fs = StringProperty('''
void main (void) {
	gl_FragColor =
    white_point *
    frag_color *
    texture2D(texture0, tex_coord0)
    - black_point;
}
''')
    vs = StringProperty('''
void main (void) {
  frag_color = color;
  tex_coord0 = vTexCoords0;
  gl_Position =
  projection_mat *
  modelview_mat *
  vec4(vPosition.xy, 0.0, 1.0);
}
''')


    def __init__(self, **kwargs):
        print("Shader viwer initializing")
        self.canvas = RenderContext()
        super(ShaderViewer, self).__init__(**kwargs)
        self.canvas.shader.fs = fs_header + self.fs
        self.canvas.shader.vs = vs_header + self.vs
        Clock.schedule_interval(self.update_shader, 0)

    def on_touch_down(self, touch):
        # Override Scatter's `on_touch_down` behavior for mouse scroll
        if touch.is_mouse_scrolling:
            if touch.button == 'scrolldown':
                if self.scale < 100:
                    self.scale = self.scale * 1.1
            elif touch.button == 'scrollup':
                if self.scale > 0.1:
                    self.scale = self.scale * 0.8
        # If some other kind of "touch": Fall back on Scatter's behavior
        else:
            super(ShaderViewer, self).on_touch_down(touch)

    def update_shader(self, *args):
        global gain_vals

        c = self.canvas
        c['projection_mat'] = Window.render_context['projection_mat']
        c['time'] = Clock.get_boottime()
        c['resolution'] = list(map(float, self.size))
        c['black_point'] = (0., )*4
        # adjust for false color
        c['white_point'] = gain_vals

    def on_fs(self, instance, value):
        self.canvas.shader.fs = value

    def on_vs(self, instance, value):
        self.canvas.shader.vs = value

Factory.register('ShaderViewer', cls=ShaderViewer)

class ShaderEditor(BoxLayout):
    fs = StringProperty('''
void main (void){
	gl_FragColor =
    white_point *
    frag_color *
    texture2D(texture0, tex_coord0)
    - black_point;
}
''')
    vs = StringProperty('''
void main (void) {
  frag_color = color;
  tex_coord0 = vTexCoords0;
  gl_Position =
  projection_mat *
  modelview_mat *
  vec4(vPosition.xy, 0.0, 1.0);
}
''')

    viewer = ObjectProperty(None)
    hide_editor = ObjectProperty(None)
    hide_editor = True


    def __init__(self, **kwargs):
        print("Shader Editor initializing")
        super(ShaderEditor, self).__init__(**kwargs)
        self.test_canvas = RenderContext()
        s = self.test_canvas.shader
        self.trigger_compile = Clock.create_trigger(self.compile_shaders, -1)
        self.bind(fs=self.trigger_compile, vs=self.trigger_compile)

    def compile_shaders(self, *largs):
        print('ShaderEditor.compile_shaders()')
        if not self.viewer:
            print('ShaderEditor.compile_shaders() Fail')
            return

        # we don't use str() here because it will crash with non-ascii char
        fs = fs_header + self.fs
        vs = vs_header + self.vs

        print('-->', fs)
        self.viewer.fs = fs
        print('-->', vs)
        self.viewer.vs = vs

    # Hide (and unhide) Shader settings
    def toggle_editor(self):
        # update protocol
        if self.hide_editor == False:
            self.hide_editor = True
            self.pos = -285, 0
        else:
            self.hide_editor = False
            self.pos = 0, 0

# -------------------------------------------------------------------------
# RUN LUMAVIEWPRO APP
# -------------------------------------------------------------------------
class Main(BoxLayout):
    pass

class ShaderCodeApp(App):
    def build(self):
        print("About to run main")
        return Main()

ShaderCodeApp().run()
