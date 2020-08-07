from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Rectangle
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image

class Imglayout(FloatLayout):

    def __init__(self,**args):
        super(Imglayout,self).__init__(**args)

        with self.canvas.before:
            Color(0,0,0,0)
            self.rect=Rectangle(size=self.size,pos=self.pos)

        self.bind(size=self.updates,pos=self.updates)
    def updates(self,instance,value):
        self.rect.size=instance.size
        self.rect.pos=instance.pos


class MainTApp(App):

    im=Image(source='../data/sample.tif')
    def build(self):
        root = BoxLayout(orientation='vertical')
        c = Imglayout()
        root.add_widget(c)


        self.im.keep_ratio= False
        self.im.allow_stretch = True
        cat=Button(text="Categories",size_hint=(1,.07))
        cat.bind(on_press=self.callback)
        c.add_widget(self.im)
        root.add_widget(cat);
        return root

    def callback(self, instance, value):
        self.im.source = '../data/etaluma.png'

if __name__ == '__main__':
    MainTApp().run()
