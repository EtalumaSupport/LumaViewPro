from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder

Builder.load_string("""

<KivyButton>:
    Button:
        text: "Hello Button!"
        size_hint: .1, .425
        Image:
            source: 'images.jpg'
            center_x: self.parent.center_x
            center_y: self.parent.center_y

""")

class KivyButton(App, BoxLayout):
    def build(self):
        return self

KivyButton().run()
