import kivy

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder

import os

Builder.load_string("""
<MyWidget>:
    id: my_widget
    Button
        text: "open"
        on_release: my_widget.open(filechooser.path, filechooser.selection)
    FileChooserListView:
        id: filechooser
        on_selection: my_widget.selected(filechooser.selection)
""")

class MyWidget(BoxLayout):
    def open(self, path, filename):
        with open(os.path.join(path, filename[0])) as f:
            print ('haterver')

    def selected(self, filename):
        print( "selected: %s" % filename[0])


class MyApp(App):
    def build(self):
        return MyWidget()

if __name__ == '__main__':
    MyApp().run()
