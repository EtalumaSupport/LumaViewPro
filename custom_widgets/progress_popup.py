
import threading

from kivy.uix.popup import Popup
from kivy.lang import Builder


def show_popup(function):
    def wrap(app, *args, **kwargs):
        popup = CustomPopup()  # Instantiate CustomPopup (could add some kwargs if you wish)
        app.done = False  # Reset the app.done BooleanProperty
        app.bind(done=popup.dismiss)  # When app.done is set to True, then popup.dismiss is fired
        popup.open()  # Show popup
        t = threading.Thread(target=function, args=[app, popup, *args], kwargs=kwargs)  # Create thread
        t.start()  # Start thread
        return t

    return wrap


class CustomPopup(Popup):
    pass


kv = Builder.load_string(
"""
<CustomPopup>:
    size_hint: .6, .3
    auto_dismiss: False
    progress: 0
    text: ''
    title: ''
    
    BoxLayout:
        orientation: 'vertical'
        
        Label:
            text: root.text
            size_hint: 1, 0.8
            
        ProgressBar:
            value: root.progress
"""
)
