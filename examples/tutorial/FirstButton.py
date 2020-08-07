from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from functools import partial

class FirstButton(App):
    #custom method to disable button
    def disable(self, instance, *args):
        instance.disabled = True

    #custom method to update text of button
    def update(self, instance, *args):
        instance.text = "I am Disabled"

    # runs automatically when the button is created
    def build(self):
        myBtn = Button(text="Click Me", pos=(100,350), size_hint = (.1, .1))
        myBtn.bind(on_press=partial(self.disable, myBtn))
        myBtn.bind(on_press=partial(self.update, myBtn))


        return myBtn

FirstButton().run()
