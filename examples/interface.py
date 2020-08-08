import kivy
from kivy.app import App

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.button import Button

from kivy.properties import ObjectProperty
from kivy.properties import NumericProperty


class AppScreen(FloatLayout):
    app = ObjectProperty(None)

class MainMenu(AppScreen):
    pass

class ScoreScreen(AppScreen):
    count_r = NumericProperty(0)
    score = NumericProperty(0)

    # This is making a new WordComprehension object, and thus will not have
    # the correct value of score. We no longer need this is we are binding properties
    # together.
    #def get_score(self):
    #    wordcomp = WordComprehension()
    #    self.score = wordcomp.count_r


class WordComprehension(AppScreen):
    count_r = NumericProperty(0)
    count_w = NumericProperty(0)

    def do_something(self):
        self.count_r += 1


class InterfaceApp(App):
    def build(self):
        self.screens = {}
        self.screens["wordcomp"] = WordComprehension(app=self)
        self.screens["menu"] = MainMenu(app=self)
        self.screens["score"] = ScoreScreen(app=self)
        self.root = FloatLayout()
        self.goto_screen("menu")

        # Bind the two properties together. Whenever count_r changes in the wordcomp
        # screen, the score property in the score screen will reflect those changes.
        self.screens["wordcomp"].bind(count_r=self.screens["score"].setter('score'))

        return self.root

    def goto_screen(self, screen_name):
        self.root.clear_widgets()
        self.root.add_widget(self.screens[screen_name])


if __name__ == "__main__":
    InterfaceApp().run()
