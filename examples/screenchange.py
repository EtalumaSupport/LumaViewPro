import kivy
import os

kivy.require('1.10.0')

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.image import Image
from kivy.properties import StringProperty
from kivy.clock import Clock
from kivy.uix.widget import Widget
from sshtest import ssh #import the ssh class from the ssh python file


class ScreenOne(Screen):
    def login(self): #define login using ssh as a function

        #Make a Class Variable called connection. This allows for other
        #classes to call on it without passing it as an argument

        ScreenOne.connection = ssh("192.168.1.3", "pi", "seniordesign")
                #ScreenOne.connection.sendCommand("ls")
        #ScreenOne.connection.sendCommand("mkdir thisistest")
        print("Logging in") #For error checking

    def gpioSet(self): #allows for gpio pins to trigger image capture
        ScreenOne.connection.sendCommand("echo '18' > /sys/class/gpio/export")
        ScreenOne.connection.sendCommand("echo 'out' > /sys/class/gpio/gpio18/direction")

class ScreenTwo(Screen): #This is where all the functions are
    img_src = StringProperty('../data/sample.tif')
    def command(self, input): #create a function that sends command through ssh
        ScreenOne.connection.sendCommand(input) #Call on connection made before to send command


class MyScreenManager(ScreenManager):
    pass


#Create the Application that will change screens. Add App(App) to end of wanted classname
class ScreenChangeApp(App):

#Create a function that builds the app. Keyword self make sures to reference
#current instantiation

    def build(self):
        screen_manager = ScreenManager()

        screen_manager.add_widget(ScreenOne(name = "screen_one"))
        screen_manager.add_widget(ScreenTwo(name = "screen_two"))

        return screen_manager #after building the screens, return them
#MyScreenManager.login()

sample_app = ScreenChangeApp()
sample_app.run()
