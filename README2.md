# LumaViewPro (pre-release) 
LumaViewPro is an open source fluorescence microscope control interface written in Python.  It is sponsored by Etaluma, Inc. and includes an example of an interface to the Etaluma LED control board with Multi-axis motion control. It offers multi OS functional GUI as well as a development tool for testing and building applications from.

![LVProScreenshotSept21](https://user-images.githubusercontent.com/74261093/132131289-ce4dce0b-3fcc-4d69-8dba-0862944329d9.png)

## Instalation Guide For Windows to run as Python Script 
Download and install from the python page https://www.python.org/downloads/.
- select "add python to PATH"
- select "install launcher for all users"
- select "install now"
- restart the computer

<img src="https://user-images.githubusercontent.com/108957480/178378391-fffaa372-6472-4022-88e4-571670b97fcd.png" width="800" height="450">

Download and unzip the LumaViewPro-master file 

Select to open all Python documents using the Python application 

<img src="https://user-images.githubusercontent.com/108957480/178375526-ada1cade-14ee-4f9a-8695-8b55f698c4b0.png" width="800" height="400">

Open a comand Prompt and enter the following 
```
python -m pip install --upgrade pip setuptools virtualenv
python -m pip install numpy
python -m pip install pyserial
python -m pip install Pillow
python -m pip install opencv-python
python -m pip install pypylon
python -m pip install docutils pygments pypiwin32
python -m pip install kivy.deps.sdl2
python -m pip install kivy.deps.glew
python -m pip install kivy
python -m pip install plyer
python -m pip install scipy
python -m pip install mcp2210-python
python -m pip install pandas 
```
Go to the location of the LumaViewPro-master folder in the command prompt

Open LumaViewPro using the command prompt 

<img width="700" alt="Screen Shot 2022-07-11 at 4 47 24 PM" src="https://user-images.githubusercontent.com/108957480/178377130-e620e8a4-de45-4f6b-9031-8c1b18192164.png">

If having trouble with camera install the basler pylon driver at https://www.baslerweb.com/en/downloads/software-downloads/


## Installation Guide for Linux to run as Python Script

Download and unzip the LumaViewPro-master file 

To install Python open the terminal and enter the following 

``
sudo apt-get update 
``

``
sudo apt-get install python3.10
``

``
sudo apt-get install python3-pip        
``

To check if Python works enter the following in the terminal

``
python3 --version
``

![Screenshot from 2022-07-12 18-49-14](https://user-images.githubusercontent.com/108957480/178634740-8e8f300d-d2b3-41c8-9e8b-c0db899af022.png)

Enter the following into the terminal to install the package dependencies

``
pip3 install numpy plyer kivy kivymd opencv-python scipy pyserial pypylon mcp2210-python pandas
``

![Screenshot from 2022-07-12 18-48-26 (1)](https://user-images.githubusercontent.com/108957480/178635382-b43c80f3-1c32-4e0f-bebf-7bc060824be5.png)

Go to the location of the LumaViewPro-master folder in the terminal 

Open LumaViewPro using the terminal 

![Screenshot from 2022-07-12 18-53-04](https://user-images.githubusercontent.com/108957480/178635069-8e45b9f7-479a-40f0-bc2e-67554b0ff49a.png)

If having trouble with camera install the basler pylon driver at https://www.baslerweb.com/en/downloads/software-downloads/

