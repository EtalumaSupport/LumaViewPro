# LumaViewPro (pre-release)
LumaViewPro is an open source fluorescence microscope control interface written in Python.  It is sponsored by Etaluma, Inc. and includes an example of an interface to the Etaluma LED control board and an initial version of a Basler camera.  Multi-axis motion control is anticipated in 2021. It is intended to offer a multi OS functional GUI as well as a development tool for testing and building applications from.

![LVProScreenshotSept21](https://user-images.githubusercontent.com/74261093/132131289-ce4dce0b-3fcc-4d69-8dba-0862944329d9.png)


## Executable for Windows 10

The most recent executable for Windows 10 can be retrieved from:
https://drive.google.com/drive/folders/1vFG5wKAaB_DNMNctJCE5Sax5WXlNiw_3?usp=sharing

To use, download 'LumaViewPro.zip' and extract file. First install the camera driver. Then you can run the executable by double-clicking with no further installation. Note that the code may be more current than the executable. This executable was designed for the LS720.

## Installing dependencies on Windows 10 to run as a Python Script
Download and install from the python page https://www.python.org/downloads/.
- select "install now"
- select "add python to PATH"
- select "install launcher for all users"

Restart the computer

Open a command prompt and type the following
```
pip install --upgrade pip setuptools virtualenv
pip install numpy
pip install pyserial
pip install Pillow
pip install opencv-python
pip install pypylon
pip install docutils pygments pypiwin32
pip install kivy.deps.sdl2
pip install kivy.deps.glew
pip install kivy
pip install plyer
pip install scipy
pip install mcp2210-python
```

## Installing dependencies on Mac OS X to run as a Python Script
Upgrade to the latest version of Mac OS X.  Also ensure you have the latest version of XCode installed from the [Mac App store](https://apps.apple.com/us/app/xcode/id497799835?mt=12).  Open XCode and let installation scripts download and run any additional tools needed.  Then open a terminal and enter the following command to install the XCode command line tools:

```
xcode-select --install
```

Download and install Homebrew according to the instructions on the [Homebrew page](https://brew.sh).  Then open a terminal and download and install Python 3.9 using Homebrew:
```
brew install python3
```

Move the /usr/local/bin/pip binary to a temporary directory and create symlinks for pip3 and python3:

```
mkdir ~/temp
sudo mv /usr/local/bin/pip ~/temp
sudo ln -s /usr/local/bin/pip3.9 /usr/local/bin/pip
sudo ln -s /usr/local/bin/python3 /usr/local/bin/python
```

Install python packages using pip:
```
python3 -m pip install --upgrade pip setuptools virtualenv
python3 -m pip install numpy
python3 -m pip install pyserial
python3 -m pip install Pillow
python3 -m pip install opencv-python
python3 -m pip install pypylon
python3 -m pip install docutils pygments pypi
```

Install dependencies using Homebrew:
```
brew install sdl2 sdl2_image sdl2_ttf sdl2_mixer
brew install gstreamer gst-plugins-base
brew install gst-plugins-good gst-plugins-bad gst-plugins-ugly
brew install gst-libav
brew install glew
```

Install kivy using pip:
```
python -m pip install kivy
python -m pip install kivymd
```

## Running from Code on Windows 10 and MacOS

* Download code from github as a .ZIP file
* Unzip to your preferred folder
* Open command prompt
* Navigate to the folder containing lumaviewpro.py
* type: python lumaviewpro.py


## Creating your own scripts (without a GUI)

Download the files for interacting with the hardware
* trinamic.py
* ledboard.py
* pyloncamera.py
* script_example.py

Install the necessary libraries
* numpy
* pyserial
* time
* pypylon

Download and install the Basler Camera Driver
https://www.baslerweb.com/en/sales-support/downloads/software-downloads/

Edit the sample script
