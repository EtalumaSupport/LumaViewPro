# lumaviewpro
LumaViewPro is an open source fluorescence microscope control interface written in Python.  It is sponsored by Etaluma, Inc. and includes an example of an interface to the Etaluma LED control board and an initial version of a Basler camera.  Multi-axis motion control is anticipated in 2021. It is intended to offer a multi OS functionl GUI as well as a development tool for testing and building applications from.


## Installing dependencies on Windows 10
Download and install Python 3.9 from python page https://www.python.org/downloads/ for ALL users into new folder at (C:\Program Files\Python39). You will have to run the program in administrative mode.

Run the command prompt as administrator, and enter each of these commands. The versions that are installed on my current development computer are in parentheses.

* `python -m pip install --upgrade pip setuptools virtualenv` (pip-21.0.1 setuptools-54.2.0 virtualenv-20.4.3)
* `python -m pip install numpy` (numpy-1.20.1)
* `python -m pip install pyserial` (pyserial-3.5)
* `python -m pip install Pillow` (Pillow-8.1.2)
* `python -m pip install opencv-python` (opencv-python-4.5.1.48)
* `python -m pip install pypylon` (pypylon-1.7.2)
* `python -m pip install docutils pygments pypiwin32` (docutils-0.16 pygments-2.8.1 pypiwin32-223 pywin32-300)
* `python -m pip install kivy.deps.sdl2` (kivy_deps.sdl2-0.3.1-cp39-cp39-win_amd64.whl)
* `python -m pip install kivy.deps.glew` (kivy_deps.glew-0.3.0-cp39-cp39-win_amd64.whl)
*` python -m pip install kivy.deps.gstreamer` (kivy_deps.gstreamer-0.3.1-cp39-cp39-win_amd64.whl)
* `python -m pip install kivy`


## Installing dependencies on Mac OS X
Upgrade to the latest version of Mac OS X.  Also ensure you have the latest version of XCode installed from the [Mac App store](https://apps.apple.com/us/app/xcode/id497799835?mt=12).  Open XCode and let installation scripts download and run any additional tools needed.  Then open a terminal and enter the following command to install the XCode command line tools:

- `xcode-select --install` 

Download and install Homebrew according to the instructions on the [Homebrew page](https://brew.sh).  Then open a terminal and download and install Python 3.9 using Homebrew:

- `brew install python3`

Move the /usr/local/bin/pip binary to a temporary directory and create symlinks for pip3 and python3:

* `mkdir ~/temp` (if not already present)
* `sudo mv /usr/local/bin/pip ~/temp`
* `sudo ln -s /usr/local/bin/pip3.9 /usr/local/bin/pip`
* `sudo ln -s /usr/local/bin/python3 /usr/local/bin/python`

Install python packages using pip:
* `python -m pip install --upgrade pip setuptools virtualenv`
* `python -m pip install numpy`
* `python -m pip install pyserial`
* `python -m pip install Pillow`
* `python -m pip install opencv-python`
* `python -m pip install pypylon`
* `python -m pip install docutils pygments pypi`

Install dependencies using Homebrew:

* `brew install sdl2 sdl2_image sdl2_ttf sdl2_mixer`
* `brew install gstreamer gst-plugins-base`
* `brew install gst-plugins-good gst-plugins-bad gst-plugins-ugly`
* `brew install gst-libav`
* `brew install glew`

Install kivy using pip:
* `python -m pip install kivy`

## Running from Code on Windows 10 and MacOS

* Download code from github as a .ZIP file
* Unzip to your preferred folder
* Open command prompt
* Navigate to the folder containing lumaviewpro.py
* type: python lumaviewpro.py
