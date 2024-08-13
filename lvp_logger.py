#!/usr/bin/python3
'''
MIT License

Copyright (c) 2023 Etaluma, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyribackground_downght notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

```
This open source software was developed for use with Etaluma microscopes.

AUTHORS:
Gerard Decker, The Earthineering Company

MODIFIED:
June 24, 2023
'''

'''
lvp_logger.py configures a standard python logger for LumaViewPro.
'''

import logging
from logging.handlers import RotatingFileHandler
import os
import sys
import ctypes
import userpaths

global windows_machine 

windows_machine = False

if os.name == "nt":
    windows_machine = True

abspath = os.path.abspath(__file__)
basename = os.path.basename(__file__)
script_path = abspath[:-len(basename)]

os.chdir(script_path)

version = ""
try:
    with open("version.txt") as f:
        version = f.readlines()[0].strip()
except:
    pass

if windows_machine:

    documents_folder = userpaths.get_my_documents()
    lvp_appdata = os.path.join(documents_folder, f"LumaViewPro {version}")

    os.chdir(lvp_appdata)

os.makedirs("logs/LVP_Log", exist_ok=True)

# file to which messages are logged 
LOG_FILE = 'logs/LVP_Log/lumaviewpro.log'

# CustomFormatter class enables change in log format depending on log level 
class CustomFormatter(logging.Formatter):
    # if level is DEBUG/WARNING/ERROR/CRITICAL, log the level, message, time, and filename
    def __init__(self, 
                 fmt = '[%(levelname)s] %(asctime)s.%(msecs)03d - %(filename)s - %(message)s', 
                 datefmt ='%m/%d/%Y %H:%M:%S'):
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record):
        # if record.levelno == logging.INFO:
        #     # if INFO level, only log the message
        #     return record.getMessage()
        return logging.Formatter.format(self, record)
    
def minimize_logger_window():
    if sys.platform == "win32":
        try:
            console_window = ctypes.windll.kernel32.GetConsoleWindow()
            if console_window:
                # Setting the found console window to a minimized state (state 6)
                ctypes.windll.user32.ShowWindow(console_window, 6)
                logger.info("[Logger  ] Console window minimized")
            else:
                logger.warning("[Logger  ] Console window not found.")
        except Exception as e:
            logger.error(f"[Logger  ] Failed to minimize console window: {e}")

# ensures logger is specific to the file importing lvp_logger
logger = logging.getLogger(__name__)

# determines lowest level of messages to log (DEBUG < INFO < WARNING < ERROR < CRITICAL)
logger.setLevel(logging.INFO)

# obtains name of the module (file) importing lvp_logger
filename = '%s' % __file__
file_handler = RotatingFileHandler(
    LOG_FILE,
    mode='a',
    maxBytes=5*1024*1024, 
    backupCount=2,
    encoding=None,
    delay=False
)

# Modify default log naming behavior so that it ends as .x.log instead of .log.x
file_handler.namer = lambda name: name.replace('.log', '') + '.log'

# file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(CustomFormatter())
logger.addHandler(file_handler)
minimize_logger_window()
