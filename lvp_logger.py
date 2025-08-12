#!/usr/bin/python3
'''
MIT License

Copyright (c) 2024 Etaluma, Inc.

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

try:
    with open("marker.lvpinstalled") as f:
        lvp_installed = True
except:
    lvp_installed = False

if windows_machine and (lvp_installed == True):

    documents_folder = userpaths.get_my_documents()
    lvp_appdata = os.path.join(documents_folder, f"LumaViewPro {version}")

    os.chdir(lvp_appdata)
    
else:
    lvp_appdata = script_path

os.makedirs("logs/LVP_Log", exist_ok=True)

# files to which messages are logged 
LOG_FILE = 'logs/LVP_Log/lumaviewpro.log'

# CustomFormatter class enables change in log format depending on log level 
class CustomFormatter(logging.Formatter):
    # if level is DEBUG/WARNING/ERROR/CRITICAL, log the level, message, time, and filename
    def __init__(self, 
                 fmt = '[%(levelname)s] [%(threadName)s] %(asctime)s.%(msecs)03d - %(filename)s - %(message)s', 
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

#TODO Separate crash logs into a separate file that contains any other info we might need to debug (settings.json maybe) besides stacktrace

# Log traceback if we have a crash to tell us more info on what happened
def custom_except_hook(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        logger.critical("Logger ] Keyboard interrupt quit.")
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logger.critical("Logger ] CRASH - Uncaught Exception: ", exc_info=(exc_type, exc_value, exc_traceback))

# ensures logger is specific to the file importing lvp_logger
logger = logging.getLogger(__name__)
# Prevent logs from propagating to root (and the console)
logger.propagate = False

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
    delay=False,
)
file_handler.namer = lambda name: name.replace('.log', '') + '.log'
file_handler.setFormatter(CustomFormatter())

# Separate error/critical file with extra debugging context (e.g., settings.json)

logger.addHandler(file_handler)

# Best-effort: remove any existing console/stream handlers from root to reduce terminal noise
try:
    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        if isinstance(h, logging.StreamHandler):
            root_logger.removeHandler(h)
except Exception:
    pass

sys.excepthook = custom_except_hook
minimize_logger_window()
logging.disable(logging.DEBUG)
