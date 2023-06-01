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
May 31, 2023
'''

'''
lvp_logger.py configures a standard python logger for LumaViewPro.
'''

import logging
import os
if not os.path.exists("logs/LVP_Log"):
    os.makedirs("logs/LVP_Log")

# file to which messages are logged 
LOG_FILE = 'logs/LVP_log/lumaviewpro.log'

# CustomFormatter class enables change in log format depending on log level 
class CustomFormatter(logging.Formatter):
    # if level is DEBUG/WARNING/ERROR/CRITICAL, log the level, message, time, and filename
    def __init__(self, 
                 fmt = '[%(levelname)s] %(asctime)s - %(filename)s - %(message)s', 
                 datefmt ='%m/%d/%Y %I:%M:%S %p'):
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record):
        if record.levelno == logging.INFO:
            # if INFO level, only log the message
            return record.getMessage()
        return logging.Formatter.format(self, record)

# ensures logger is specific to the file importing lvp_logger
logger = logging.getLogger(__name__)

# determines lowest level of messages to log (DEBUG < INFO < WARNING < ERROR < CRITICAL)
logger.setLevel(logging.INFO)

# obtains name of the module (file) importing lvp_logger
filename = '%s' % __file__
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(CustomFormatter())
logger.addHandler(file_handler)
