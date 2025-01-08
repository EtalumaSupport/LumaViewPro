import os
import json
from lvp_logger import lvp_appdata, logger

global settings

def load_settings(logger, filename):
        
        global settings

        # load settings JSON file
        try:
            os.chdir(lvp_appdata)
            read_file = open(filename, "r")
        except:
            logger.exception('[LVP Main  ] Unable to open file '+filename)
            raise
            
        else:
            try:
                settings = json.load(read_file)
            except:
                logger.exception('[LVP Main  ] Incompatible JSON file for Microscope Settings')

os.chdir(lvp_appdata)

if os.path.exists("./data/current.json"):
    load_settings(logger, "./data/current.json")
elif os.path.exists("./data/settings.json"):
    load_settings(logger, "./data/settings.json")
else:
    if not os.path.isdir('./data'):
        raise FileNotFoundError("Cound't find 'data' directory.")
    else:
        raise FileNotFoundError('No settings files found.')