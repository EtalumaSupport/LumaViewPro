# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
import os
import json


global settings

global debug_setting

def load_settings(logger, filename, lvp_appdata):
        
        global settings

        # load settings JSON file
        try:
            os.chdir(lvp_appdata)
            with open(filename, "r") as read_file:
                settings = json.load(read_file)
        except json.JSONDecodeError:
            logger.exception('[LVP Main  ] Incompatible JSON file for Microscope Settings')
        except Exception:
            logger.exception('[LVP Main  ] Unable to open file '+filename)
            raise

def load_lvp_settings(logger, lvp_appdata):
    global settings

    os.chdir(lvp_appdata)

    if os.path.exists("./data/current.json"):
        load_settings(logger, "./data/current.json", lvp_appdata)
    elif os.path.exists("./data/settings.json"):
        load_settings(logger, "./data/settings.json", lvp_appdata)
    else:
        if not os.path.isdir('./data'):
            raise FileNotFoundError("Cound't find 'data' directory.")
        else:
            raise FileNotFoundError('No settings files found.')

def load_debug_setting(directory):
    global debug_setting

    try:
        os.chdir(directory)
        if os.path.exists("./data/current.json"):
            filename = "./data/current.json"
        elif os.path.exists("./data/settings.json"):
            filename = "./data/settings.json"
        else:
            if not os.path.isdir('./data'):
                raise FileNotFoundError("Cound't find 'data' directory.")
            else:
                raise FileNotFoundError('No settings files found.')

        with open(filename, "r") as read_file:
            temp_settings = json.load(read_file)

        debug_setting = temp_settings.get("debug_mode", False)
        return debug_setting

    except Exception as e:
        raise e
    
