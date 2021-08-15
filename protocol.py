# USED TO GENERATE PROTOCOL JSON FILE

# Protocol Layout
protocol = {
    "microscope": "LS720",
    "objective": {
        "name": "unknown",
        "magnification": 40,
        "FOV": "tbd"
    },
    "port": 'COM3',
    "frame_width": 1900,
    "frame_height": 1900,
    "live_folder": "./capture",
    "x_bookmark": 0,
    "y_bookmark": 0,
    "z_bookmark": 0,
    "period": 0.1,
    "duration": 0.1,
    "BF": {
        "save_folder": "./capture",
        "file_root": "W_",
        "ill": 100.,
        "gain": 0.,
        "gain_auto": False,
        "exp": 100.,
        "false_color": False,
        "acquire": True,
        "focus": 0.
    },
    "Blue": {
        "save_folder": "./capture",
        "file_root": "B_",
        "ill": 100.,
        "gain": 0.,
        "gain_auto": False,
        "exp": 100.,
        "false_color": True,
        "acquire": True,
        "focus": 0.
    },
    "Green": {
        "save_folder": "./capture",
        "file_root": "G_",
        "ill": 100.,
        "gain": 0.,
        "gain_auto": False,
        "exp": 100.,
        "false_color": True,
        "acquire": True,
        "focus": 0.
    },
    "Red": {
        "save_folder": "./capture",
        "file_root": "R_",
        "ill": 100.,
        "gain": 0.,
        "gain_auto": False,
        "exp": 100.,
        "false_color": True,
        "acquire": True,
        "focus": 0.
    },
    "Composite": {
        "acquire": False
    }
}

import json

with open("./data/protocol.json", "w") as write_file:
    json.dump(protocol, write_file, indent = 4)
