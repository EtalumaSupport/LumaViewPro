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
        "acquire": True
    },
    "Blue": {
        "save_folder": "./capture",
        "file_root": "B_",
        "ill": 100.,
        "gain": 0.,
        "gain_auto": False,
        "exp": 100.,
        "false_color": True,
        "acquire": True
    },
    "Green": {
        "save_folder": "./capture",
        "file_root": "G_",
        "ill": 100.,
        "gain": 0.,
        "gain_auto": False,
        "exp": 100.,
        "false_color": True,
        "acquire": True
    },
    "Red": {
        "save_folder": "./capture",
        "file_root": "R_",
        "ill": 100.,
        "gain": 0.,
        "gain_auto": False,
        "exp": 100.,
        "false_color": True,
        "acquire": True
    },
    "Composite": {
        "acquire": False
    }
}

import json

with open("./data/protocol.json", "w") as write_file:
    json.dump(protocol, write_file, indent = 4)
