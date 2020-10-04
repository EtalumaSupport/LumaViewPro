# USED TO GENERATE PROtOCOL JSON FILE

# Protocol Layout
protocol = {
    "microscope": 'LS620',
    "frame_width": 2592,
    "frame_height": 1944,
    "objective": '???',
    "period": 0.017,
    "duration": 0.17,
    "BF": {
        "save_folder": ".\\capture\\",
        "file_root": "W_",
        "ill": 0.,
        "gain": 0.,
        "exp": 20.,
        "led": 20.,
        "acquire": True
    },
    "Blue": {
        "save_folder": ".\\capture\\",
        "file_root": "B_",
        "ill": 0.,
        "gain": 0.,
        "exp": 20.,
        "led": 20.,
        "acquire": True
    },
    "Green": {
        "save_folder": ".\\capture\\",
        "file_root": "G_",
        "ill": 0.,
        "gain": 0.,
        "exp": 20.,
        "led": 20.,
        "acquire": True
    },
    "Red": {
        "save_folder": ".\\capture\\",
        "file_root": "R_",
        "ill": 0.,
        "gain": 0.,
        "exp": 20.,
        "led": 20.,
        "acquire": True
    },
    "Composite": {
        "acquire": False
    }
}

import json

with open("./data/protocol.json", "w") as write_file:
    json.dump(protocol, write_file)
