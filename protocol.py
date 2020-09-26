# USED TO GENERATE PROtOCOL JSON FILE

# Protocol Layout
protocol = {
    "period": 0.01,
    "duration": 0.1.,
    "BF": {
        "save_folder": ".\\",
        "file_root": "bright_",
        "ill": 1.,
        "gain": 1.,
        "exp": 150.,
        "led": 20.,
        "acquire": False
    },
    "Blue": {
        "save_folder": ".\\",
        "file_root": "blue_",
        "ill": 1.,
        "gain": 1.,
        "exp": 150.,
        "led": 20.,
        "acquire": False
    },
    "Green": {
        "save_folder": ".\\",
        "file_root": "green_",
        "ill": 1.,
        "gain": 1.,
        "exp": 150.,
        "led": 20.,
        "acquire": False
    },
    "Red": {
        "save_folder": ".\\",
        "file_root": "red_",
        "ill": 1.,
        "gain": 1.,
        "exp": 150.,
        "led": 20.,
        "acquire": False
    },
    "Composite": {
        "acquire": False
    }
}

import json

with open("./data/protocol.json", "w") as write_file:
    json.dump(protocol, write_file)
