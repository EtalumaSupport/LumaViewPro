# USED TO GENERATE PROtOCOL JSON FILE

# Protocol Layout
protocol = {
    "period": 5.,
    "duration": 48.,
    "BF": {
        "save_folder": ".\\",
        "file_root": "bright_",
        "ill": 1.,
        "gain": 1.,
        "exp": 150.,
        "acquire": True
    },
    "Blue": {
        "save_folder": ".\\",
        "file_root": "blue_",
        "ill": 1.,
        "gain": 1.,
        "exp": 150.,
        "acquire": True
    },
    "Green": {
        "save_folder": ".\\",
        "file_root": "green_",
        "ill": 1.,
        "gain": 1.,
        "exp": 150.,
        "acquire": True
    },
    "Red": {
        "save_folder": ".\\",
        "file_root": "red_",
        "ill": 1.,
        "gain": 1.,
        "exp": 150.,
        "acquire": True
    },
    "Composite": {
        "save_folder": ".\\",
        "file_root": "composite_",
        "ill": '',
        "gain": '',
        "exp": '',
        "acquire": True
    }
}

import json

with open("./data/protocol.json", "w") as write_file:
    json.dump(protocol, write_file)
