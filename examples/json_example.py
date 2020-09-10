
# https://realpython.com/python-json/
import json

protocol = {
    "period": 5.,
    "duration": 48.,
    "BF": {
        "save_folder": ".\\",
        "file_root": "bright_",
        "ill": 1.,
        "gain": 1.,
        "exp": 150.,
        "Acq": True
    },
    "Blue": {
        "save_folder": ".\\",
        "file_root": "bright_",
        "ill": 1.,
        "gain": 1.,
        "exp": 150.,
        "Acq": True
    },
    "Green": {
        "save_folder": ".\\",
        "file_root": "bright_",
        "ill": 1.,
        "gain": 1.,
        "exp": 150.,
        "Acq": True
    },
    "Red": {
        "save_folder": ".\\",
        "file_root": "bright_",
        "ill": 1.,
        "gain": 1.,
        "exp": 150.,
        "Acq": True
    },
    "Composite": {
        "save_folder": ".\\",
        "file_root": "bright_",
        "Acq": True
    }
}

protocol['period'] = 12

with open("protocol.json", "w") as write_file:
    json.dump(protocol, write_file)

with open("protocol.json", "r") as read_file:
    protocol = json.load(read_file)
