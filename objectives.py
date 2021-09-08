# USED TO GENERATE PROTOCOL JSON FILE

objectives = {
    "4x": {
        "desc": "4x description",
        "magnification": 4,
        "aperture": 0.1,
        "DOF": 55.5,
        "AF_min": 20.,
        "AF_max": 72.,
        "AF_range": 150.,
        "step_fine": 27.75,
        "step_course": 277.5
    },

    "10x": {
        "desc": "10x description",
        "magnification": 10,
        "aperture": 0.25,
        "DOF": 8.5,
        "AF_min": 8.,
        "AF_max": 36.,
        "AF_range": 75.,
        "step_fine": 4.25,
        "step_course": 42.5
    },

    "20x": {
        "desc": "20x description",
        "magnification": 20,
        "aperture": 0.40,
        "DOF": 5.8,
        "AF_min": 2.,
        "AF_max": 18.,
        "AF_range": 40.,
        "step_fine": 2.9,
        "step_course": 29.
    },

    "40x": {
        "desc": "40x description",
        "magnification": 40,
        "aperture": 0.65,
        "DOF": 1.0,
        "AF_min": 1.,
        "AF_max": 9.,
        "AF_range": 20.,
        "step_fine": 0.5,
        "step_course": 5.
    },

    "60x": {
        "desc": "60x description",
        "magnification": 60,
        "aperture": 0.85,
        "DOF": 0.4,
        "AF_min": 0.5,
        "AF_max": 6.,
        "AF_range": 15.,
        "step_fine": 0.2,
        "step_course": 2.
    },

    "100x": {
        "desc": "100x description",
        "magnification": 100,
        "aperture": 0.95,
        "DOF": 0.19,
        "AF_min": 0.2,
        "AF_max": 4.,
        "AF_range": 10.,
        "step_fine": 0.11,
        "step_course": 1.1
    }
}

import json

with open("./data/objectives.json", "w") as write_file:
    json.dump(objectives, write_file, indent = 4)
