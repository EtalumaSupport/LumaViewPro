# USED TO GENERATE SCOPES JSON FILE

scopes = {
  "LS460": {
      "Fl channels": 0,
      "Focus": False,
      "XYStage": False,
  },
  "LS560": {
      "Fl channels": 1,
      "Focus": False,
      "XYStage": False,
  },
  "LS620": {
      "Fl channels": 3,
      "Focus": False,
      "XYStage": False,
  },
  "LS650": {
      "Fl channels": 3,
      "Focus": True,
      "XYStage": False,
  },
  "LS720": {
      "Fl channels": 3,
      "Focus": True,
      "XYStage": True,
  }
}


import json

with open("./data/scopes.json", "w") as write_file:
    json.dump(scopes, write_file, indent = 4)
