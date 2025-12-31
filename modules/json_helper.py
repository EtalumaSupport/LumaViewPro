import json
import numpy as np

from modules.image_capture.image_capture_enums import ImageFileFormat

class CustomJSONizer(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, ImageFileFormat):
            return obj.value
        
        return super().default(obj)
