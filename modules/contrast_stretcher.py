
import numpy as np

class ContrastStretcher:

    def __init__(
        self,
        window_len: int,
        bottom_pct: float,
        top_pct: float,
    ):
        self._window_len = window_len
        self._bottom_pct = bottom_pct
        self._top_pct = top_pct

        self._data = {
            'min': [0],
            'range': [255],
        }

    
    def update(self, image: np.ndarray) -> np.ndarray:

        # Calculate bottom % and top % of pixel values to use for contrast stretching
        min_val = np.percentile(image, self._bottom_pct)
        max_val = np.percentile(image, 100-self._top_pct)
        range_diff = max_val - min_val

        # Save the values to a list for calculating rolling averages
        self._data['min'].append(min_val)
        self._data['range'].append(range_diff)

        # Clamp the rolling average window length to last N values
        if len(self._data['min']) > self._window_len:
            self._data['min'].pop(0)
            self._data['range'].pop(0)

        # Calculate the rolling average values
        min_val_avg = np.average(self._data['min'])
        range_val_avg = np.average(self._data['range'])
        if range_val_avg == 0:
            return image

        # Apply constrast stretching
        image = (255*((image - min_val_avg)/range_val_avg))
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image
