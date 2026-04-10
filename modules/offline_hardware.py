import datetime
import math
from collections import deque

import numpy as np

from lvp_logger import logger


class OfflineLEDBoard:
    COLORS = ("BF", "PC", "DF", "Red", "Blue", "Green")

    def __init__(self):
        self.found = True
        self.driver = True
        self.led_ma = {color: -1.0 for color in self.COLORS}
        now = datetime.datetime.now()
        self._history = {
            color: deque([(now, -1.0)], maxlen=4096)
            for color in self.COLORS
        }

    def _append_state(self, color: str, mA: float):
        now = datetime.datetime.now()
        self.led_ma[color] = float(mA)
        self._history[color].append((now, float(mA)))
        cutoff = now - datetime.timedelta(seconds=5)
        history = self._history[color]
        while len(history) > 2 and history[1][0] < cutoff:
            history.popleft()

    def leds_enable(self):
        return

    def leds_disable(self):
        self.leds_off()

    def color2ch(self, color):
        if color == 'Blue':
            return 0
        elif color == 'Green':
            return 1
        elif color == 'Red':
            return 2
        elif color == 'BF':
            return 3
        elif color == 'PC':
            return 4
        elif color == 'DF':
            return 5
        return 3

    def ch2color(self, channel):
        if channel == 0:
            return 'Blue'
        elif channel == 1:
            return 'Green'
        elif channel == 2:
            return 'Red'
        elif channel == 3:
            return 'BF'
        elif channel == 4:
            return 'PC'
        elif channel == 5:
            return 'DF'
        return 'BF'

    def get_led_ma(self, color):
        return self.led_ma.get(color, -1)

    def is_led_on(self, color) -> bool:
        return self.get_led_ma(color) > 0

    def get_led_state(self, color) -> dict:
        mA = self.get_led_ma(color)
        return {
            'enabled': mA > 0,
            'illumination': mA,
        }

    def get_led_states(self) -> dict:
        return {
            color: self.get_led_state(color=color)
            for color in self.led_ma.keys()
        }

    def led_on(self, channel, mA):
        color = self.ch2color(channel=channel)
        self._append_state(color=color, mA=mA)
        logger.info(f"[OFFLINE LED] {color} ON @ {float(mA):.1f} mA")

    def led_off(self, channel):
        color = self.ch2color(channel=channel)
        self._append_state(color=color, mA=-1)
        logger.info(f"[OFFLINE LED] {color} OFF")

    def led_on_fast(self, channel, mA):
        self.led_on(channel=channel, mA=mA)

    def led_off_fast(self, channel):
        self.led_off(channel=channel)

    def leds_off(self):
        for color in self.led_ma.keys():
            self._append_state(color=color, mA=-1)
        logger.info("[OFFLINE LED] ALL OFF")

    def leds_off_fast(self):
        self.leds_off()

    def _average_ma_over_window(self, color: str, window_s: float) -> float:
        now = datetime.datetime.now()
        history = list(self._history[color])
        if len(history) == 0:
            return 0.0

        window_s = max(0.001, float(window_s))
        start = now - datetime.timedelta(seconds=window_s)

        state = history[0][1]
        idx = 0
        while idx < len(history) and history[idx][0] <= start:
            state = history[idx][1]
            idx += 1

        prev_t = start
        weighted = 0.0
        for ts, next_state in history[idx:]:
            dt = (ts - prev_t).total_seconds()
            if dt > 0:
                weighted += max(0.0, state) * dt
            state = next_state
            prev_t = ts

        dt = (now - prev_t).total_seconds()
        if dt > 0:
            weighted += max(0.0, state) * dt

        return weighted / window_s

    def get_average_total_ma(self, window_s: float) -> float:
        return sum(
            self._average_ma_over_window(color=color, window_s=window_s)
            for color in self.led_ma.keys()
        )


class OfflineMotorBoard:
    def __init__(self):
        self.found = True
        self.driver = True
        self.overshoot = False
        self._has_turret = False
        self.initial_homing_complete = True
        self.initial_t_homing_complete = True
        self._positions = {
            'X': 0.0,
            'Y': 0.0,
            'Z': 0.0,
            'T': 0.0,
        }
        self._targets = self._positions.copy()

    def zhome(self):
        self._positions['Z'] = 0.0
        self._targets['Z'] = 0.0

    def xyhome(self):
        self._positions['X'] = 0.0
        self._positions['Y'] = 0.0
        self._targets['X'] = 0.0
        self._targets['Y'] = 0.0
        self.initial_homing_complete = True

    def xycenter(self):
        self._positions['X'] = 60000.0
        self._positions['Y'] = 40000.0
        self._targets['X'] = 60000.0
        self._targets['Y'] = 40000.0

    def thome(self):
        self._positions['T'] = 1.0
        self._targets['T'] = 1.0
        self.initial_t_homing_complete = True

    def has_xyhomed(self):
        return True

    def has_thomed(self):
        return True

    def has_turret(self):
        return self._has_turret

    def target_pos(self, axis):
        return self._targets.get(axis, 0.0)

    def current_pos(self, axis):
        return self._positions.get(axis, 0.0)

    def move_abs_pos(self, axis, pos, overshoot_enabled=True, ignore_limits=False):
        self._targets[axis] = float(pos)
        self._positions[axis] = float(pos)
        self.overshoot = False

    def move_rel_pos(self, axis, um, overshoot_enabled=True):
        self.move_abs_pos(axis=axis, pos=self.current_pos(axis) + float(um), overshoot_enabled=overshoot_enabled)

    def home_status(self, axis):
        return True

    def target_status(self, axis):
        return True

    def reference_status(self, axis='X'):
        return "0" * 32

    def limit_switch_status(self, axis='X'):
        return False

    def set_acceleration_limits(self, val_pct: int):
        return

    def get_microscope_model(self):
        return "OFFLINE"


class OfflineCamera:
    SUPPORTED_PIXEL_FORMATS = ('Mono8', 'Mono12')

    def __init__(self, led_board: OfflineLEDBoard):
        self.active = True
        self.array = np.array([])
        self.model_name = "OfflineCamera"
        self.max_exposure = 10_000
        self._led_board = led_board
        self._gain = 1.0
        self._exposure_ms = 10.0
        self._auto_gain = False
        self._auto_exposure = False
        self._auto_target_brightness = 0.5
        self._binning_size = 1
        self._pixel_format = 'Mono8'
        self._frame_width = 1900
        self._frame_height = 1900
        self._min_frame = {'width': 64, 'height': 64}
        self._max_frame = {'width': 4096, 'height': 4096}
        self._rng = np.random.default_rng(seed=12345)
        self._base_scene_cache = None
        self._base_scene_shape = None

    def find_model_name(self):
        self.model_name = "OfflineCamera"

    def get_model_name(self):
        return self.model_name

    def set_max_exposure_time(self):
        self.max_exposure = 10_000

    def get_max_exposure(self):
        return self.max_exposure

    def set_pixel_format(self, pixel_format: str) -> bool:
        if pixel_format not in self.SUPPORTED_PIXEL_FORMATS:
            return False
        self._pixel_format = pixel_format
        return True

    def get_supported_pixel_formats(self) -> tuple:
        return self.SUPPORTED_PIXEL_FORMATS

    def set_binning_size(self, size: int) -> bool:
        size = int(size)
        if size < 1 or size > 4:
            return False
        self._binning_size = size
        return True

    def get_binning_size(self) -> int:
        return self._binning_size

    def update_auto_gain_target_brightness(self, auto_target_brightness: float):
        self._auto_target_brightness = float(auto_target_brightness)

    def get_min_frame_size(self) -> dict:
        return self._min_frame.copy()

    def get_max_frame_size(self) -> dict:
        return self._max_frame.copy()

    def get_frame_size(self):
        return {
            'width': self._frame_width,
            'height': self._frame_height,
        }

    def get_gain(self):
        return float(self._gain)

    def gain(self, gain):
        self._gain = float(gain)

    def auto_gain(self, state=True, target_brightness: float = 0.5, min_gain=None, max_gain=None):
        self._auto_gain = bool(state)
        self._auto_exposure = bool(state)
        self._auto_target_brightness = float(target_brightness)

    def auto_gain_once(self, state=True, target_brightness: float = 0.5, min_gain=None, max_gain=None):
        self._auto_target_brightness = float(target_brightness)
        if state:
            avg_ma = self._led_board.get_average_total_ma(window_s=max(0.01, self._exposure_ms / 1000.0))
            if avg_ma > 0:
                target_gain = np.clip((255.0 * target_brightness) / max(1.0, avg_ma * 0.35), 0.0, 30.0)
                self._gain = float(target_gain)
        else:
            self._auto_gain = False
            self._auto_exposure = False

    def exposure_t(self, t):
        self._exposure_ms = float(np.clip(t, 0.0, self.max_exposure))

    def get_exposure_t(self):
        return float(self._exposure_ms)

    def auto_exposure_t(self, state=True):
        self._auto_exposure = bool(state)

    def set_frame_size(self, w, h):
        width = int(np.clip(int(w), self._min_frame['width'], self._max_frame['width']))
        height = int(np.clip(int(h), self._min_frame['height'], self._max_frame['height']))
        self._frame_width = max(4, width - (width % 4))
        self._frame_height = max(4, height - (height % 4))
        self._base_scene_cache = None
        self._base_scene_shape = None

    def _base_scene(self):
        shape = (self._frame_height, self._frame_width)
        if self._base_scene_cache is not None and self._base_scene_shape == shape:
            return self._base_scene_cache

        y, x = np.indices(shape, dtype=np.float32)
        cx = self._frame_width / 2.0
        cy = self._frame_height / 2.0
        dx = (x - cx) / max(1.0, self._frame_width * 0.18)
        dy = (y - cy) / max(1.0, self._frame_height * 0.18)
        blob = np.exp(-(dx * dx + dy * dy) / 2.0)

        ring_dx = (x - self._frame_width * 0.36) / max(1.0, self._frame_width * 0.09)
        ring_dy = (y - self._frame_height * 0.58) / max(1.0, self._frame_height * 0.08)
        ring = np.exp(-(ring_dx * ring_dx + ring_dy * ring_dy) / 2.0)

        gradient = 0.25 + 0.15 * (x / max(1.0, self._frame_width)) + 0.1 * (y / max(1.0, self._frame_height))
        scene = np.clip(0.75 * blob + 0.25 * ring + 0.2 * gradient, 0.0, 1.0)
        self._base_scene_cache = scene
        self._base_scene_shape = shape
        return scene

    def _render_frame(self):
        exposure_s = max(0.001, self._exposure_ms / 1000.0)
        avg_ma = self._led_board.get_average_total_ma(window_s=exposure_s)
        gain_factor = 1.0 + max(0.0, self._gain) / 24.0
        signal = np.clip(avg_ma * 0.55 * gain_factor, 0.0, 255.0)

        scene = self._base_scene()
        frame = 8.0 + signal * scene
        noise_sigma = 2.0 + math.sqrt(max(signal, 1.0)) * 0.05
        frame = frame + self._rng.normal(loc=0.0, scale=noise_sigma, size=frame.shape)

        if self._pixel_format == 'Mono12':
            frame = np.clip(frame * 16.0, 0, 4095).astype(np.uint16)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        return frame

    def grab(self):
        self.array = self._render_frame()
        return True, datetime.datetime.now()
