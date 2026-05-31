# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import itertools
import json
import math
import pathlib

from lvp_logger import logger

import modules.common_utils as common_utils


class TilingConfig:
    DEFAULT_FILL_FACTORS = {
        'position': 1.0  # No overlap needed for position-based tiling
    }

    def __init__(self, tiling_configs_file_loc: pathlib.Path):
        try:
            with open(tiling_configs_file_loc) as fp:
                self._available_configs = json.load(fp)
        except FileNotFoundError as e:
            logger.error(f'[Tiling    ] tiling.json not found at {tiling_configs_file_loc}')
            raise RuntimeError(
                f'Required file tiling.json not found at {tiling_configs_file_loc}. '
                'Please reinstall or restore from backup.'
            ) from e
        except json.JSONDecodeError as e:
            logger.error(f'[Tiling    ] tiling.json is corrupt: {e}')
            raise RuntimeError(
                f'tiling.json is corrupt ({e}). Please restore from backup or reinstall.'
            ) from e

        self._validate_tiling(tiling_configs_file_loc)

    def _validate_tiling(self, filepath):
        """Check tiling.json has required structure."""
        cfg = self._available_configs
        if not isinstance(cfg, dict):
            raise ValueError(f'tiling.json at {filepath}: expected dict, got {type(cfg).__name__}')
        if 'metadata' not in cfg:
            logger.warning(f"[Tiling    ] missing 'metadata' key in {filepath}")
        if 'data' not in cfg:
            raise ValueError(f"tiling.json at {filepath}: missing required 'data' key")
        if not isinstance(cfg['data'], dict):
            raise ValueError(f"tiling.json at {filepath}: 'data' must be a dict")
        for label, entry in cfg['data'].items():
            if not isinstance(entry, dict):
                logger.warning(f"[Tiling    ] '{label}' should be dict in {filepath}")
                continue
            for field in ('m', 'n'):
                if field not in entry:
                    logger.warning(f"[Tiling    ] '{label}' missing '{field}' in {filepath}")
                elif not isinstance(entry[field], int):
                    logger.warning(
                        f"[Tiling    ] '{label}'.'{field}' should be int, "
                        f'got {type(entry[field]).__name__} in {filepath}'
                    )

    def available_configs(self) -> list[str]:
        return list(self._available_configs['data'].keys())

    def get_mxn_size(self, config_label: str) -> dict:
        return self._available_configs['data'][config_label]

    def get_label_from_mxn_size(self, m: int, n: int) -> str | None:
        for config_label, config_data in self._available_configs['data'].items():
            if (config_data['m'] == m) and (config_data['n'] == n):
                return config_label

        return None

    def determine_tiling_label_from_names(self, names: list):
        label_letters = set()
        label_numbers = set()
        for name in names:
            label = common_utils.get_tile_label_from_name(name=name)
            if label is None:
                continue

            label_letter = label[0]
            label_number = int(label[1:])

            label_letters.add(label_letter)
            label_numbers.add(label_number)

        m = len(label_letters)
        n = len(label_numbers)
        if m != n:
            logger.warning(
                f'TilingConfig] Tiling configuration found as non-symmetric ({m}x{n}). Protocol tiling label will be innacurate.'
            )
            return None
            # raise Exception(f"Tiling configuration requires equal dimensions, but found {m}x{n}")

        return self.get_label_from_mxn_size(m=m, n=n)

    def default_config(self) -> str:
        return self._available_configs['metadata']['default']

    def no_tiling_label(self) -> str:
        return '1x1'

    def _calc_range(
        self,
        config_label: str,
        focal_length: float,
        frame_size: dict[int],
        fill_factor: int,
        binning_size: int,
    ) -> dict[dict]:

        tiling_mxn = self.get_mxn_size(config_label)

        fov_size = common_utils.get_field_of_view(
            focal_length=focal_length,
            frame_size=frame_size,
            binning_size=binning_size,
        )

        x_step = fill_factor * fov_size['width']
        y_step = fill_factor * fov_size['height']

        target_m = tiling_mxn['m']
        target_n = tiling_mxn['n']
        actual_m = self._overlap_preserving_tile_count(
            target_count=target_m,
            fill_factor=fill_factor,
        )
        actual_n = self._overlap_preserving_tile_count(
            target_count=target_n,
            fill_factor=fill_factor,
        )

        # Stage center derived from motorconfig travel limits.
        # Guards: ctx not initialized (CLI / headless startup), scope
        # not built, or no XY stage (NullMotionBoard fallback when
        # motor is absent). Each falls to DEFAULT_STAGE_TRAVEL_UM rather
        # than KeyError'ing on an empty axis_travel_limits_um dict.
        import modules.app_context as _app_ctx

        ctx = _app_ctx.ctx
        caps = None
        if ctx is not None and ctx.scope is not None:
            caps = ctx.scope.capabilities
        if caps is not None and caps.has_xy_stage:
            x_center = caps.axis_travel_limits_um['X'] / 2
            y_center = caps.axis_travel_limits_um['Y'] / 2
        else:
            from modules.common_utils import DEFAULT_STAGE_TRAVEL_UM

            x_center = DEFAULT_STAGE_TRAVEL_UM['x'] / 2
            y_center = DEFAULT_STAGE_TRAVEL_UM['y'] / 2
        tiling_min = {
            'x': x_center - target_n * fov_size['width'] / 2,
            'y': y_center - target_m * fov_size['height'] / 2,
        }

        tiling_max = {
            'x': x_center + target_n * fov_size['width'] / 2,
            'y': y_center + target_m * fov_size['height'] / 2,
        }

        return {
            'mxn': tiling_mxn,
            'actual_mxn': {
                'm': actual_m,
                'n': actual_n,
            },
            'step': {
                'x': x_step,
                'y': y_step,
            },
            'min': tiling_min,
            'max': tiling_max,
        }

    def get_tile_centers(
        self,
        config_label: str,
        focal_length: float,
        frame_size: dict[int],
        fill_factor: int,
        binning_size: int,
    ) -> dict:
        ranges = self._calc_range(
            config_label=config_label,
            focal_length=focal_length,
            frame_size=frame_size,
            fill_factor=fill_factor,
            binning_size=binning_size,
        )

        tiling_mxn = ranges['actual_mxn']
        x_step = ranges['step']['x']
        y_step = ranges['step']['y']

        tiles = {}

        PRECISION = 2  # Digits

        for i, j in itertools.product(range(tiling_mxn['m']), range(tiling_mxn['n'])):
            if (tiling_mxn['m'] == 1) and (tiling_mxn['n'] == 1):
                # Handle special case where tiling is 1x1 (i.e. no tiling)
                tile_label = ''
            else:
                row_letter = chr(i + ord('A'))
                col_number = j + 1
                tile_label = f'{row_letter}{col_number}'

            tiles[tile_label] = {
                'x': round((j - (tiling_mxn['n'] - 1) / 2) * x_step, PRECISION),
                'y': round((i - (tiling_mxn['m'] - 1) / 2) * y_step, PRECISION),
            }

        return tiles

    @staticmethod
    def _overlap_preserving_tile_count(target_count: int, fill_factor: float) -> int:
        if target_count <= 1:
            return target_count
        if fill_factor >= 1.0:
            return target_count

        target_span_in_fovs = target_count - 1
        return math.ceil(target_span_in_fovs / fill_factor + 1 - 1e-12)

    @staticmethod
    def validate_overlap_percent(overlap_percent: float) -> float:
        try:
            overlap_percent = float(overlap_percent)
        except (TypeError, ValueError):
            raise ValueError(
                f'Tile overlap must be a number, got {overlap_percent!r}'
            ) from None

        if overlap_percent < 0.0 or overlap_percent > 50.0:
            raise ValueError(
                f'Tile overlap must be between 0 and 50 percent, got {overlap_percent}'
            )
        return overlap_percent

    @staticmethod
    def fill_factor_from_overlap_percent(overlap_percent: float) -> float:
        overlap_percent = TilingConfig.validate_overlap_percent(overlap_percent)
        overlap_fraction = overlap_percent / 100.0
        return 1.0 - overlap_fraction
