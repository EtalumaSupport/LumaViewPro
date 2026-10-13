# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import itertools
import pathlib
import re
from typing import TYPE_CHECKING, ClassVar

from lvp_logger import logger

import modules.common_utils as common_utils
from modules.exceptions import InstallationFileError, ProtocolRunRefusedError
from modules.api_surface import api, api_fields
from modules.path_utils import read_installation_file

if TYPE_CHECKING:
    from modules.scope_capabilities import ScopeCapabilities


def _row_label(index: int) -> str:
    """Spreadsheet-style row label for a zero-based mosaic row index.

    Bijective base-26: A..Z for the first 26 rows, then AA, AB, ... A naive
    chr(index + ord('A')) walks off the end of the alphabet past row 25 into
    punctuation -- including '_' at row 30, the very character that separates
    step-name tokens -- so a large-mosaic tile label could neither be matched
    as a tile nor round-tripped through the step name. Staying inside the
    uppercase alphabet keeps every label parseable and sortable.
    """
    label = ''
    index += 1
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        label = chr(ord('A') + remainder) + label
    return label


def _split_row_col(tile_label: str) -> tuple[str, int]:
    """Split a tile label into its row letters and integer column."""
    letters = ''.join(itertools.takewhile(str.isalpha, tile_label))
    return letters, int(tile_label[len(letters) :])


@api_fields('available', 'default', record=True)
class TilingConfig:
    """The tiling grids an installation offers, read from its tiling.json.

    Attributes:
        available: Each grid's label, as ``create_protocol`` and
            ``Protocol.apply_tiling`` accept it.
        default: The grid to preselect, one of ``available``.
    """

    available: tuple[str, ...]
    default: str

    DEFAULT_FILL_FACTORS: ClassVar[dict] = {
        'position': 1.0  # No overlap needed for position-based tiling
    }

    def __init__(self, tiling_configs_file_loc: pathlib.Path):
        """Read the installation's tiling.json.

        Raises:
            InstallationFileError: the file is missing, unreadable or not a
                JSON object, or is not the shape every reader of it needs: a
                ``data`` object of grids, each with a whole-number ``m`` and
                ``n``, and a ``metadata.default`` naming one of them. A grid
                left out of a broken file would be a grid the person cannot
                choose and a run could not lay out, so the file is refused
                whole, where it is read.
        """
        self._available_configs = read_installation_file(tiling_configs_file_loc)
        self._refuse_unless_whole(pathlib.Path(tiling_configs_file_loc))
        self.available = tuple(self._available_configs['data'].keys())
        self.default = self._available_configs['metadata']['default']

    def _refuse_unless_whole(self, path: pathlib.Path) -> None:
        """Refuse a tiling.json that is not the shape its readers need."""
        cfg = self._available_configs
        grids = cfg.get('data')
        if not isinstance(grids, dict):
            raise InstallationFileError(path, "has no 'data' object of tiling grids")
        for label, grid in grids.items():
            if not (
                isinstance(grid, dict) and all(type(grid.get(field)) is int for field in ('m', 'n'))
            ):
                raise InstallationFileError(
                    path, f"has a grid {label!r} without a whole-number 'm' and 'n'"
                )
        metadata = cfg.get('metadata')
        default = metadata.get('default') if isinstance(metadata, dict) else None
        if default not in grids:
            raise InstallationFileError(
                path, f"names no default grid among its own ('metadata.default' is {default!r})"
            )

    @api(in_process=True, reads=True)
    def available_configs(self) -> list[str]:
        """``available``, as a list."""
        return list(self.available)

    def get_mxn_size(self, config_label: str) -> dict:
        return self._available_configs['data'][config_label]

    def get_label_from_mxn_size(self, m: int, n: int) -> str | None:
        for config_label, config_data in self._available_configs['data'].items():
            if (config_data['m'] == m) and (config_data['n'] == n):
                return config_label

        return None

    def determine_tiling_label_from_tiles(self, tiles: list):
        """Infer the m x n tiling label from the steps' Tile column values.

        The Tile column is the authoritative per-step tile assignment
        (empty for untiled steps); reading it directly means a step name
        containing tile-shaped user text can never fake a tiling.
        """
        label_letters = set()
        label_numbers = set()
        malformed = set()
        for tile in tiles:
            if tile in (None, ''):
                continue

            tile = str(tile)
            if not re.fullmatch(r'[A-Z]+\d+', tile):
                # A malformed cell (hand-edited or corrupt TSV) contributes
                # no tile -- _split_row_col would raise on it -- matching
                # the old regex-gated name parse, which simply never matched
                # such content.
                malformed.add(tile)
                continue

            label_letter, label_number = _split_row_col(tile)

            label_letters.add(label_letter)
            label_numbers.add(label_number)

        if malformed:
            logger.warning(
                f'[Tiling    ] Ignored malformed Tile value(s) during tiling '
                f'inference: {", ".join(repr(v) for v in sorted(malformed)[:5])}'
            )

        m = len(label_letters)
        n = len(label_numbers)
        if m != n:
            logger.warning(
                f'TilingConfig] Tiling configuration found as non-symmetric ({m}x{n}). Protocol tiling label will be innacurate.'
            )
            return None
            # raise Exception(f"Tiling configuration requires equal dimensions, but found {m}x{n}")

        return self.get_label_from_mxn_size(m=m, n=n)

    @api(in_process=True, reads=True)
    def default_config(self) -> str:
        """``default``."""
        return self.default

    @staticmethod
    def no_tiling_label() -> str:
        """The label for a single, un-tiled field of view.

        Static because it names a fixed point in the label vocabulary
        rather than reading the loaded config: callers that need "no
        tiling" while assembling a run config get it without loading
        tiling.json, and there is still only one spelling of it.
        """
        return '1x1'

    def _calc_range(
        self,
        config_label: str,
        focal_length: float,
        frame_size: dict[int],
        fill_factor: int,
        binning_size: int,
        capabilities: 'ScopeCapabilities',
    ) -> dict[dict]:

        tiling_mxn = self.get_mxn_size(config_label)

        fov_size = common_utils.get_field_of_view(
            focal_length=focal_length,
            frame_size=frame_size,
            binning_size=binning_size,
            capabilities=capabilities,
        )
        if fov_size is None:
            # Tiling steps are derived from the field of view; a scope that
            # cannot report its pixel size has no field of view to tile
            # against. A grid laid out from a guessed FOV produces a
            # misaligned montage, which is worse than no tiling -- refuse.
            raise ProtocolRunRefusedError(
                reason='tiling_scale_unknown',
                message=(
                    'This microscope cannot report its pixel size, so the spacing of a '
                    'tile grid is unknown. Build the protocol without tiling (1x1).'
                ),
            )

        x_step = fill_factor * fov_size['width']
        y_step = fill_factor * fov_size['height']

        # The requested m x n is the contract: overlap shrinks the step
        # (and therefore the total covered area) but never changes the
        # tile count. At 0% overlap tiles abut exactly. An earlier design
        # grew the grid to preserve the zero-overlap coverage area, which
        # silently turned a requested 2x2 into a 3x3.
        return {
            'mxn': tiling_mxn,
            'step': {
                'x': x_step,
                'y': y_step,
            },
        }

    def get_tile_centers(
        self,
        config_label: str,
        focal_length: float | None,
        frame_size: dict[int],
        fill_factor: int,
        binning_size: int,
        *,
        capabilities: 'ScopeCapabilities',
    ) -> dict:
        """Tile centre offsets for a grid, from the optics the caller hands in.

        The spacing derives from the field of view, which derives from the
        scope's optics and the step's objective; the layout holds no scope, so
        the caller that owns one passes its capabilities. A 1x1 grid needs no
        spacing and asks nothing of them.
        """
        tiling_mxn = self.get_mxn_size(config_label)
        if (tiling_mxn['m'] == 1) and (tiling_mxn['n'] == 1):
            # The offset formula below multiplies the step by (index -
            # (count - 1) / 2), which is identically zero for a single
            # tile -- the step is not an input to this layout, so the
            # field of view (and therefore the pixel size) is not
            # required. Grids with real spacing still refuse below when
            # the scope cannot report its pixel size.
            x_step = 0.0
            y_step = 0.0
        else:
            ranges = self._calc_range(
                config_label=config_label,
                focal_length=focal_length,
                frame_size=frame_size,
                fill_factor=fill_factor,
                binning_size=binning_size,
                capabilities=capabilities,
            )
            x_step = ranges['step']['x']
            y_step = ranges['step']['y']

        tiles = {}

        PRECISION = 2  # Digits

        for i, j in itertools.product(range(tiling_mxn['m']), range(tiling_mxn['n'])):
            if (tiling_mxn['m'] == 1) and (tiling_mxn['n'] == 1):
                # Handle special case where tiling is 1x1 (i.e. no tiling)
                tile_label = ''
            else:
                row_letter = _row_label(i)
                col_number = j + 1
                tile_label = f'{row_letter}{col_number}'

            tiles[tile_label] = {
                'x': round((j - (tiling_mxn['n'] - 1) / 2) * x_step, PRECISION),
                'y': round((i - (tiling_mxn['m'] - 1) / 2) * y_step, PRECISION),
            }

        return tiles

    @staticmethod
    def validate_overlap_percent(overlap_percent: float) -> float:
        try:
            overlap_percent = float(overlap_percent)
        except (TypeError, ValueError):
            raise ValueError(f'Tile overlap must be a number, got {overlap_percent!r}') from None

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
