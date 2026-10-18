# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The simulated scene: one specimen lying on the stage, which every simulated sensor images.

Both simulated cameras -- the stand-in ``SimulatedCamera`` and the simulated
FX2's MT9P031 -- draw their frames from this one sample, so a simulated
LS720 and a simulated LS850 at one stage position and objective look at the
same cells, and a change to the scene is made once.

The sample is laid out in stage micrometres. A stage position shows the
same cells every time it is imaged, two frames whose fields overlap on the
stage overlap in their images, and the sample never repeats, so a stitch
or a likeness judge on the simulator has a known answer. A frame draws the
window its camera sees at the objective's image scale, so a turret move
changes the scale.

The sample is seen drifting one pixel around a circle so a running stream
looks alive. The four frames of an older cycle were different crops with
no focus, so what a focus metric scored at one Z was whichever crop came
up, and a sweep found its peak there.
"""

from __future__ import annotations

import collections
import math

import cv2
import numpy as np

# The scale a frame is drawn at when no objective's is known: a camera with
# no scope around it, no objective set, or a model whose optics are not
# declared. It is the scene's own, never a value written into an image.
NOMINAL_UM_PER_PX = 2.0

# The drift: a step a frame around a circle one pixel across, which closes
# the loop so the wrap is the same size step as the others. One pixel was
# set by watching the live display: larger reads as the sample being
# jostled. It changes a focus score by about 0.03 %, against 4 % for a
# quarter of a depth of field off focus, so a sweep still finds the plane.
DRIFT_STEPS = tuple(
    (round(np.sin(2 * np.pi * i / 4)), round(np.cos(2 * np.pi * i / 4))) for i in range(4)
)

_SEED = 20260827

# The stage is cut into square cells of this size, each holding the cells
# its own seeded generator places, so a cell is the same whenever it is
# drawn and the plane needs no store of its own.
_GRID_UM = 128.0
# One cell per 16000 um^2: the field's density when it was drawn in sensor
# pixels at the nominal scale, one cell per 4000 pixels.
_CELLS_PER_GRID = _GRID_UM**2 / 16000.0
# A cell's radii, and the farthest it reaches past its centre: the largest
# radius with its membrane, so a cell centred outside a tile still draws
# the part of it inside.
_RADIUS_UM = (20.0, 52.0)
_REACH_UM = 60.0
_MEMBRANE_UM = 4.0
_NUCLEUS_MIN_UM = 6.0
# The medium's gentle unevenness, as a real background has: value noise on
# a lattice this far apart.
_MEDIUM_UM = 320.0

# The plane is drawn in square tiles of this many pixels at each scale and
# a frame composed from the tiles it touches, so a jog draws only the
# tiles entering the view. Each tile is drawn with a margin wider than the
# resolution blur reaches, so the blur sees its neighbours' pixels and a
# tile's edge is not a seam.
_TILE_PX = 256
_TILE_MARGIN_PX = 4
# A bound on the cache: more than a full native frame at binning 1 needs,
# about 64 kB a tile.
_TILES_KEPT = 512
# The grid squares' cells kept, drawn again by the next tile they reach
# into: a few hundred bytes each.
_SQUARES_KEPT = 65536

# cv2 draws at sub-pixel positions in fixed point with this many fraction
# bits, so a cell lands where its micrometres put it at every scale.
_SHIFT_BITS = 4
_ONE = 1 << _SHIFT_BITS


class Specimen:
    """The sample on the stage, drawn as a camera sees it.

    Holds the tiles drawn so far and the cells of the grid squares they
    drew. One per camera: a camera draws from its
    own frame-making thread, so the cache is never shared between threads.
    """

    def __init__(self, seed: int = _SEED):
        self._seed = seed
        self._tiles: collections.OrderedDict[tuple, np.ndarray] = collections.OrderedDict()
        self._squares: collections.OrderedDict[tuple, tuple] = collections.OrderedDict()

    def view(
        self,
        x_um: float | None,
        y_um: float | None,
        h: int,
        w: int,
        um_per_px: float | None,
        shift: tuple[int, int] = (0, 0),
    ) -> np.ndarray:
        """The in-focus ``h`` x ``w`` window centred on the stage at (``x_um``, ``y_um``).

        Stage X runs along the columns and Y along the rows, so as the stage
        moves to higher X and Y the sample moves toward lower column and
        row: the direction the stitcher's placement and the save's row flip
        expect.

        Args:
            x_um: The stage's X; None on a scope with no X axis, which holds
                still at 0.
            y_um: The stage's Y, likewise.
            h: Window height in pixels.
            w: Window width in pixels.
            um_per_px: The image scale; None draws at ``NOMINAL_UM_PER_PX``.
            shift: (rows, columns) the window is moved by on the sample:
                the drift.

        Returns:
            A grayscale uint8 array of shape ``(h, w)``: cells with a bright
            membrane, a fainter cytoplasm, a bright nucleus and small bright
            puncta, with edges at several scales, so a focus metric has a
            sharp peak at focus and falls away on either side. Grey values
            stay inside a mid band, so no frame is fully black or blown out
            and a caller's brightness scaling reads as exposure.
        """
        scale = NOMINAL_UM_PER_PX if um_per_px is None else float(um_per_px)
        top = round((y_um or 0.0) / scale) - h // 2 + shift[0]
        left = round((x_um or 0.0) / scale) - w // 2 + shift[1]
        out = np.empty((h, w), dtype=np.uint8)
        for ty in range(top // _TILE_PX, (top + h - 1) // _TILE_PX + 1):
            for tx in range(left // _TILE_PX, (left + w - 1) // _TILE_PX + 1):
                tile = self._tile(ty, tx, scale)
                r0, c0 = ty * _TILE_PX, tx * _TILE_PX
                r1, c1 = max(top, r0), max(left, c0)
                r2, c2 = min(top + h, r0 + _TILE_PX), min(left + w, c0 + _TILE_PX)
                out[r1 - top : r2 - top, c1 - left : c2 - left] = tile[
                    r1 - r0 : r2 - r0, c1 - c0 : c2 - c0
                ]
        return out

    def _tile(self, ty: int, tx: int, scale: float) -> np.ndarray:
        """Tile (``ty``, ``tx``) of the plane at ``scale``, drawn once and kept while it is in use."""
        key = (scale, ty, tx)
        tile = self._tiles.get(key)
        if tile is not None:
            self._tiles.move_to_end(key)
            return tile
        tile = self._draw(ty * _TILE_PX - _TILE_MARGIN_PX, tx * _TILE_PX - _TILE_MARGIN_PX, scale)
        tile = tile[_TILE_MARGIN_PX:-_TILE_MARGIN_PX, _TILE_MARGIN_PX:-_TILE_MARGIN_PX]
        self._tiles[key] = tile
        if len(self._tiles) > _TILES_KEPT:
            self._tiles.popitem(last=False)
        return tile

    def _draw(self, top: int, left: int, scale: float) -> np.ndarray:
        """The plane's pixels from (``top``, ``left``), a tile and its margin square.

        Plane pixel (r, c) covers the stage from (c, r) x ``scale`` to the
        next pixel's start.
        """
        size = _TILE_PX + 2 * _TILE_MARGIN_PX
        rows_um = (top + np.arange(size) + 0.5) * scale
        cols_um = (left + np.arange(size) + 0.5) * scale
        canvas = (45.0 + self._medium(rows_um, cols_um) * 12.0).astype(np.uint8)

        y0, y1 = top * scale - _REACH_UM, (top + size) * scale + _REACH_UM
        x0, x1 = left * scale - _REACH_UM, (left + size) * scale + _REACH_UM
        squares = [
            self._cells(gx, gy)
            for gy in range(math.floor(y0 / _GRID_UM), math.floor(y1 / _GRID_UM) + 1)
            for gx in range(math.floor(x0 / _GRID_UM), math.floor(x1 / _GRID_UM) + 1)
        ]
        cells = np.concatenate([cells for cells, _ in squares])
        puncta = np.concatenate([puncta for _, puncta in squares])

        def fixed(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
            # Stage points as this canvas's fixed-point pixels, cv2's pixel
            # centres being at whole numbers.
            return np.stack(
                [
                    np.rint((xs / scale - left - 0.5) * _ONE),
                    np.rint((ys / scale - top - 0.5) * _ONE),
                ],
                axis=-1,
            ).astype(np.int32)

        # Each outline a polygon, its sides at most a few pixels long. cv2's
        # own ellipse clips a shape crossing the canvas edge by where the edge
        # falls, so a large cell drawn into two tiles differed at their seam;
        # a filled polygon is drawn the same wherever the edge cuts it.
        sides = int(min(720, max(16, math.pi * _RADIUS_UM[1] / scale)))
        t = np.linspace(0.0, 2 * np.pi, sides, endpoint=False)
        cx, cy, rx, ry, angle = (cells[:, i : i + 1] for i in range(5))
        cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))

        def outlines(ax: np.ndarray, ay: np.ndarray) -> np.ndarray:
            ex, ey = ax * np.cos(t), ay * np.sin(t)
            return fixed(cx + ex * cos_a - ey * sin_a, cy + ex * sin_a + ey * cos_a)

        # The membrane at least a pixel wide at every scale.
        membrane = max(_MEMBRANE_UM, scale)
        nucleus_x = np.maximum(_NUCLEUS_MIN_UM, rx / 3)
        nucleus_y = np.maximum(_NUCLEUS_MIN_UM, ry / 3)
        # Every membrane, then every cytoplasm, then every nucleus: where
        # cells overlap a pixel shows the innermost part any of them puts
        # there, whatever order a tile meets them in. One polygon a call:
        # cv2 fills a list of polygons by parity, which would cut a hole
        # where two overlap.
        for parts, grey in (
            (outlines(rx, ry), 150),
            (outlines(rx - membrane, ry - membrane), 85),
            (outlines(nucleus_x, nucleus_y), 175),
        ):
            for outline in parts:
                cv2.fillPoly(canvas, [outline], grey, cv2.LINE_8, _SHIFT_BITS)
        # A punctum is a pixel's size at every scale: the brightest parts a
        # layer at its shipped settings shows.
        for point in fixed(puncta[:, 0], puncta[:, 1]):
            cv2.circle(
                canvas, (int(point[0]), int(point[1])), _ONE, 215, -1, cv2.LINE_8, _SHIFT_BITS
            )
        # The objective's own resolution: even in focus an edge is not a step.
        return cv2.GaussianBlur(canvas, (0, 0), 0.8)

    def _cells(self, gx: int, gy: int) -> tuple[np.ndarray, np.ndarray]:
        """The cells grid square (``gx``, ``gy``) holds, the same every time it is asked.

        Returns:
            The cells, one row each of centre x, centre y, radius x and
            radius y in um and angle in degrees; and their puncta, one row
            each of x and y in um.
        """
        key = (gx, gy)
        cached = self._squares.get(key)
        if cached is not None:
            self._squares.move_to_end(key)
            return cached
        # The generator's seed takes no negative numbers; the stage's
        # squares run both ways from 0.
        rng = np.random.default_rng([self._seed, gx + 2**31, gy + 2**31])
        n = int(rng.poisson(_CELLS_PER_GRID))
        cells = np.empty((n, 5))
        cells[:, 0] = (gx + rng.random(n)) * _GRID_UM
        cells[:, 1] = (gy + rng.random(n)) * _GRID_UM
        cells[:, 2:4] = rng.uniform(*_RADIUS_UM, size=(n, 2))
        cells[:, 4] = rng.uniform(0, 180, size=n)
        owner = np.repeat(np.arange(n), rng.integers(2, 6, size=n))
        reach = cells[owner, 2:4] / 2
        puncta = cells[owner, 0:2] + rng.uniform(-reach, reach)
        self._squares[key] = cached = (cells, puncta)
        if len(self._squares) > _SQUARES_KEPT:
            self._squares.popitem(last=False)
        return cached

    def _medium(self, rows_um: np.ndarray, cols_um: np.ndarray) -> np.ndarray:
        """The medium's unevenness, 0 to 1, at each (row, column): value noise in stage coordinates."""
        fy, fx = rows_um / _MEDIUM_UM, cols_um / _MEDIUM_UM
        iy, ix = np.floor(fy).astype(np.int64), np.floor(fx).astype(np.int64)
        ty, tx = (fy - iy)[:, None], (fx - ix)[None, :]
        iy, ix = iy[:, None], ix[None, :]
        top = self._lattice(iy, ix) * (1 - tx) + self._lattice(iy, ix + 1) * tx
        bottom = self._lattice(iy + 1, ix) * (1 - tx) + self._lattice(iy + 1, ix + 1) * tx
        return (top * (1 - ty) + bottom * ty).astype(np.float32)

    def _lattice(self, iy: np.ndarray, ix: np.ndarray) -> np.ndarray:
        """A value 0 to 1 for each lattice point, fixed by the point and the seed."""
        h = (
            iy.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
            ^ ix.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)
            ^ np.uint64(self._seed)
        )
        # splitmix64's finaliser: neighbouring points get unrelated values.
        h = (h ^ (h >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        h = (h ^ (h >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        h = h ^ (h >> np.uint64(31))
        return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)
