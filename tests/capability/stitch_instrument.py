"""The stitch-registration instrument: every seam, every rule, every placement, against labels.

A tiled set's stitch is a choice of two things: which seam matches to
believe (a match rule) and where each tile then goes (a placement). This
module scores both against labels: for every seam, the offset its content
really asks for, proposed by two methods of different principle and kept
only where they agree, fit together and agree across specimens, then
confirmed by a person looking at the seam.

It runs the shipped estimators and the shipped flood as production runs
them, at production's own window, and changes nothing about a stitch. The
two placements no production code computes yet (the joint solve and the
fitted stage) are computed here; one the design adopts moves into
production and is called from there.

The probe that drives it over a corpus of run folders is
``postproc_probe_12_stitch_registration.py``; this module needs no scope
and writes nothing itself.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
from collections.abc import Callable

import cv2
import numpy as np

from modules.stitch_algorithms import (
    _correction_range,
    _grid_keys,
    _overlap_views,
    align_tile_positions,
    estimate_overlap_offset,
    estimate_phase_offset,
    stitch_registered_tiles,
)


def _default(function, name):
    return inspect.signature(function).parameters[name].default


# The window and overlap floor every production stitch runs at: no caller
# passes either, so stitch_registered_tiles' defaults reach both estimators.
PRODUCTION_WINDOW_PX: int = _default(stitch_registered_tiles, 'max_correction_px')
MIN_OVERLAP_PX: int = _default(stitch_registered_tiles, 'min_overlap_px')

# How far the proposers look past the production window.
PROPOSER_WINDOW_PX = 60
# The box low-pass a strip's high-pass subtracts, in pixels.
HIGHPASS_BOX_PX = 15
# Two proposers agree when every axis is within this many pixels.
PROPOSERS_AGREE_PX = 1.0
# A labelled seam's residual in the fit over all its set's seams. A label
# is held to 1 px, the scoring's own bound: production places tiles in
# whole pixels, so a finer bound refuses labels no placement could use.
FIT_RESIDUAL_PX = 1.0
# A seam's proposal against the same move on the other specimens.
ACROSS_SPECIMENS_PX = 2.0
# A rule's answer is right when every axis is within this of the label.
RIGHT_WITHIN_PX = 1.0

ESTIMATORS: dict[str, Callable[..., tuple[int, int, float]]] = {
    'quality': estimate_overlap_offset,
    'fast': estimate_phase_offset,
}


# ---------------------------------------------------------------- seams


@dataclasses.dataclass(frozen=True)
class Answer:
    """One estimator's answer on one seam, as production returns it."""

    dx: int
    dy: int
    score: float
    # Quality: the best lay on its window's edge (the window may be too small).
    border: bool = False
    # Fast: an answer of exactly (0, 0) is a true zero or a shift outside
    # the window; the phase estimator returns both the same way.
    zero_or_out: bool = False


@dataclasses.dataclass(frozen=True)
class Seam:
    """Two lattice neighbours: ``moving`` sits right of or below ``reference``."""

    reference: int
    moving: int
    nominal_dx: int
    nominal_dy: int

    @property
    def key(self) -> str:
        return f'{self.reference}->{self.moving}'


def seams(tiles: list[dict]) -> list[Seam]:
    """Every adjacent pair of the lattice, not only the ones the flood walks."""
    x_values, y_values, by_position = _grid_keys(tiles)
    found = []
    for (x, y), idx in sorted(by_position.items(), key=lambda item: item[1]):
        xi = x_values.index(x)
        yi = y_values.index(y)
        neighbours = []
        if xi + 1 < len(x_values):
            neighbours.append((x_values[xi + 1], y))
        if yi + 1 < len(y_values):
            neighbours.append((x, y_values[yi + 1]))
        for nx, ny in neighbours:
            nidx = by_position.get((nx, ny))
            if nidx is not None:
                found.append(Seam(idx, nidx, nx - x, ny - y))
    return found


def answer(
    estimator: str, tiles: list[dict], seam: Seam, window: int = PRODUCTION_WINDOW_PX
) -> Answer:
    kwargs = {
        'reference': tiles[seam.reference]['tile'],
        'moving': tiles[seam.moving]['tile'],
        'nominal_dx': seam.nominal_dx,
        'nominal_dy': seam.nominal_dy,
        'max_correction_px': window,
        'min_overlap_px': MIN_OVERLAP_PX,
    }
    return _answer(estimator, kwargs, *ESTIMATORS[estimator](**kwargs))


def _answer(estimator: str, kwargs: dict, dx: int, dy: int, score: float) -> Answer:
    """An estimator's return with what the harness can read off it: Quality's border, Fast's zero."""
    if score <= -1.0:
        return Answer(dx, dy, score)
    if estimator == 'fast':
        return Answer(dx, dy, score, zero_or_out=(dx, dy) == (0, 0))
    h, w = kwargs['reference'].shape[:2]
    window, floor = kwargs['max_correction_px'], kwargs['min_overlap_px']
    rx = _correction_range(kwargs['nominal_dx'], w, window, floor)
    ry = _correction_range(kwargs['nominal_dy'], h, window, floor)
    return Answer(dx, dy, score, border=bool(rx and ry and (dx in rx or dy in ry)))


# ---------------------------------------------------------------- rules


@dataclasses.dataclass(frozen=True)
class Rule:
    """A match rule: which estimator it reads, and whether it accepts an answer."""

    name: str
    estimator: str
    accepts: Callable[[Answer], bool]


RULES: tuple[Rule, ...] = (
    Rule('today, Quality', 'quality', lambda a: True),
    Rule('today, Fast Preview', 'fast', lambda a: True),
    Rule('phase response >= 0.2', 'fast', lambda a: a.score >= 0.2),
    Rule('stage only', 'quality', lambda a: False),
)


# ---------------------------------------------------------------- labels


@dataclasses.dataclass(frozen=True)
class Label:
    """What a seam's content really asks for.

    ``kind`` is ``offset`` (``dx``/``dy`` sub-pixel, from stage), ``no_content``
    (a person marked nothing to match; the right answer is to decline) or
    ``unknown`` (``why`` says what failed; the seam is named, never scored).
    """

    kind: str
    dx: float = 0.0
    dy: float = 0.0
    why: str = ''


def _band(tile: np.ndarray, seam: Seam, side: str, margin: int) -> tuple[slice, slice]:
    """The rows or columns of a tile the seam's overlap can reach, plus a margin."""
    h, w = tile.shape[:2]
    if seam.nominal_dx:
        overlap = w - abs(seam.nominal_dx)
        start = w - overlap - margin if side == 'reference' else 0
        return slice(None), slice(max(0, start), min(w, start + overlap + 2 * margin))
    overlap = h - abs(seam.nominal_dy)
    start = h - overlap - margin if side == 'reference' else 0
    return slice(max(0, start), min(h, start + overlap + 2 * margin)), slice(None)


def _highpassed(tile: np.ndarray, seam: Seam, side: str) -> np.ndarray:
    """The tile with only its overlap band high-passed and the rest zero.

    The high-pass is applied to the strip, never the whole tile (whole-tile
    high-pass biased smooth content 1-6 px in the plan's earlier passes).
    """
    gray = (
        tile.astype(np.float32)
        if tile.ndim == 2
        else cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY).astype(np.float32)
    )
    out = np.zeros_like(gray)
    rows, cols = _band(gray, seam, side, PROPOSER_WINDOW_PX + MIN_OVERLAP_PX)
    strip = gray[rows, cols]
    out[rows, cols] = strip - cv2.blur(strip, (HIGHPASS_BOX_PX, HIGHPASS_BOX_PX))
    return out


def propose_ncc(tiles: list[dict], seam: Seam) -> tuple[float, float] | None:
    """Proposer one: NCC on high-passed strips, through the shipped matcher, past its window."""
    dx, dy, score = estimate_overlap_offset(
        reference=_highpassed(tiles[seam.reference]['tile'], seam, 'reference'),
        moving=_highpassed(tiles[seam.moving]['tile'], seam, 'moving'),
        nominal_dx=seam.nominal_dx,
        nominal_dy=seam.nominal_dy,
        max_correction_px=PROPOSER_WINDOW_PX,
        min_overlap_px=MIN_OVERLAP_PX,
    )
    return None if score <= -1.0 else (float(dx), float(dy))


def propose_phase(tiles: list[dict], seam: Seam) -> tuple[float, float] | None:
    """Proposer two: phase correlation of the high-passed overlap, sub-pixel and unbounded."""
    views = _overlap_views(
        _highpassed(tiles[seam.reference]['tile'], seam, 'reference'),
        _highpassed(tiles[seam.moving]['tile'], seam, 'moving'),
        dx=seam.nominal_dx,
        dy=seam.nominal_dy,
    )
    if views is None:
        return None
    ref_view, mov_view = views
    if float(ref_view.std()) <= 1e-6 or float(mov_view.std()) <= 1e-6:
        return None
    window = cv2.createHanningWindow((ref_view.shape[1], ref_view.shape[0]), cv2.CV_32F)
    (sx, sy), _ = cv2.phaseCorrelate(ref_view, mov_view, window)
    return -float(sx), -float(sy)


def fit_positions(
    n_tiles: int,
    anchor: int,
    stage: list[tuple[float, float]],
    relations: dict[Seam, tuple[float, float]],
) -> tuple[list[tuple[float, float] | None], dict[Seam, float]]:
    """Least-squares tile positions from seam relations, each a correction from stage.

    Every component of the seams is anchored at its first tile's stage
    position (the anchor's component at the anchor); a tile no relation
    reaches is None. Returns the positions and each relation's residual.
    """
    parent = list(range(n_tiles))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for seam in relations:
        parent[root(seam.reference)] = root(seam.moving)
    positions: list[tuple[float, float] | None] = [None] * n_tiles
    residuals: dict[Seam, float] = {}
    members: dict[int, list[int]] = {}
    for i in range(n_tiles):
        members.setdefault(root(i), []).append(i)
    for group in members.values():
        group_seams = [s for s in relations if s.reference in group]
        if not group_seams:
            continue
        fixed = anchor if anchor in group else min(group)
        index = {t: k for k, t in enumerate(t for t in group if t != fixed)}
        for axis in (0, 1):
            rows, rhs = [], []
            for s in group_seams:
                row = np.zeros(len(index))
                b = (s.nominal_dx, s.nominal_dy)[axis] + relations[s][axis]
                for t, sign in ((s.moving, 1.0), (s.reference, -1.0)):
                    if t == fixed:
                        b -= sign * stage[fixed][axis]
                    else:
                        row[index[t]] = sign
                rows.append(row)
                rhs.append(b)
            solved = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0] if index else []
            for t in group:
                value = stage[fixed][axis] if t == fixed else float(solved[index[t]])
                current = positions[t] or (0.0, 0.0)
                positions[t] = (value, current[1]) if axis == 0 else (current[0], value)
        for s in group_seams:
            pa, pb = positions[s.reference], positions[s.moving]
            ex = pb[0] - pa[0] - (s.nominal_dx + relations[s][0])
            ey = pb[1] - pa[1] - (s.nominal_dy + relations[s][1])
            residuals[s] = max(residuals.get(s, 0.0), math.hypot(ex, ey))
    return positions, residuals


def fit_check(
    n_tiles: int,
    anchor: int,
    stage: list[tuple[float, float]],
    relations: dict[Seam, tuple[float, float]],
) -> tuple[dict[Seam, tuple[float, float]], dict[Seam, float]]:
    """Check (b): the relations that fit together, and each one set aside with its residual.

    The worst relation over ``FIT_RESIDUAL_PX`` is set aside and the rest
    refit, until every one kept is under it.
    """
    kept = dict(relations)
    set_aside: dict[Seam, float] = {}
    while True:
        _, residuals = fit_positions(n_tiles, anchor, stage, kept)
        worst = [s for s, r in residuals.items() if r > FIT_RESIDUAL_PX]
        if not worst:
            return kept, set_aside
        bad = max(worst, key=residuals.get)
        set_aside[bad] = residuals[bad]
        del kept[bad]


def propose_labels(tiles: list[dict]) -> dict[Seam, Label]:
    """Each seam's proposed label within one set: checks (a) and (b) of the plan.

    (a) the two proposers agree; (b) the agreed seams fit together, every
    residual under ``FIT_RESIDUAL_PX`` (``fit_check``). Check (c), across specimens, is ``across_specimens``.
    """
    labels: dict[Seam, Label] = {}
    agreed: dict[Seam, tuple[float, float]] = {}
    for seam in seams(tiles):
        a, b = propose_ncc(tiles, seam), propose_phase(tiles, seam)
        if a is None or b is None:
            labels[seam] = Label('unknown', why='a proposer found nothing to correlate')
        elif max(abs(a[0] - b[0]), abs(a[1] - b[1])) > PROPOSERS_AGREE_PX:
            labels[seam] = Label(
                'unknown', why=f'proposers disagree: NCC {a}, phase ({b[0]:.2f}, {b[1]:.2f})'
            )
        else:
            agreed[seam] = b
    fitted, set_aside = fit_check(len(tiles), anchor_of(tiles), place_stage(tiles), agreed)
    for seam, residual in set_aside.items():
        labels[seam] = Label('unknown', why=f'fit residual {residual:.2f} px')
    for seam, (dx, dy) in fitted.items():
        labels[seam] = Label('offset', dx, dy)
    return labels


def across_specimens(sets: dict[str, dict[str, Label]]) -> dict[str, dict[str, Label]]:
    """Check (c): a seam's offset against the same move on the other specimens.

    ``sets`` maps a set's name to its labels by seam key, for sets of one
    grid at one overlap. A seam more than ``ACROSS_SPECIMENS_PX`` from the
    median of the others is unknown. Fewer than two others: unchecked.
    """
    out = {name: dict(labels) for name, labels in sets.items()}
    for name, labels in sets.items():
        for key, label in labels.items():
            if label.kind != 'offset':
                continue
            others = [
                other[key]
                for other_name, other in sets.items()
                if other_name != name and key in other and other[key].kind == 'offset'
            ]
            if len(others) < 2:
                continue
            mx = float(np.median([o.dx for o in others]))
            my = float(np.median([o.dy for o in others]))
            if max(abs(label.dx - mx), abs(label.dy - my)) > ACROSS_SPECIMENS_PX:
                out[name][key] = Label(
                    'unknown',
                    why=f"({label.dx:.2f}, {label.dy:.2f}) against the other specimens' ({mx:.2f}, {my:.2f})",
                )
    return out


# ---------------------------------------------------------------- scoring


OUTCOMES = ('right', 'wrong', 'missed', 'declined right')


def outcome(rule: Rule, ans: Answer, label: Label) -> str | None:
    """One rule's outcome on one seam; None on an ``unknown`` label, which is never scored."""
    accepted = rule.accepts(ans)
    if label.kind == 'unknown':
        return None
    if label.kind == 'no_content':
        return 'wrong' if accepted else 'declined right'
    if not accepted:
        return 'missed'
    off = max(abs(ans.dx - label.dx), abs(ans.dy - label.dy))
    return 'right' if off <= RIGHT_WITHIN_PX else 'wrong'


@dataclasses.dataclass
class SeamScore:
    counts: dict[str, int]
    border: int
    zero_or_out: int
    unknown: list[str]


def score_seams(rule: Rule, answers: dict[Seam, Answer], labels: dict[Seam, Label]) -> SeamScore:
    counts = dict.fromkeys(OUTCOMES, 0)
    unknown = []
    border = zero_or_out = 0
    for seam, ans in answers.items():
        result = outcome(rule, ans, labels[seam])
        if result is None:
            unknown.append(f'{seam.key}: {labels[seam].why}')
            continue
        counts[result] += 1
        border += ans.border
        zero_or_out += ans.zero_or_out
    return SeamScore(counts, border, zero_or_out, unknown)


# ---------------------------------------------------------------- placements


def anchor_of(tiles: list[dict]) -> int:
    """The flood's anchor: the tile at the lattice's first x and first y."""
    x_values, y_values, by_position = _grid_keys(tiles)
    return by_position[(x_values[0], y_values[0])]


def place_flood(rule: Rule, tiles: list[dict]) -> list[tuple[float, float]]:
    """Production's flood with the rule applied: a declined seam answers (0, 0, score).

    Under today's rule this is the shipped stitch. Under any other rule a
    declined tile moves rigidly with its parent's accumulated correction;
    it is not stage placement.
    """
    estimator = ESTIMATORS[rule.estimator]

    def ruled(**kwargs):
        dx, dy, score = estimator(**kwargs)
        accepted = rule.accepts(_answer(rule.estimator, kwargs, dx, dy, score))
        return (dx, dy, score) if accepted else (0, 0, score)

    placed = align_tile_positions(
        tiles,
        max_correction_px=PRODUCTION_WINDOW_PX,
        min_overlap_px=MIN_OVERLAP_PX,
        estimator=ruled,
    )
    return [(float(t['registered_x_px']), float(t['registered_y_px'])) for t in placed]


def place_stage(tiles: list[dict]) -> list[tuple[float, float]]:
    return [(float(t['x_px']), float(t['y_px'])) for t in tiles]


def place_joint(
    rule: Rule, tiles: list[dict], answers: dict[Seam, Answer]
) -> list[tuple[float, float]]:
    """Least squares over the seams the rule accepted; a tile no accepted seam reaches stays at stage."""
    stage = place_stage(tiles)
    accepted = {s: (float(a.dx), float(a.dy)) for s, a in answers.items() if rule.accepts(a)}
    positions, _ = fit_positions(len(tiles), anchor_of(tiles), stage, accepted)
    return [p if p is not None else stage[i] for i, p in enumerate(positions)]


@dataclasses.dataclass(frozen=True)
class StageModel:
    """Stage corrected by a linear map fitted from seams: scale per axis and rotation."""

    matrix: tuple[tuple[float, float], tuple[float, float]]
    seams_fitted: int

    @property
    def scale_x(self) -> float:
        return math.hypot(self.matrix[0][0], self.matrix[1][0])

    @property
    def scale_y(self) -> float:
        return math.hypot(self.matrix[0][1], self.matrix[1][1])

    @property
    def rotation_deg(self) -> float:
        return math.degrees(math.atan2(self.matrix[1][0], self.matrix[0][0]))


def fit_stage_model(relations: dict[Seam, tuple[float, float]]) -> StageModel | None:
    """The linear map taking each seam's nominal step to the step its content asks for."""
    if len(relations) < 2:
        return None
    nominal = np.array([(s.nominal_dx, s.nominal_dy) for s in relations], dtype=float)
    actual = nominal + np.array(list(relations.values()), dtype=float)
    if np.linalg.matrix_rank(nominal) < 2:
        return None
    solved = np.linalg.lstsq(nominal, actual, rcond=None)[0].T
    return StageModel(((solved[0, 0], solved[0, 1]), (solved[1, 0], solved[1, 1])), len(relations))


def place_fitted_stage(
    rule: Rule, tiles: list[dict], answers: dict[Seam, Answer]
) -> tuple[list[tuple[float, float]], StageModel | None]:
    """Stage corrected by the map fitted from the seams the rule accepted."""
    stage = place_stage(tiles)
    model = fit_stage_model(
        {s: (float(a.dx), float(a.dy)) for s, a in answers.items() if rule.accepts(a)}
    )
    if model is None:
        return stage, None
    ax, ay = stage[anchor_of(tiles)]
    m = np.array(model.matrix)
    return [
        tuple(np.array([ax, ay]) + m @ (np.array(p) - np.array([ax, ay]))) for p in stage
    ], model


def worst_tile_error(
    placed: list[tuple[float, float]], truth: list[tuple[float, float] | None]
) -> tuple[float, int]:
    """The largest distance from a tile's labelled position, and how many tiles had one."""
    errors = [
        math.hypot(p[0] - t[0], p[1] - t[1])
        for p, t in zip(placed, truth, strict=True)
        if t is not None
    ]
    return (max(errors) if errors else math.nan), len(errors)


def labelled_positions(
    tiles: list[dict], labels: dict[Seam, Label]
) -> list[tuple[float, float] | None]:
    relations = {s: (lab.dx, lab.dy) for s, lab in labels.items() if lab.kind == 'offset'}
    positions, _ = fit_positions(len(tiles), anchor_of(tiles), place_stage(tiles), relations)
    return positions


# ---------------------------------------------------------------- crop sheets


def crop_sheet(
    tiles: list[dict],
    seam_answers: dict[Seam, dict[str, Answer]],
    labels: dict[Seam, Label],
    half: int = 90,
) -> np.ndarray:
    """One image of every seam: a row each, the overlap blended at stage, at the label, at each estimator's answer.

    Each panel is the mid-seam window of the reference with the moving tile
    laid over it at that offset, as a red/cyan pair: a seam that lines up
    reads grey, a misaligned one shows coloured fringes.
    """
    rows = []
    for seam, answers in seam_answers.items():
        label = labels[seam]
        offsets = [('stage', (0.0, 0.0))]
        if label.kind == 'offset':
            offsets.append(('label', (label.dx, label.dy)))
        offsets += [(name, (float(a.dx), float(a.dy))) for name, a in answers.items()]
        panels = [_seam_panel(tiles, seam, off, half, title) for title, off in offsets]
        while len(panels) < 4:
            panels.append(np.zeros_like(panels[0]))
        row = np.hstack(panels)
        cv2.putText(
            row,
            f'{seam.key}  {label.kind} {label.why}'[:90],
            (4, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
        rows.append(row)
    return np.vstack(rows)


def _seam_panel(tiles, seam, offset, half, title):
    ref = tiles[seam.reference]['tile']
    mov = tiles[seam.moving]['tile']
    ref = ref if ref.ndim == 2 else cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    mov = mov if mov.ndim == 2 else cv2.cvtColor(mov, cv2.COLOR_RGB2GRAY)
    h, w = ref.shape
    dx = seam.nominal_dx + offset[0]
    dy = seam.nominal_dy + offset[1]
    if seam.nominal_dx:
        cx, cy = w - (w - seam.nominal_dx) // 2, h // 2
    else:
        cx, cy = w // 2, h - (h - seam.nominal_dy) // 2
    size = 2 * half
    ref_crop = ref[cy - half : cy + half, cx - half : cx + half].astype(np.float32)
    shift = np.float32([[1, 0, -(cx - half - dx)], [0, 1, -(cy - half - dy)]])
    mov_crop = cv2.warpAffine(mov.astype(np.float32), shift, (size, size))
    ref8 = cv2.normalize(ref_crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mov8 = cv2.normalize(mov_crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    panel = cv2.merge([mov8, mov8, ref8])
    cv2.putText(
        panel,
        f'{title} ({offset[0]:.1f},{offset[1]:.1f})',
        (4, size - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 0),
        1,
    )
    return panel


# ---------------------------------------------------------------- synthetic sets


SCENES = (
    'texture',
    'featureless shading',
    'periodic ruling',
    'saturated band',
    'straight edge',
    'row pattern',
    'sparse cells',
)


def synthetic_set(
    scene: str,
    overlap: float,
    *,
    tile: int = 600,
    grid: int = 3,
    step_scale: tuple[float, float] = (1.006, 1.004),
    rotation_deg: float = -0.13,
    noise: float = 2.0,
    seed: int = 0,
) -> tuple[list[dict], dict[Seam, Label]]:
    """A grid cut from one scene at true positions off stage, with independent noise per tile.

    The true positions are stage under a scale and rotation like the 2026-11-03
    corpus's measured pattern, so a rule is tested where it will run. Every
    tile carries its own noise: no two share pixels. The labels are the
    construction (``no_content`` for the featureless scene).
    """
    rng = np.random.default_rng(seed)
    step = round(tile * (1 - overlap))
    span = step * (grid - 1) + tile + 200
    scene_img = _scene(scene, span, rng)
    theta = math.radians(rotation_deg)
    m = np.array(
        [
            [step_scale[0] * math.cos(theta), -step_scale[1] * math.sin(theta)],
            [step_scale[0] * math.sin(theta), step_scale[1] * math.cos(theta)],
        ]
    )
    tiles, truth = [], []
    for gy in range(grid):
        for gx in range(grid):
            nominal = np.array([gx * step, gy * step], dtype=float)
            actual = m @ nominal + 50
            shift = np.float32([[1, 0, -actual[0]], [0, 1, -actual[1]]])
            cut = cv2.warpAffine(scene_img, shift, (tile, tile), flags=cv2.INTER_LINEAR)
            cut = np.clip(cut + rng.normal(0, noise, cut.shape), 0, 255).astype(np.uint8)
            tiles.append({'tile': cut, 'x_px': int(nominal[0]), 'y_px': int(nominal[1])})
            truth.append(actual - 50)
    labels = {}
    for seam in seams(tiles):
        if scene == 'featureless shading':
            labels[seam] = Label('no_content')
            continue
        d = truth[seam.moving] - truth[seam.reference]
        labels[seam] = Label('offset', float(d[0] - seam.nominal_dx), float(d[1] - seam.nominal_dy))
    return tiles, labels


def _scene(scene: str, size: int, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    shading = 120 + 40 * np.cos((xx - size / 2) / size * math.pi) * np.cos(
        (yy - size / 2) / size * math.pi
    )
    if scene == 'texture':
        img = cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32), (0, 0), 3)
        return (shading + 40 * img / img.std()).astype(np.float32)
    if scene == 'featureless shading':
        return shading.astype(np.float32)
    if scene == 'periodic ruling':
        return (shading + 50 * (np.sin(xx * 2 * math.pi / 24) > 0)).astype(np.float32)
    if scene == 'saturated band':
        img = cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32), (0, 0), 3)
        out = shading + 25 * img / img.std()
        out[:, size // 3 : size // 2] = 255
        return out.astype(np.float32)
    if scene == 'straight edge':
        return (shading + 60 * (xx + 0.3 * yy > size * 0.6)).astype(np.float32)
    if scene == 'row pattern':
        return (30 + 6 * (np.arange(size)[:, None] % 2) + 0 * xx).astype(np.float32)
    if scene == 'sparse cells':
        out = np.full((size, size), 40, np.float32)
        for _ in range(max(4, size * size // 60000)):
            cx, cy = rng.integers(0, size, 2)
            cv2.circle(out, (int(cx), int(cy)), int(rng.integers(6, 14)), 160, -1)
        return cv2.GaussianBlur(out, (0, 0), 1.5)
    raise ValueError(f'no scene named {scene!r}; the scenes are {SCENES}')
