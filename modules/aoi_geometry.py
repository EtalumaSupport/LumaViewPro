# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Camera-agnostic AOI geometry for oversize-then-crop framing.

A sensor can only set its area-of-interest on a coarse alignment grid, so a
requested frame size that is not a multiple of the grid step cannot be hit
exactly. Rounding the AOI DOWN (the historical behavior) silently delivers a
smaller image than asked. These helpers instead round the AOI UP to the next
legal size and report the centered sub-rectangle to crop back to the exact
request, so the consumer always receives precisely what it asked for -- centered
on the sensor's optical axis rather than on whatever offset the hardware
defaulted to.

Pure functions only -- no SDK, no hardware, no driver state -- so the framing
math is unit-testable without a camera. The driver supplies the sensor's grid
step, offset increment, current max AOI (which shrinks with binning), and the
per-unit optical-center bias; these return the AOI size to set, the hardware
offset, and the crop window.
"""

from dataclasses import dataclass


def ceil_to(value: int, step: int, base: int = 0) -> int:
    """Smallest legal grid value (``base + k*step``, k >= 0) that is >= ``value``.

    ``base`` is the grid phase -- a hardware AOI node's Minimum, whose legal set
    is ``Min + k*Inc``, not plain multiples of ``Inc``. The two coincide only
    when ``Min % Inc == 0``; they diverge whenever a node reports a Minimum off
    the increment grid (e.g. a binned sensor height, Min=418 Inc=4), and a
    plain-multiple snap then lands off the legal grid and the SDK rejects it.
    ``base`` defaults to 0 (the grid through the origin). ``step`` <= 1 is a no-op.
    """
    if step <= 1:
        return int(value)
    n = (int(value) - base + step - 1) // step
    return base + max(n, 0) * step


def floor_to(value: int, step: int, base: int = 0) -> int:
    """Largest legal grid value (``base + k*step``, k >= 0) that is <= ``value``.

    Never below the grid floor: ``base`` when a phase is supplied, else ``step``
    itself (a zero-size AOI is never legal). See ``ceil_to`` for ``base``.
    """
    if step <= 1:
        return int(value)
    snapped = base + ((int(value) - base) // step) * step
    return max(snapped, base if base else step)


def _snap(value: int, step: int, base: int = 0) -> int:
    """Round ``value`` down to a legal grid value (``base + k*step``).

    Used for the hardware offset grid; ``base`` is the offset node's Minimum so
    the snapped offset stays on the legal ``Min + k*Inc`` set. See ``ceil_to``.
    """
    if step <= 1:
        return int(value)
    n = (int(value) - base) // step
    return base + max(n, 0) * step


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class SensorOrientation:
    """The flip/rotation that maps the optical-center measurement frame to the
    delivered array frame.

    The optical-center offset is measured in software that applies no flips,
    while the live camera enables a horizontal sensor reverse (and, depending on
    how the sensor is mounted in the system, possibly a vertical reverse or a 90
    degree transpose). The flags here are NOT necessarily the raw nodemap
    reverse bits: they are whatever transform the bench collimator calibration
    found makes the optical axis land at the output center, which folds in the
    measurement software's own sign convention. Re-run that calibration if the
    sensor orientation changes; geometric centering is unaffected either way.
    """

    reverse_x: bool = False
    reverse_y: bool = False
    transpose: bool = False


def reorient_image_center(dx: int, dy: int, orientation: SensorOrientation) -> tuple[int, int]:
    """Express a measured optical-center offset in the delivered array's frame.

    ``dx``/``dy`` are the sensor-to-optical-axis offset in pixels, measured with
    no flips. Apply the same transpose/reverses the delivered array carries so
    the returned offset can bias the AOI offset to land the optical axis at the
    output center. A 90 degree mount swaps the axes; each active reverse negates
    its axis. Pure geometry -- the calibration decides which flags are set.
    """
    x, y = int(dx), int(dy)
    if orientation.transpose:
        x, y = y, x
    if orientation.reverse_x:
        x = -x
    if orientation.reverse_y:
        y = -y
    return (x, y)


@dataclass(frozen=True)
class AoiPlan:
    """A planned oversize-then-crop AOI.

    ``acq_*`` is the AOI to set in hardware (>= the request, on the grid);
    ``offset_*`` lands the optical axis at the AOI center; ``crop_*`` is the
    centered window to slice back to exactly the requested size.
    """

    acq_width: int
    acq_height: int
    offset_x: int
    offset_y: int
    crop_x0: int
    crop_y0: int
    crop_width: int
    crop_height: int

    @property
    def needs_crop(self) -> bool:
        """Whether the crop window is a strict sub-rectangle of the acquired AOI.

        False when the window already covers the whole AOI -- the request landed
        on the alignment grid (or clamped to the sensor max), so there is no
        surplus to trim and the acquired frame IS the delivered frame. Lets a
        caller skip the crop entirely without re-deriving the "no surplus" test
        from the individual fields.
        """
        return (
            self.crop_x0 != 0
            or self.crop_y0 != 0
            or self.crop_width != self.acq_width
            or self.crop_height != self.acq_height
        )


def plan_aoi(
    target: tuple[int, int],
    step: tuple[int, int],
    max_size: tuple[int, int],
    offset_step: tuple[int, int],
    size_min: tuple[int, int] = (0, 0),
    offset_min: tuple[int, int] = (0, 0),
    bias: tuple[int, int] = (0, 0),
) -> AoiPlan:
    """Plan a centered, oversize-then-crop AOI for a requested frame size.

    All sizes are in displayed (post-binning) pixels -- the space the AOI nodes
    operate in. ``target`` is the exact request; ``step`` is the alignment grid
    the AOI must land on; ``max_size`` is the current max AOI (smaller at higher
    binning); ``offset_step`` is the hardware offset increment; ``bias`` is the
    optical-center offset already reoriented into the array frame and divided by
    binning (default ``(0, 0)`` -> geometric center).

    ``size_min`` / ``offset_min`` are the AOI and offset nodes' Minimums, which
    set the grid PHASE: the legal AOI sizes are ``size_min + k*step`` and the
    legal offsets ``offset_min + k*offset_step``, NOT plain multiples (the two
    differ whenever a Minimum is off its increment grid -- e.g. a binned sensor
    height with Min=418, Inc=4). Both default to ``(0, 0)`` (the grid through the
    origin), which preserves the plain-multiple behavior for phase-0 nodes.

    The AOI rounds UP to the next legal size but never past the sensor. The
    offset and the crop window are then derived from the optical-axis position,
    with the crop computed from the ACTUAL (grid-snapped, sensor-clamped) offset
    so that any snap or clamp residual is absorbed by the crop -- the delivered
    frame stays centered on the axis to the pixel while the surplus allows, and
    degrades gracefully only as the request approaches the sensor max.

    The exact-size guarantee holds except in one unavoidable case: when the
    request is within an alignment step of the sensor max (or the max is not a
    multiple of the step), no legal AOI can supply the full width/height. There
    the AOI -- and the returned ``crop_*`` -- come back smaller than asked,
    reported truthfully rather than silently padded. Callers that require the
    exact size should compare ``crop_width``/``crop_height`` to the request.

    Raises ValueError on a non-positive target or a sensor max smaller than one
    alignment step (both indicate a mis-configured caller, not a framing choice).
    """
    wt, ht = int(target[0]), int(target[1])
    sx, sy = step
    wmax, hmax = max_size
    ox, oy = offset_step
    smin_x, smin_y = size_min
    omin_x, omin_y = offset_min
    bx, by = bias

    # The placeable width is the sensor max minus the offset floor: the AOI's
    # left edge cannot sit below the offset node's Minimum, so the largest AOI
    # that still fits at a legal offset is wmax - omin (the offset window
    # [omin, wmax-acq] is non-empty exactly when acq <= wmax - omin). With
    # offset_min 0 (every camera today) this is just wmax.
    place_w = wmax - omin_x
    place_h = hmax - omin_y

    # The smallest legal AOI is the node Minimum when it carries a phase
    # (Min > 0, e.g. a binned Height of 418), else one alignment step (the
    # historical phase-0 floor). Reject only a sensor that cannot place it --
    # a placeable extent EQUAL to the minimum is still a valid one-AOI sensor.
    min_aoi_w = smin_x if smin_x else sx
    min_aoi_h = smin_y if smin_y else sy

    if wt < 1 or ht < 1:
        raise ValueError(f'target must be positive, got {target!r}')
    if place_w < min_aoi_w or place_h < min_aoi_h:
        raise ValueError(
            f'sensor max {max_size!r} cannot place the minimum AOI '
            f'{(min_aoi_w, min_aoi_h)!r} at offset minimum {offset_min!r}'
        )

    # Largest legal AOI the sensor can supply, vs the request rounded up to the
    # grid -- whichever is smaller. min() with the legal max is what caps acq at
    # the sensor even when the max is not itself on the grid. Both snaps carry
    # the AOI Minimum as the grid phase so the value the SDK receives is legal.
    # Cap against the PLACEABLE extent (max - offset_min), not the raw max, so
    # the centered offset below is guaranteed a legal slot (offset >= omin with
    # offset + acq <= max) without a post-hoc clamp.
    legal_max_w = floor_to(place_w, sx, smin_x)
    legal_max_h = floor_to(place_h, sy, smin_y)
    acq_w = min(ceil_to(wt, sx, smin_x), legal_max_w)
    acq_h = min(ceil_to(ht, sy, smin_y), legal_max_h)

    crop_w = min(wt, acq_w)
    crop_h = min(ht, acq_h)

    # Where the optical axis sits on the sensor: geometric center shifted by the
    # (already reoriented, already binning-scaled) bias.
    axis_x = wmax // 2 + bx
    axis_y = hmax // 2 + by

    # Position the AOI to put the axis near its center -- clamp into the sensor
    # FIRST so the grid-snap can't push the offset back out of range, then snap
    # down to the offset grid (phased by the offset node's Minimum).
    off_x = _snap(_clamp(axis_x - acq_w // 2, omin_x, wmax - acq_w), ox, omin_x)
    off_y = _snap(_clamp(axis_y - acq_h // 2, omin_y, hmax - acq_h), oy, omin_y)

    # Crop relative to the ACTUAL offset, so snap/clamp residual is compensated;
    # clamp keeps the window inside the acquired AOI at the very edge.
    crop_x0 = _clamp(axis_x - off_x - crop_w // 2, 0, acq_w - crop_w)
    crop_y0 = _clamp(axis_y - off_y - crop_h // 2, 0, acq_h - crop_h)

    return AoiPlan(acq_w, acq_h, off_x, off_y, crop_x0, crop_y0, crop_w, crop_h)
