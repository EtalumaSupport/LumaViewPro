# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The cell-count method: what a count keeps, as a record that refuses what the count cannot use.

A method is built whole and checked as it is built, so a method a caller
holds -- the panel's, a script's, one sent over the wire, one read from a
file -- is one the count can use. It is changed by building another
(``dataclasses.replace``).

A method file is the method's saved form, the nested JSON LumaViewPro has
always written, plus a ``metadata`` entry that marks it as a method file;
``saved_form`` and ``from_saved_form`` are the one translation between the
two.

Light by design: no image library is imported, so the API declares the
record without loading the counter.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import NoReturn

from modules.api_surface import api_fields
from modules.exceptions import PostProcessingOperation, PostProcessingRefusedError
from modules.finite_number import is_finite_number

CELL_COUNT_OPERATION = PostProcessingOperation.CELL_COUNT

# The one segmentation the count has; a method file names it.
_SEGMENTATION_ALGORITHM = 'initial'

# Each filter's path in the saved form, by its field.
_FILTER_PATHS = {
    'area_um2': ('area',),
    'perimeter_um': ('perimeter',),
    'sphericity': ('sphericity',),
    'intensity_min_pct': ('intensity', 'min'),
    'intensity_mean_pct': ('intensity', 'mean'),
    'intensity_max_pct': ('intensity', 'max'),
}


def _refuse_method(source: str, problem: str) -> NoReturn:
    raise PostProcessingRefusedError(
        operation=CELL_COUNT_OPERATION,
        reason='method_invalid',
        message=f'{source} cannot be used: {problem}.',
    )


def _check_filters(filters: Mapping[str, object], source: str) -> None:
    """Refuse filters the count cannot use, naming the filter; *source* is what the method is called."""
    for name, bounds in filters.items():
        if not isinstance(bounds, FilterBounds):
            _refuse_method(source, f'filters.{name} must be a FilterBounds, not {bounds!r}')
        for bound in ('min', 'max'):
            value = getattr(bounds, bound)
            if value is not None and not is_finite_number(value):
                _refuse_method(
                    source, f'filters.{name}.{bound} must be a number or null, not {value!r}'
                )
        if bounds.min is not None and bounds.max is not None and bounds.min > bounds.max:
            _refuse_method(
                source, f'filters.{name}.min ({bounds.min}) is above its max ({bounds.max})'
            )


def _check_method(
    pixels_per_um: object, fluorescent_mode: object, threshold_pct: object, source: str
) -> None:
    """Refuse a scale, mode or threshold the count cannot use, naming the field.

    *source* is what the method is called: the record's own build, or the
    file it is read from, which is checked before the record is built.
    """
    if pixels_per_um is not None and (not is_finite_number(pixels_per_um) or pixels_per_um <= 0):
        _refuse_method(
            source,
            f'pixels_per_um must be a positive number of camera pixels per micron, '
            f"or null to use each image's own scale, not {pixels_per_um!r}",
        )
    if not isinstance(fluorescent_mode, bool):
        _refuse_method(source, f'fluorescent_mode must be true or false, not {fluorescent_mode!r}')
    if not is_finite_number(threshold_pct):
        _refuse_method(source, f'threshold_pct must be a number, not {threshold_pct!r}')


@api_fields('min', 'max')
@dataclasses.dataclass(frozen=True)
class FilterBounds:
    """One filter's bounds: a region outside them is not counted.

    Attributes:
        min: The lowest value kept; None is no lower bound.
        max: The highest value kept; None is no upper bound.
    """

    min: float | None = None
    max: float | None = None


@api_fields(
    'area_um2',
    'perimeter_um',
    'sphericity',
    'intensity_min_pct',
    'intensity_mean_pct',
    'intensity_max_pct',
)
@dataclasses.dataclass(frozen=True)
class CellCountFilters:
    """Which regions a count keeps, each bound checked as the filters are built.

    Attributes:
        area_um2: A region's area, in square microns.
        perimeter_um: A region's perimeter, in microns.
        sphericity: From 0 to 1.
        intensity_min_pct: A region's darkest pixel, in percent of full scale.
        intensity_mean_pct: A region's mean pixel, in percent of full scale.
        intensity_max_pct: A region's brightest pixel, in percent of full scale.

    Raises:
        PostProcessingRefusedError: reason ``method_invalid``, a filter
            that is not a ``FilterBounds``, a bound that is not a number or
            None, or a min above its max; the message names the filter.
    """

    area_um2: FilterBounds = FilterBounds()
    perimeter_um: FilterBounds = FilterBounds()
    sphericity: FilterBounds = FilterBounds(0.0, 1.0)
    intensity_min_pct: FilterBounds = FilterBounds(0, 100)
    intensity_mean_pct: FilterBounds = FilterBounds(0, 100)
    intensity_max_pct: FilterBounds = FilterBounds(0, 100)

    def __post_init__(self):
        _check_filters(
            {name: getattr(self, name) for name in _FILTER_PATHS}, 'The cell-count method'
        )


@api_fields('pixels_per_um', 'fluorescent_mode', 'threshold_pct', 'filters')
@dataclasses.dataclass(frozen=True)
class CellCountMethod:
    """How a count finds and keeps cells, checked as it is built.

    ``CellCountMethod()`` is the method LumaViewPro's panel starts from. It
    sets no scale, so each image is counted at the scale it states, and no
    size filter, so an image that states none is counted in pixels.

    Attributes:
        pixels_per_um: The scale every image is counted at, in camera
            pixels per micron; None counts each image at its own scale.
        fluorescent_mode: Bright cells on a dark field; False inverts the
            image first.
        threshold_pct: The brightness that separates a cell from the field,
            in percent of full scale.
        filters: Which regions are kept.

    Raises:
        PostProcessingRefusedError: reason ``method_invalid``, a
            *pixels_per_um* that is not a positive number or None (0 or
            below counts nothing, NaN writes NaN areas), a
            *fluorescent_mode* that is not True or False, a *threshold_pct* that
            is not a number, or *filters* that is not a
            ``CellCountFilters``; the message names the field.
    """

    pixels_per_um: float | None = None
    fluorescent_mode: bool = True
    threshold_pct: float = 20
    filters: CellCountFilters = CellCountFilters()

    def __post_init__(self):
        _check_method(
            self.pixels_per_um, self.fluorescent_mode, self.threshold_pct, 'The cell-count method'
        )
        if not isinstance(self.filters, CellCountFilters):
            _refuse_method(
                'The cell-count method', f'filters must be a CellCountFilters, not {self.filters!r}'
            )


def saved_form(method: CellCountMethod) -> dict:
    """*method* in the nested form a method file holds and the counter reads, without the file's metadata."""
    filters: dict = {}
    for name, path in _FILTER_PATHS.items():
        bounds = getattr(method.filters, name)
        holder = filters
        for key in path[:-1]:
            holder = holder.setdefault(key, {})
        holder[path[-1]] = {'min': bounds.min, 'max': bounds.max}
    return {
        'context': {
            'pixels_per_um': method.pixels_per_um,
            'fluorescent_mode': method.fluorescent_mode,
        },
        'segmentation': {
            'algorithm': _SEGMENTATION_ALGORITHM,
            'parameters': {'threshold': method.threshold_pct},
        },
        'filters': filters,
    }


def _saved_field(saved: object, path: tuple[str, ...], source: str):
    value = saved
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            _refuse_method(source, f'it has no {".".join(path)}')
        value = value[key]
    return value


def from_saved_form(saved: object, *, source: str) -> CellCountMethod:
    """The method a method file's form holds.

    Its ``metadata`` is not part of the method and is not read.

    Args:
        source: What a refusal calls the form, e.g. naming its file.

    Raises:
        PostProcessingRefusedError: reason ``method_invalid``, a field is
            missing (named by its path in the form) or holds what the
            method refuses.
    """
    fields = {
        'pixels_per_um': _saved_field(saved, ('context', 'pixels_per_um'), source),
        'fluorescent_mode': _saved_field(saved, ('context', 'fluorescent_mode'), source),
        'threshold_pct': _saved_field(saved, ('segmentation', 'parameters', 'threshold'), source),
    }
    _saved_field(saved, ('segmentation', 'algorithm'), source)
    filters = {
        name: FilterBounds(
            min=_saved_field(saved, ('filters', *path, 'min'), source),
            max=_saved_field(saved, ('filters', *path, 'max'), source),
        )
        for name, path in _FILTER_PATHS.items()
    }
    _check_method(**fields, source=source)
    _check_filters(filters, source)
    return CellCountMethod(**fields, filters=CellCountFilters(**filters))
