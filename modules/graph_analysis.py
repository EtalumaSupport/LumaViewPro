# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A graph of cell-count results: the figure, its trendlines, and the fit behind each one.

A fit the data cannot take is refused in words that say which values and how
many, never drawn as an arbitrary curve: a log-transformed kind cannot take a
value of 0 or below, and a polynomial needs more distinct X values than its
degree.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib.dates import ConciseDateFormatter
from matplotlib.figure import Figure

from modules.closed_sets import refuse_unless_offered
from modules.exceptions import PostProcessingOperation, PostProcessingRefusedError

TRENDLINE_OPERATION = PostProcessingOperation.TRENDLINE


class Trendline(enum.StrEnum):
    """The kind of curve a graph's trendline is fitted as."""

    LINEAR = 'Linear'
    QUADRATIC = 'Quadratic'
    EXPONENTIAL = 'Exponential'
    POWER = 'Power'
    LOGARITHMIC = 'Logarithmic'


# Power and Logarithmic take the log of X; a time is measured from the first
# one, which is 0, so a time X takes only the kinds that use X as it is.
_TIME_X_KINDS = (Trendline.LINEAR, Trendline.QUADRATIC, Trendline.EXPONENTIAL)

# The distinct X values each kind needs: more than its degree.
_DISTINCT_X_NEEDED = {
    Trendline.LINEAR: 2,
    Trendline.QUADRATIC: 3,
    Trendline.EXPONENTIAL: 2,
    Trendline.POWER: 2,
    Trendline.LOGARITHMIC: 2,
}


@dataclass(frozen=True)
class TrendlineFit:
    """A fitted curve, ready to draw.

    Attributes:
        kind: The kind of curve.
        x: The X values in ascending order, in the X column's own type.
        y: The curve's value at each of them.
    """

    kind: Trendline
    x: pd.Series
    y: np.ndarray


def trendline_kinds(x: pd.Series | None, y: pd.Series | None) -> tuple[Trendline, ...]:
    """The trendline kinds a graph of *y* against *x* can draw, by the columns' types.

    An axis not chosen yet (None) rules no kind out, so a form offers each
    kind until an axis it cannot take is chosen.
    """
    if y is not None and not pd.api.types.is_numeric_dtype(y):
        return ()
    if x is None:
        return tuple(Trendline)
    if pd.api.types.is_datetime64_any_dtype(x):
        return _TIME_X_KINDS
    if pd.api.types.is_numeric_dtype(x):
        return tuple(Trendline)
    return ()


def fit_trendline(kind: Trendline, x: pd.Series, y: pd.Series) -> TrendlineFit:
    """Fit a *kind* trendline to *y* against *x*.

    A time X is measured in seconds from its first value.

    Raises:
        ArgumentRefusedError: ``'not_offered'``, naming ``trendline``: the
            kind is not one this pair of columns takes, offering those it
            takes (``trendline_kinds``).
        PostProcessingRefusedError: operation ``'Trendline'``, reason
            ``fit_impossible``; the values cannot give that curve. The
            message names the values and how many.
        numpy.linalg.LinAlgError: a fit the checks admitted did not converge;
            a fault, not a refusal.
    """

    def refuse(problem: str) -> PostProcessingRefusedError:
        return PostProcessingRefusedError(
            operation=TRENDLINE_OPERATION,
            reason='fit_impossible',
            message=f'The {kind} trendline of {y.name} against {x.name} cannot be drawn: {problem}.',
        )

    kind = refuse_unless_offered(Trendline, kind, 'trendline', within=trendline_kinds(x, y))
    order = np.argsort(x.to_numpy(), kind='stable')
    x = x.iloc[order]
    y = y.iloc[order]
    if pd.api.types.is_datetime64_any_dtype(x):
        xs = (x - x.min()).dt.total_seconds().to_numpy(dtype=float)
    else:
        xs = x.to_numpy(dtype=float)
    ys = y.to_numpy(dtype=float)

    def count_where(values: np.ndarray, bad: np.ndarray, name: str, what: str) -> None:
        count = int(bad.sum())
        if count:
            verb = 'is' if count == 1 else 'are'
            raise refuse(f'{count} of the {len(values)} {name} values {verb} {what}')

    count_where(xs, ~np.isfinite(xs), x.name, 'missing or not a number')
    count_where(ys, ~np.isfinite(ys), y.name, 'missing or not a number')
    if kind in (Trendline.EXPONENTIAL, Trendline.POWER):
        count_where(
            ys, ys <= 0, y.name, f'0 or below, and the {kind} curve needs every one above 0'
        )
    if kind in (Trendline.POWER, Trendline.LOGARITHMIC):
        count_where(
            xs, xs <= 0, x.name, f'0 or below, and the {kind} curve needs every one above 0'
        )
    distinct = len(np.unique(xs))
    if distinct < _DISTINCT_X_NEEDED[kind]:
        raise refuse(
            f'{x.name} has {distinct} distinct value(s), and the {kind} curve needs at least '
            f'{_DISTINCT_X_NEEDED[kind]}'
        )

    if kind == Trendline.LINEAR:
        curve = np.poly1d(np.polyfit(xs, ys, 1))(xs)
    elif kind == Trendline.QUADRATIC:
        curve = np.poly1d(np.polyfit(xs, ys, 2))(xs)
    elif kind == Trendline.EXPONENTIAL:
        curve = np.exp(np.poly1d(np.polyfit(xs, np.log(ys), 1))(xs))
    elif kind == Trendline.POWER:
        curve = np.exp(np.poly1d(np.polyfit(np.log(xs), np.log(ys), 1))(np.log(xs)))
    else:
        curve = np.poly1d(np.polyfit(np.log(xs), ys, 1))(np.log(xs))
    return TrendlineFit(kind=kind, x=x, y=curve)


def draw_graph(
    table: pd.DataFrame,
    x: str | None,
    y: str | None,
    trendline: TrendlineFit | None,
    *,
    x_label: str,
    y_label: str,
    title: str,
) -> Figure:
    """Draw *y* against *x* from *table*, with *trendline* over it when one is given.

    Built without pyplot, so no figure is held after its last reference
    goes and two threads can each draw their own. With *x* or *y* None it
    is the empty frame, labelled, that a graph shows before its axes are
    chosen.
    """
    figure = Figure()
    axes = figure.add_subplot()
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_title(title)
    if x is None or y is None:
        axes.scatter([], [])
        return figure
    axes.scatter(table[x], table[y])
    if pd.api.types.is_datetime64_any_dtype(table[x]):
        axes.xaxis.set_major_formatter(ConciseDateFormatter(axes.xaxis.get_major_locator()))
    if trendline is not None:
        axes.plot(trendline.x, trendline.y, 'r--')
    return figure
