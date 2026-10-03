# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The trendlines a graph of results can draw, and the fit behind each one.

A fit the data cannot take is refused in words that say which values and how
many, never drawn as an arbitrary curve: a log-transformed kind cannot take a
value of 0 or below, and a polynomial needs more distinct X values than its
degree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from modules.exceptions import PostProcessingRefusedError

TRENDLINE_OPERATION = 'Trendline'

# The spinner's "no trendline" choice.
NO_TRENDLINE = 'None'

TRENDLINE_KINDS = ('Linear', 'Quadratic', 'Exponential', 'Power', 'Logarithmic')

# Power and Logarithmic take the log of X; a time is measured from the first
# one, which is 0, so a time X takes only the kinds that use X as it is.
_TIME_X_KINDS = ('Linear', 'Quadratic', 'Exponential')

# The distinct X values each kind needs: more than its degree.
_DISTINCT_X_NEEDED = {'Linear': 2, 'Quadratic': 3, 'Exponential': 2, 'Power': 2, 'Logarithmic': 2}


@dataclass(frozen=True)
class Trendline:
    """A fitted curve, ready to draw.

    Attributes:
        kind: One of ``TRENDLINE_KINDS``.
        x: The X values in ascending order, in the X column's own type.
        y: The curve's value at each of them.
    """

    kind: str
    x: pd.Series
    y: np.ndarray


def trendline_kinds(x: pd.Series, y: pd.Series) -> tuple[str, ...]:
    """The trendline kinds a graph of *y* against *x* can draw, by the columns' types."""
    if not pd.api.types.is_numeric_dtype(y):
        return ()
    if pd.api.types.is_datetime64_any_dtype(x):
        return _TIME_X_KINDS
    if pd.api.types.is_numeric_dtype(x):
        return TRENDLINE_KINDS
    return ()


def fit_trendline(kind: str, x: pd.Series, y: pd.Series) -> Trendline:
    """Fit a *kind* trendline to *y* against *x*.

    A time X is measured in seconds from its first value.

    Raises:
        PostProcessingRefusedError: operation ``'Trendline'``, reason
            ``fit_impossible``; the kind is not one this pair of columns
            takes, or the values cannot give that curve. The message names
            the values and how many.
        numpy.linalg.LinAlgError: a fit the checks admitted did not converge;
            a fault, not a refusal.
    """

    def refuse(problem: str) -> PostProcessingRefusedError:
        return PostProcessingRefusedError(
            operation=TRENDLINE_OPERATION,
            reason='fit_impossible',
            message=f'A {kind} trendline of {y.name} against {x.name} cannot be drawn: {problem}.',
        )

    if kind not in trendline_kinds(x, y):
        offered = ', '.join(trendline_kinds(x, y)) or 'none'
        raise refuse(f'these columns take {offered}')

    order = np.argsort(x.to_numpy(), kind='stable')
    x = x.iloc[order]
    y = y.iloc[order]
    if pd.api.types.is_datetime64_any_dtype(x):
        xs = (x - x.min()).dt.total_seconds().to_numpy(dtype=float)
    else:
        xs = x.to_numpy(dtype=float)
    ys = y.to_numpy(dtype=float)

    def count_where(values: np.ndarray, bad: np.ndarray, name: str, what: str) -> None:
        if bad.any():
            raise refuse(f'{int(bad.sum())} of the {len(values)} {name} values {what}')

    count_where(xs, ~np.isfinite(xs), x.name, 'are missing or not a number')
    count_where(ys, ~np.isfinite(ys), y.name, 'are missing or not a number')
    if kind in ('Exponential', 'Power'):
        count_where(
            ys, ys <= 0, y.name, f'are 0 or below, and a {kind} curve needs every one above 0'
        )
    if kind in ('Power', 'Logarithmic'):
        count_where(
            xs, xs <= 0, x.name, f'are 0 or below, and a {kind} curve needs every one above 0'
        )
    distinct = len(np.unique(xs))
    if distinct < _DISTINCT_X_NEEDED[kind]:
        raise refuse(
            f'{x.name} has {distinct} distinct value(s), and a {kind} curve needs at least '
            f'{_DISTINCT_X_NEEDED[kind]}'
        )

    if kind == 'Linear':
        curve = np.poly1d(np.polyfit(xs, ys, 1))(xs)
    elif kind == 'Quadratic':
        curve = np.poly1d(np.polyfit(xs, ys, 2))(xs)
    elif kind == 'Exponential':
        curve = np.exp(np.poly1d(np.polyfit(xs, np.log(ys), 1))(xs))
    elif kind == 'Power':
        curve = np.exp(np.poly1d(np.polyfit(np.log(xs), np.log(ys), 1))(np.log(xs)))
    else:
        curve = np.poly1d(np.polyfit(np.log(xs), ys, 1))(np.log(xs))
    return Trendline(kind=kind, x=x, y=curve)
