# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Call tripwire for API members that are about to become private or disappear.

TEMPORARY BY DESIGN. This whole module exists to be deleted once the surface
it watches has settled; see "Removing this" at the bottom.

A static census can prove a member has no callers in code we can see. It
cannot prove there is no caller at all -- a name reached through getattr, a
string in a config, an integrator's script outside both repos. Deleting on a
clean census and finding out from a field report is the failure this exists
to prevent.

So a member scheduled for removal gets wrapped for a period first. It keeps
working exactly as before; the wrapper only records that someone called it,
and from where. A quiet member at the end of that period has evidence behind
its removal instead of an absence of evidence.

The distinction that makes the record useful is INTERNAL versus EXTERNAL
callers. Every one of these members still has callers inside this package --
that is why it is being made private rather than deleted outright -- so
warning about those would bury the one signal that matters under the noise we
already predicted. Internal calls are counted silently. A call from outside
the package is the thing nobody knew about, and only that warns.

Usage, on the member being retired:

    @retiring('motion.get_overshoot', becoming=PRIVATE)
    def get_overshoot(self) -> bool:
        ...

Order matters for properties -- decorate the underlying function, then let
``property`` wrap the result:

    @property
    @retiring('motion.is_homing', becoming=PRIVATE)
    def is_homing(self) -> bool:
        ...

Removing this: delete the module, then delete every ``@retiring`` line and
its import. ``grep -rn "@retiring" modules/`` finds the complete set, which is
why the decorator is always spelled that way at the use site.
"""

from __future__ import annotations

import functools
import logging
import os
import sys
import threading

_api_log = logging.getLogger('LVP.api')

PRIVATE = 'private'
REMOVED = 'removed'
_DISPOSITIONS = (PRIVATE, REMOVED)

# Calls originating inside this directory are the API talking to itself. Those
# are the callers the retirement already accounts for.
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

_lock = threading.Lock()
# member -> caller site -> {'count', 'external'}
_fires: dict[str, dict[str, dict]] = {}


def _caller_site(depth: int = 2) -> tuple[str, bool]:
    """(file:line of the caller, whether it is outside this package).

    Reads the frame directly rather than building a traceback: this runs on
    every call to a wrapped member, including ones on the capture path.
    """
    try:
        frame = sys._getframe(depth)
    except ValueError:
        return '<unknown>', True
    path = frame.f_code.co_filename
    external = not os.path.abspath(path).startswith(_PACKAGE_DIR)
    return f'{path}:{frame.f_lineno}', external


def retiring(member: str, *, becoming: str):
    """Record calls to a member that is scheduled to be made private or removed.

    Behaviour is untouched -- arguments, return value and raised exceptions all
    pass straight through. An external caller is reported once per call site,
    not once per call, so a member on a per-frame path cannot flood the log.
    """
    if becoming not in _DISPOSITIONS:
        raise ValueError(f'{member}: becoming must be one of {_DISPOSITIONS!r}, got {becoming!r}')

    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            site, external = _caller_site()
            with _lock:
                sites = _fires.setdefault(member, {})
                record = sites.get(site)
                if record is None:
                    record = {'count': 0, 'external': external}
                    sites[site] = record
                    first_sighting = True
                else:
                    first_sighting = False
                record['count'] += 1
            if first_sighting and external:
                _api_log.warning(
                    'API member scheduled to become %s was called from outside '
                    'the API package: %s (from %s). If this caller is '
                    'legitimate the retirement is wrong -- say so before the '
                    'member goes.',
                    becoming,
                    member,
                    site,
                )
            return fn(*args, **kwargs)

        wrapper.__retiring__ = (member, becoming)
        return wrapper

    return decorate


def retirement_fires(*, external_only: bool = True) -> dict[str, dict[str, int]]:
    """What has actually been called, as ``{member: {site: count}}``.

    Reading this beats grepping logs when the question is "did anything touch
    these", and it is what makes each fire dispositionable one at a time.
    """
    with _lock:
        out: dict[str, dict[str, int]] = {}
        for member, sites in _fires.items():
            hits = {
                site: rec['count']
                for site, rec in sites.items()
                if rec['external'] or not external_only
            }
            if hits:
                out[member] = hits
        return out


def clear_retirement_fires() -> None:
    """Drop the record. For tests that need a known starting point."""
    with _lock:
        _fires.clear()
