# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every value the API would put on a wire has a declared form.

``test_the_api_is_declared`` closes the API over project classes only, so a
member handing out an image array, a table or a thread passed it, and a
wire client would have been offered something no wire carries. This guard
walks the API from the Session along every wire-marked member and field
(``modules.wire_encoding.wire_gaps``) and fails on a type with no form, on
a callable anywhere but a parameter, and on a wire-marked method of a
record, which has no address to call it at. It is shown to catch each on a
fixture, and to reach the real API by flipping one in-process member back
onto the wire.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np
import pytest

from modules import wire_encoding
from modules.api_surface import MARK_ATTRIBUTE, api, api_fields


@pytest.fixture(scope='module')
def classes():
    return wire_encoding.project_classes()


@pytest.fixture(scope='module')
def aliases():
    return wire_encoding.project_aliases()


def test_the_session_has_no_wire_edge_without_a_form(classes, aliases):
    from modules.scope_session import ScopeSession

    gaps = wire_encoding.wire_gaps(ScopeSession, classes, aliases)
    assert not gaps, (
        'A wire edge has no form: give the type one in modules/wire_encoding.py, or mark '
        'the member @api(in_process=True):\n  ' + '\n  '.join(gaps)
    )


def test_a_member_put_back_on_the_wire_is_caught(classes, aliases, monkeypatch):
    from modules.lumascope_api.imaging import ImagingAPI
    from modules.scope_session import ScopeSession

    monkeypatch.setattr(ImagingAPI.get_image, MARK_ATTRIBUTE, 'api')

    gaps = wire_encoding.wire_gaps(ScopeSession, classes, aliases)
    assert any(g.startswith('ImagingAPI.get_image') and 'ndarray' in g for g in gaps), gaps


@api_fields('n', record=True)
class _Plate:
    n: int

    @api
    def where(self) -> int:
        return 0


class _Root:
    @api
    def image(self) -> np.ndarray:
        raise NotImplementedError

    @api
    def worker(self) -> threading.Thread:
        raise NotImplementedError

    @api
    def plate(self) -> _Plate:
        raise NotImplementedError

    @api
    def handed_back(self) -> Callable[[], None]:
        raise NotImplementedError

    @api
    def counted(self, on_progress: Callable[[int], None] | None = None) -> int:
        raise NotImplementedError


def test_each_kind_of_gap_is_caught():
    gaps = wire_encoding.wire_gaps(_Root, {'_Root': _Root, '_Plate': _Plate}, {})

    assert gaps == [
        '_Plate.where: a method of a record',
        '_Root.handed_back return: Callable is not on the wire',
        '_Root.image return: ndarray has no wire form',
        '_Root.worker return: Thread has no wire form',
        # Its routes cannot be built either: the first return with no form.
        '_Root: np.ndarray: ndarray has no wire form',
    ]


class _Kept:
    """A live object no member hands out: a client never holds its id."""

    @api
    def poke(self) -> None:
        raise NotImplementedError


class _Given:
    """A live object a member hands out."""

    @api
    def poke(self) -> None:
        raise NotImplementedError


class _Holder:
    @api
    def given(self) -> _Given:
        raise NotImplementedError

    @api
    def needs_kept(self, kept: _Kept) -> None:
        raise NotImplementedError

    @api
    def may_take_kept(self, n: int, kept: _Kept | None = None) -> None:
        raise NotImplementedError

    @api
    def takes_given(self, given: _Given) -> None:
        raise NotImplementedError


def test_a_live_object_no_member_hands_out_cannot_be_required():
    classes = {'_Holder': _Holder, '_Kept': _Kept, '_Given': _Given}

    assert wire_encoding.handed_out(_Holder, classes, {}) == {_Given}
    gaps = wire_encoding.wire_gaps(_Holder, classes, {})
    assert gaps == [
        '_Holder: _Holder.needs_kept parameter kept: _Kept: no wire member hands out '
        'the live object _Kept'
    ]


def test_an_optional_live_object_no_member_hands_out_is_not_sent():
    class _Optional:
        @api
        def given(self) -> _Given:
            raise NotImplementedError

        may_take_kept = _Holder.may_take_kept
        takes_given = _Holder.takes_given

    classes = {'_Optional': _Optional, '_Kept': _Kept, '_Given': _Given}
    reach = wire_encoding.handed_out(_Optional, classes, {})

    sent = {
        m.name: [p.name for p in m.parameters]
        for m in wire_encoding.wire_members(_Optional, classes, {}, handed_out=reach)
    }
    assert sent['may_take_kept'] == ['n']
    assert sent['takes_given'] == ['given']


def test_validate_steps_put_back_on_the_wire_is_caught(classes, aliases, monkeypatch):
    from modules.protocol import Protocol
    from modules.scope_session import ScopeSession

    monkeypatch.setattr(Protocol.validate_steps, MARK_ATTRIBUTE, 'api')

    gaps = wire_encoding.wire_gaps(ScopeSession, classes, aliases)
    assert any('validate_steps' in g and 'ObjectiveLoader' in g for g in gaps), gaps
