# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every read is marked a read, and every declared stop is a member that exists.

A host decides how to carry a member from its marks (``src/_lumascope/modules/api_surface.py``):
a member marked ``reads`` changes nothing, so the REST server may answer it
as a read; a member marked ``stop`` names what a cancel of its job calls.
Before the marks, a host could only guess from a name: 53 reads were routed
as commands, and nothing said which member stopped a run a client started.
This guard holds the marks to what the members are:

1. **A prefixed read is marked.** A method named ``is_``, ``has_`` or
   ``read_`` is marked ``reads``. A property is a read by its kind and takes
   no mark.
2. **No member is named ``get_``.** ``read_`` is a fresh read of the
   hardware or the host, a bare noun the state held; ``get_`` says neither,
   so a published member or field never carries it.
3. **A setter is named for what it sets.** A member named ``set_x`` that
   declares what it sets declares ``x``, the read a caller pairs it with.
4. **A read has no stop.** Nothing a read starts outlives it.
5. **A stop exists.** A declared stop is ``'Class.member'`` naming a marked
   member of a project class, which is not itself a read.

Its subject is a class's members: a read and a stop are what the server
routes, and a module's marked function is in-process, never routed
(``test_the_api_is_declared``). A static read: no session.
"""

from __future__ import annotations

import re

from _lumascope.modules import wire_encoding
from _lumascope.modules.api_surface import fields_of, mark_of, reads_of, sets_of, stop_of

_READ_NAME = re.compile(r'(is|has|read)_')


def _marked_methods():
    """Each marked member that is not a property, as ``('Class.name', member)``."""
    for owner, cls in sorted(wire_encoding.project_classes().items()):
        for name, member in vars(cls).items():
            if mark_of(member) is not None and not isinstance(member, property):
                yield f'{owner}.{name}', member


def test_every_prefixed_read_is_marked_a_read():
    unmarked = [
        qualified
        for qualified, member in _marked_methods()
        if _READ_NAME.match(qualified.split('.')[1]) and not reads_of(member)
    ]
    assert unmarked == [], f'mark each @api(reads=True), or rename it: {unmarked}'


def test_no_published_name_says_get():
    named_get = sorted(
        f'{owner}.{name}'
        for owner, cls in wire_encoding.project_classes().items()
        for name in (*(n for n, m in vars(cls).items() if mark_of(m) is not None), *fields_of(cls))
        if name.startswith('get_')
    )
    assert named_get == [], (
        f'name a fresh hardware or host read read_<fact>, a held answer <fact>: {named_get}'
    )


def test_every_setter_sets_what_it_is_named_for():
    misnamed = []
    for owner, cls in sorted(wire_encoding.project_classes().items()):
        for name, member in vars(cls).items():
            if mark_of(member) is None or not name.startswith('set_'):
                continue
            sets = sets_of(member)
            if sets is not None and sets != name.removeprefix('set_'):
                misnamed.append(f'{owner}.{name} sets {sets!r}')
    assert misnamed == [], f'name each setter set_<the read it sets>: {misnamed}'


def test_no_read_declares_a_stop():
    both = [q for q, member in _marked_methods() if reads_of(member) and stop_of(member)]
    assert both == [], f'a read starts nothing to stop: {both}'


def test_every_declared_stop_is_a_marked_member_that_is_not_a_read():
    classes = wire_encoding.project_classes()
    wrong = []
    for qualified, member in _marked_methods():
        stop = stop_of(member)
        if stop is None:
            continue
        owner, _, name = stop.partition('.')
        target = vars(classes[owner]).get(name) if owner in classes else None
        if target is None or mark_of(target) is None:
            wrong.append(f'{qualified} -> {stop}: no marked member')
        elif reads_of(target):
            wrong.append(f'{qualified} -> {stop}: a read cannot stop anything')
    assert wrong == [], '\n'.join(wrong)
