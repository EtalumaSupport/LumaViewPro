# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every read is marked a read, and every declared stop is a member that exists.

A host decides how to carry a member from its marks (``modules/api_surface.py``):
a member marked ``reads`` changes nothing, so the REST server may answer it
as a read; a member marked ``stop`` names what a cancel of its job calls.
Before the marks, a host could only guess from a name: 53 reads were routed
as commands, and nothing said which member stopped a run a client started.
This guard holds the marks to what the members are:

1. **A prefixed read is marked.** A method named ``get_``, ``is_``, ``has_``
   or ``read_`` is marked ``reads``. ``ImagingAPI.get_image`` (it triggers a
   capture) and ``get_image_from_buffer`` (it draws on the buffered frame)
   are the two whose names say otherwise; they are named here until their
   members are renamed. A property is a read by its kind and takes no mark.
2. **A read has no stop.** Nothing a read starts outlives it.
3. **A stop exists.** A declared stop is ``'Class.member'`` naming a marked
   member of a project class, which is not itself a read.

Its subject is a class's members: a read and a stop are what the server
routes, and a module's marked function is in-process, never routed
(``test_the_api_is_declared``). A static read: no session.
"""

from __future__ import annotations

import re

from modules import wire_encoding
from modules.api_surface import mark_of, reads_of, stop_of

_READ_NAME = re.compile(r'(get|is|has|read)_')

# Named as reads, but not reads: each changes what a later caller sees.
_NOT_READS_BY_NAME = {'ImagingAPI.get_image', 'ImagingAPI.get_image_from_buffer'}


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
        if _READ_NAME.match(qualified.split('.')[1])
        and qualified not in _NOT_READS_BY_NAME
        and not reads_of(member)
    ]
    assert unmarked == [], f'mark each @api(reads=True), or rename it: {unmarked}'


def test_the_two_named_exceptions_are_still_not_marked_reads():
    # Each would change what a later caller sees, so a read mark on it is wrong.
    marked = [q for q, member in _marked_methods() if q in _NOT_READS_BY_NAME and reads_of(member)]
    assert marked == []


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
