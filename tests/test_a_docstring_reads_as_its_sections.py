# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A docstring is read by its sections, as the reference renders it (``api_surface.entries_of``, ``markdown_of``).

``docs/api-reference.md`` is rendered from these two readers, and its guard
compares the committed file with a fresh render, so a reader that misreads
a section would be committed with the file rather than fail it. These pin
what each reader gives.
"""

import pytest

from _lumascope.modules.api_surface import entries_of, markdown_of

DOC = """Set the gain, and answer with it.

Args:
    gain_db: The gain, in dB; a value outside
        the camera's range is refused.
    wait: Whether to wait for the camera.

Returns:
    The gain in effect, ``<None>`` when the camera went.

Raises:
    ArgumentRefusedError: ``'not_a_number'``, the gain is not a
        finite number.

The camera is asked on its lane."""


def test_each_entry_is_read_by_name_with_its_continuation_joined() -> None:
    assert entries_of(DOC, 'Args') == {
        'gain_db': "The gain, in dB; a value outside the camera's range is refused.",
        'wait': 'Whether to wait for the camera.',
    }


def test_a_section_ends_at_the_first_unindented_line() -> None:
    assert entries_of(DOC, 'Raises') == {
        'ArgumentRefusedError': "``'not_a_number'``, the gain is not a finite number.",
    }


def test_a_missing_section_has_no_entries() -> None:
    assert entries_of(DOC, 'Attributes') == {}


def test_markdown_renders_each_section_under_its_heading() -> None:
    assert markdown_of(DOC) == (
        'Set the gain, and answer with it.\n\n'
        '**Args**\n\n'
        "- `gain_db`: The gain, in dB; a value outside the camera's range is refused.\n"
        '- `wait`: Whether to wait for the camera.\n\n'
        '**Returns**\n\n'
        'The gain in effect, `<None>` when the camera went.\n\n'
        '**Raises**\n\n'
        "- `ArgumentRefusedError`: `'not_a_number'`, the gain is not a finite number.\n\n"
        'The camera is asked on its lane.'
    )


TWICE = """Save the focus.

Raises:
    ArgumentRefusedError: ``'channel_unknown'``, the channel is not one.
    ArgumentRefusedError: ``'not_a_number'``, the height is NaN."""


def test_a_class_raised_for_two_reasons_is_rendered_with_both() -> None:
    assert markdown_of(TWICE) == (
        'Save the focus.\n\n'
        '**Raises**\n\n'
        "- `ArgumentRefusedError`: `'channel_unknown'`, the channel is not one.\n"
        "- `ArgumentRefusedError`: `'not_a_number'`, the height is NaN."
    )


def test_a_name_given_twice_cannot_be_read_by_name() -> None:
    with pytest.raises(ValueError, match=r"'gain_db'"):
        entries_of('Set it.\n\nArgs:\n    gain_db: The gain.\n    gain_db: Again.', 'Args')


def test_a_tag_outside_a_literal_is_text() -> None:
    assert markdown_of('Read at /jobs/<id>, or ``/jobs/<id>``.') == (
        'Read at /jobs/&lt;id>, or `/jobs/<id>`.'
    )


def test_an_entry_section_with_no_entry_is_its_prose() -> None:
    assert markdown_of('Read it.\n\nRaises:\n    Nothing.') == 'Read it.\n\n**Raises**\n\nNothing.'


def test_a_section_left_out_is_not_rendered() -> None:
    assert markdown_of(DOC, leave_out=('Args', 'Returns', 'Raises')) == (
        'Set the gain, and answer with it.\n\nThe camera is asked on its lane.'
    )
