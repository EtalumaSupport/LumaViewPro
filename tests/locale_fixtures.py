"""A process locale whose encoding is not UTF-8, for the text files' contract.

Python's text ``open()`` without ``encoding=`` decodes with the locale's
encoding (cp1252 on a Windows machine), so a reader that names none reads
the same bytes differently on different machines. Under this locale such a
reader decodes as Latin-1, as Windows would decode as cp1252, and a reader
that names UTF-8 is unaffected: the difference the tests pin.
"""

import contextlib
import locale
import sys

import pytest

# The spellings a Latin-1 locale goes by on macOS and on Linux.
_LATIN1_LOCALES = ('en_US.ISO8859-1', 'en_US.ISO-8859-1', 'en_US.iso88591')


@contextlib.contextmanager
def a_latin1_locale():
    """Run the block with the process's text encoding Latin-1; skip when the
    host has no such locale (the stock Ubuntu runner) or Python ignores the
    locale (UTF-8 mode)."""
    if sys.flags.utf8_mode:
        pytest.skip('Python runs in UTF-8 mode, which ignores the locale')
    previous = locale.setlocale(locale.LC_CTYPE)
    for name in _LATIN1_LOCALES:
        try:
            locale.setlocale(locale.LC_CTYPE, name)
            break
        except locale.Error:
            continue
    else:
        pytest.skip(f'no Latin-1 locale on this host (tried {_LATIN1_LOCALES})')
    try:
        yield
    finally:
        locale.setlocale(locale.LC_CTYPE, previous)
