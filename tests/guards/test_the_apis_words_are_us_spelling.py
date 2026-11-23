# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The API's words are spelled the US way.

The names freeze with the API, and its readers read one spelling: a
caller who writes ``catalog`` from the guide meets ``catalogue`` nowhere.
British spellings reached four published names and about 260 reference
words before they were renamed, with no check to stop the next one.

Read: every name LVP defines in ``src/`` and ``ui/`` (a function, class,
argument, assigned name or attribute, an import's alias) and every string
there, docstrings included; the ``.kv`` files' lines; the guide, the
plugin tutorial, ``docs/lumascope-skills.md`` and the examples. A comment
is not read. A name LVP uses but does not define (``CancelledError``,
``usb1.TRANSFER_CANCELLED``, ``Future.cancelled``) is not LVP's spelling
and is not read either; a string that names one is exempt by its whole
word in ``_NOT_OURS``. A row of the plugin tutorial's level table records
the names a rename retired, so it is not read.

The words are the British forms LVP's text has carried, each with its
inflections; a word is judged by each part of an identifier
(``CatalogueNameRefusedError`` is ``catalogue``, ``name``, ...).
"""

import ast
import re

from tests.ast_seams import REPO_ROOT, Packages, Words, tracked_files

# Each British word with its inflections, one family a line.
_FAMILIES = (
    'analogue analogues analyse analysed analysing analyser',
    'artefact artefacts',
    'behaviour behaviours behavioural',
    'cancellable cancelled cancelling',
    'catalogue catalogues catalogued cataloguing',
    'centre centres centred centring recentre recentred',
    'colour colours coloured colouring colourful',
    'digitise digitised digitises digitising',
    'grey greys greyed greying greyscale',
    'honour honours honoured honouring',
    'initialise initialised initialises initialising initialisation initialiser',
    'labelled labelling',
    'metre metres micrometre micrometres millimetre millimetres',
    'centimetre centimetres nanometre nanometres',
    'modelled modelling unmodelled',
    'neighbour neighbours neighbouring',
    'normalise normalised normalises normalising normalisation',
    'prioritise prioritised prioritises prioritising',
    'recognise recognised recognises recognising unrecognised',
    'sanitise sanitised sanitiser sanitising',
    'serialise serialised serialises serialising serialisation',
    'deserialise deserialised deserialisation',
    'signalled signalling',
    'travelled travelling',
)
_BRITISH = frozenset(word for family in _FAMILIES for word in family.split())

# Whole words that are another body's names, which LVP's text may quote.
_NOT_OURS = Words(
    {
        'CancelledError',  # Python's asyncio and concurrent.futures
        'TRANSFER_CANCELLED',  # python-libusb1's transfer status
        'LightGrey',  # an IDS camera's TestPattern entry, read from the SDK
        'error_cancelled',  # Windows' ERROR_CANCELLED, mirrored by its code
    }
)

_CODE_FOLDERS = Packages({'src', 'ui'})
_PY_EXAMPLES = 'docs/api_examples'
_DOCS = ('docs/guide/**/*.md', 'docs/plugin-tutorial.md', 'docs/lumascope-skills.md')
_LEVEL_ROW = '| level '


def _british_words(text):
    """Each British word in ``text``, read by the parts of every word."""
    for word in re.findall(r'[A-Za-z_]+', text):
        if word in _NOT_OURS:
            continue
        for part in re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])', word):
            if part.lower() in _BRITISH:
                yield word


def _defined_names_and_strings(tree):
    """``(line, text)`` for every name a module defines and every string in it."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            yield node.lineno, node.name
        elif isinstance(node, ast.arg):
            yield node.lineno, node.arg
        elif isinstance(node, ast.Name | ast.Attribute) and isinstance(node.ctx, ast.Store):
            yield node.lineno, node.id if isinstance(node, ast.Name) else node.attr
        elif isinstance(node, ast.ExceptHandler) and node.name:
            yield node.lineno, node.name
        elif isinstance(node, ast.alias) and node.asname:
            yield node.lineno, node.asname
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value


def _read_lines(rel_path):
    """``(line, text)`` for every line a reader reads.

    A comment line is not one; a ``.kv`` directive (``#:import``,
    ``#:set``) is code, and is.
    """
    lines = (REPO_ROOT / rel_path).read_text(encoding='utf-8').splitlines()
    for lineno, line in enumerate(lines, 1):
        if rel_path.endswith('.md'):
            if rel_path == 'docs/plugin-tutorial.md' and line.startswith(_LEVEL_ROW):
                continue
        elif line.lstrip().startswith('#') and not line.lstrip().startswith('#:'):
            continue
        yield lineno, line


def _texts():
    for rel_path in tracked_files(_CODE_FOLDERS) + tracked_files([_PY_EXAMPLES]):
        tree = ast.parse((REPO_ROOT / rel_path).read_text(encoding='utf-8'))
        for lineno, text in _defined_names_and_strings(tree):
            yield rel_path, lineno, text
    docs = {pattern: tracked_files(pattern=pattern) for pattern in _DOCS}
    missing = [pattern for pattern, paths in docs.items() if not paths]
    assert not missing, f'a reference document the guard reads is not in the tree: {missing}'
    lines_read = (
        tracked_files(_CODE_FOLDERS, pattern='**/*.kv')
        + tracked_files([_PY_EXAMPLES], pattern='**/*.sh')
        + [path for paths in docs.values() for path in paths]
    )
    for rel_path in lines_read:
        for lineno, line in _read_lines(rel_path):
            yield rel_path, lineno, line


def test_the_apis_names_strings_and_reference_are_us_spelling():
    offenders = sorted(
        {
            f'{path}:{lineno}: {word}'
            for path, lineno, text in _texts()
            for word in _british_words(text)
        }
    )
    assert not offenders, (
        'a name, string or reference line LVP publishes spells a word the British way '
        "(spell it the US way; a name that is another body's goes in _NOT_OURS):\n"
        + '\n'.join(offenders)
    )
