"""Structural guard: the user manual names only what the code has.

The manual (``docs/manual/``) marks every name a reader types or clicks:
a control ``[Image Mode]{.guilabel}``, a channel drawer
``[488 nm Excitation]{.channel}``, a setting
``[protocol.autogain.target_brightness]{.setting}``. Each mark is checked
against the code that owns the name, so a control renamed or removed in the
code fails here instead of outliving it in the manual, as "Enable Full Pixel
Depth", "Apply to Steps" and ``max_duration_seconds`` did in the three
manuals this one replaced. When this goes red the MANUAL is what gets
corrected: it follows the code, never the reverse.

Also held: the model table in chapter 1 states what ``data/scopes.json``
says each model has; the link to the integration guide names a file in the
tree; every image is used and none is a full-resolution original; and the
manual is left out of the source archive (``export-ignore``), so its images
do not weigh down GitHub's "Download ZIP".

A default or a range the prose quotes is marked too, naming its setting:
``[90]{.default setting="jpg_quality"}``,
``[1 to 100]{.range setting="jpg_quality"}``. A default is checked against
the shipped template, a range against the setting's rule in ``_RANGES``.

Not checked: a label the UI builds by f-string (other than the drawers'),
a control whose behaviour changes under the same name, a slider's span (a
widget's range is display), a limit the hardware declares, and a quoted
value that is no number or switch ("Max", a percentage of a fraction).
Those belong to the manual's claims audit and to the read of the trunk's
behaviour changes at each manual session's start.
"""

import ast
import fnmatch
import json
import re
import urllib.parse

from PIL import Image

from _lumascope.modules.channel_record import load_scope_models
from _lumascope.modules.exceptions import SettingRefusedError
from _lumascope.modules.settings_paths import _RANGES
from tests.ast_seams import REPO_ROOT, tracked_files
from ui.image_settings import _drawer_title
from ui.microscope_settings import IMAGE_MODE_LABELS, OUTPUT_FORMAT_LABELS
from ui.zstack import ZSTACK_REFERENCE_LABELS

MANUAL = 'docs/manual'

# The longest edge a manual image may have: a whole-window shot at 2x on a
# Mac is 2992 px wide, and a camera original (about 4032 px) must not land.
MAX_IMAGE_EDGE_PX = 3000

_MARK = re.compile(r'\[([^\[\]]+)\]\{\.(guilabel|channel|setting)\}')


def _files(folder, pattern):
    """The tracked files in ``folder`` matching ``pattern``, as paths."""
    return [REPO_ROOT / rel for rel in tracked_files([folder], pattern)]


def _pages():
    return _files(MANUAL, '*.qmd')


def _marks(kind):
    """Every (page, text) the manual marks with ``kind``, whitespace folded."""
    return [
        (page.name, ' '.join(text.split()))
        for page in _pages()
        for text, mark in _MARK.findall(page.read_text(encoding='utf-8'))
        if mark == kind
    ]


def _literal_texts(node):
    """The whole labels an expression can set: a string literal, or each
    literal branch of a conditional (``'Frames' if ... else 'mp4'``,
    ``objective_id or 'Assign Objective'``)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.IfExp):
        return _literal_texts(node.body) + _literal_texts(node.orelse)
    if isinstance(node, ast.BoolOp):
        return [text for value in node.values for text in _literal_texts(value)]
    return []


def _ui_labels():
    """Every literal ``text`` or ``title`` the UI sets, ``\\n`` read as a
    space, and the label tables' values. Kv rules are read from the ``.kv``
    files and from the kv strings a module hands ``Builder.load_string``
    (Advanced Settings is one)."""
    labels = set()
    kv_line = re.compile(r'^\s*(?:text|title)\s*:\s*(\'[^\']*\'|"[^"]*")\s*$')

    def read_kv(source):
        for line in source.splitlines():
            m = kv_line.match(line)
            if m:
                labels.add(ast.literal_eval(m.group(1)))

    for kv in _files('ui', '**/*.kv'):
        read_kv(kv.read_text(encoding='utf-8'))
    for py in _files('ui', '**/*.py'):
        for node in ast.walk(ast.parse(py.read_text(encoding='utf-8'))):
            loads_kv = (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'load_string'
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            )
            if loads_kv:
                read_kv(node.args[0].value)
            sets_a_label = (isinstance(node, ast.keyword) and node.arg in ('text', 'title')) or (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Attribute) and t.attr in ('text', 'title')
                    for t in node.targets
                )
            )
            if sets_a_label:
                labels.update(_literal_texts(node.value))
    labels.update(
        IMAGE_MODE_LABELS.values(), OUTPUT_FORMAT_LABELS.values(), ZSTACK_REFERENCE_LABELS.values()
    )
    return {' '.join(label.split()) for label in labels}


def _resolves_in(tree, dotted):
    node = tree
    for part in dotted.split('.'):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def test_every_marked_setting_is_one_lvp_stores():
    """A ``{.setting}`` is a full dotted path the shipped settings hold, or
    one the settings' range table names (its ``*`` standing for any layer)."""
    shipped = json.loads((REPO_ROOT / 'data' / 'settings.json').read_text(encoding='utf-8'))
    unknown = [
        (page, path)
        for page, path in _marks('setting')
        if not _resolves_in(shipped, path)
        and not any(fnmatch.fnmatchcase(path, key) for key in _RANGES)
    ]
    assert not unknown, f'settings the manual names that LVP does not have: {unknown}'


_NUMBER_MARK = re.compile(r'\[([^\[\]]+)\]\{\.(default|range) setting="([^"]+)"\}')
_NUMBER = re.compile(r'-?\d+(?:\.\d+)?')


def _number_marks(kind):
    """Every (page, text, settings) the manual marks ``{.<kind> setting="..."}``.

    A ``{.default}`` or ``{.range}`` written any other way (no ``setting``, a
    second attribute) is not a mark this guard can read, so it fails here
    rather than going unchecked.
    """
    marks, unread = [], []
    for page in _pages():
        source = page.read_text(encoding='utf-8')
        found = _NUMBER_MARK.findall(source)
        marks += [(page.name, text, paths.split()) for text, mark, paths in found if mark == kind]
        if len(re.findall(r'\{\.(?:default|range)\b', source)) != len(found):
            unread.append(page.name)
    assert not unread, (
        f'pages with a {{.default}} or {{.range}} mark this guard cannot read: {unread}'
    )
    assert marks, f'the manual marks no {kind}: the check below would pass on nothing'
    return marks


def _shipped(path):
    node = json.loads((REPO_ROOT / 'data' / 'settings.json').read_text(encoding='utf-8'))
    for part in path.split('.'):
        node = node[part]
    return node


def test_every_marked_default_is_the_shipped_one():
    """A ``{.default setting="a b"}`` states what the shipped template holds
    for each setting named: "on" or "off" for a switch, else its numbers,
    one for every setting or one each, in order (``1900 x 1900`` for the
    frame's width and height)."""
    wrong = []
    for page, text, paths in _number_marks('default'):
        shipped = [_shipped(path) for path in paths]
        if text in ('on', 'off'):
            stated = [text == 'on'] * len(paths)
            agrees = all(
                isinstance(s, bool) and s == t for s, t in zip(shipped, stated, strict=True)
            )
        else:
            numbers = [float(n) for n in _NUMBER.findall(text)]
            stated = numbers * len(paths) if len(numbers) == 1 else numbers
            agrees = len(stated) == len(paths) and all(
                not isinstance(s, bool) and isinstance(s, (int, float)) and float(s) == t
                for s, t in zip(shipped, stated, strict=True)
            )
        if not agrees:
            wrong.append((page, text, dict(zip(paths, shipped, strict=True))))
    assert not wrong, f'defaults the manual states that the shipped settings do not hold: {wrong}'


def _refusal(rule, path, value):
    """The reason ``rule`` refuses ``value`` for, or None when it takes it."""
    try:
        rule(path, value)
    except SettingRefusedError as refused:
        return refused.reason
    return None


def test_every_marked_range_is_the_one_lvp_refuses_outside():
    """A ``{.range setting="..."}`` reading "<low> to <high>" is the range
    the setting's rule in ``_RANGES`` holds: both ends are taken, and one
    whole unit beyond either end is refused as ``out_of_range``."""
    wrong = []
    for page, text, paths in _number_marks('range'):
        low, high = (int(n) if float(n).is_integer() else float(n) for n in _NUMBER.findall(text))
        for path in paths:
            rule = next(rule for key, rule in _RANGES.items() if fnmatch.fnmatchcase(path, key))
            answers = {
                value: _refusal(rule, path, value) for value in (low, high, low - 1, high + 1)
            }
            expected = {low: None, high: None, low - 1: 'out_of_range', high + 1: 'out_of_range'}
            if answers != expected:
                wrong.append((page, text, path, answers))
    assert not wrong, f'ranges the manual states that the settings rules do not hold: {wrong}'


def test_every_marked_control_is_a_label_the_ui_shows():
    """A ``{.guilabel}`` is a literal ``text`` or ``title`` in ``ui/`` (a
    conditional's literal branches included) or a value of a label table."""
    labels = _ui_labels()
    unknown = [(page, text) for page, text in _marks('guilabel') if text not in labels]
    assert not unknown, f'controls the manual names that the UI does not show: {unknown}'


def test_every_marked_channel_is_a_drawer_lvp_titles():
    """A ``{.channel}`` is the drawer title of a layer some model has."""
    titles = {
        _drawer_title(record) for model in load_scope_models().values() for record in model.channels
    }
    unknown = [(page, text) for page, text in _marks('channel') if text not in titles]
    assert not unknown, f'channel drawers the manual names that no model has: {unknown}'


def _model_table():
    """Chapter 1's model table as {model: (focus, xy, turret, channels)}.

    A row whose every cell is empty is the gap that sets one group of
    models apart from the next, and states no model.
    """
    lines = (REPO_ROOT / MANUAL / 'index.qmd').read_text(encoding='utf-8').splitlines()
    caption = next(i for i, line in enumerate(lines) if '{#tbl-models' in line)
    end = caption
    while not lines[end - 1].startswith('|'):
        end -= 1
    start = end
    while lines[start - 1].startswith('|'):
        start -= 1
    header, _rule, *rows = (
        [cell.strip() for cell in line.strip().strip('|').split('|')] for line in lines[start:end]
    )
    assert header == ['Model', 'Focus motor', 'XY stage', 'Turret', 'Channels'], header
    return {
        model: (focus, xy, turret, channels)
        for model, focus, xy, turret, channels in rows
        if any((model, focus, xy, turret, channels))
    }


def test_the_model_table_states_what_each_model_has():
    """Every model ``data/scopes.json`` lists is a row, and each row's focus
    motor, XY stage, turret and channels are the model's."""
    yes_no = {True: 'yes', False: 'no'}
    expected = {
        name: (
            yes_no[model.has_focus],
            yes_no[model.has_xy_stage],
            yes_no[model.has_turret],
            ', '.join(record.label for record in model.channels),
        )
        for name, model in load_scope_models().items()
    }
    table = _model_table()
    wrong = {
        name: {'manual': table.get(name), 'scopes.json': expected.get(name)}
        for name in expected.keys() | table.keys()
        if table.get(name) != expected.get(name)
    }
    assert not wrong, f'model table rows that differ from data/scopes.json: {wrong}'


def test_the_integration_guide_link_names_the_guide():
    """The manual points integrators at ``docs/lumascope-skills.md`` by its
    GitHub URL (a relative link is dead in the PDF), and that file exists."""
    urls = [
        url
        for page in _pages()
        for url in re.findall(
            r'https://github\.com/EtalumaSupport/LumaViewPro/blob/[^)\s]+',
            page.read_text(encoding='utf-8'),
        )
    ]
    paths = [urllib.parse.urlsplit(url).path.split('/', 5)[5] for url in urls]
    assert 'docs/lumascope-skills.md' in paths, f'no link to the integration guide; links: {urls}'
    missing = [path for path in paths if not (REPO_ROOT / path).is_file()]
    assert not missing, f'linked files not in the tree: {missing}'


def test_every_image_is_used_and_print_sized():
    """Every file in ``images/`` is referenced by a page or the style, and
    none has an edge longer than a 2x window shot."""
    referers = ''.join(
        path.read_text(encoding='utf-8')
        for path in [*_pages(), *_files(f'{MANUAL}/style', '*.typ')]
    )
    unused = sorted(
        image.name
        for image in _files(f'{MANUAL}/images', '*')
        if f'images/{image.name}' not in referers
    )
    assert not unused, f'images nothing in the manual uses: {unused}'
    oversize = {}
    for image in _files(f'{MANUAL}/images', '*'):
        with Image.open(image) as picture:
            if max(picture.size) > MAX_IMAGE_EDGE_PX:
                oversize[image.name] = picture.size
    assert not oversize, f'images over {MAX_IMAGE_EDGE_PX} px on an edge (an original?): {oversize}'


def test_the_manual_is_left_out_of_source_archives():
    """``.gitattributes`` marks the manual's folder ``export-ignore``."""
    lines = (REPO_ROOT / '.gitattributes').read_text(encoding='utf-8').splitlines()
    assert 'docs/manual export-ignore' in (line.strip() for line in lines)
