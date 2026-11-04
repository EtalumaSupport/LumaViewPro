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

Not checked: a label the UI builds by f-string (other than the drawers'),
a control whose behaviour changes under the same name, and a range the
prose quotes. Those belong to the manual's claims audit.
"""

import ast
import fnmatch
import json
import re
import urllib.parse

from PIL import Image

from modules.layer_record import load_scope_models
from modules.settings_paths import _RANGES
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
    space, and the label tables' values."""
    labels = set()
    kv_line = re.compile(r'^\s*(?:text|title)\s*:\s*(\'[^\']*\'|"[^"]*")\s*$')
    for kv in _files('ui', '**/*.kv'):
        for line in kv.read_text(encoding='utf-8').splitlines():
            m = kv_line.match(line)
            if m:
                labels.add(ast.literal_eval(m.group(1)))
    for py in _files('ui', '**/*.py'):
        for node in ast.walk(ast.parse(py.read_text(encoding='utf-8'))):
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
    """Chapter 1's model table as {model: (focus, xy, turret, channels)}."""
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
    return {model: (focus, xy, turret, channels) for model, focus, xy, turret, channels in rows}


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
