"""The sim walk driver performs a walk's steps on real widgets, or stops.

A walk is the list of GUI steps a person performed by hand against the
simulator; ``--sim-walk`` lets the session perform the mechanical ones. These
tests run the driver in a child process on a hidden real Kivy window (the
suite's conftest mocks Kivy), over fragment trees built to hold each shape the
running app has: a rule instantiated twice with the same ids, a control in a
nested RelativeLayout and one scrolled out of a ScrollView, a control inside a
collapsed drawer, an open modal, a filtered numeric field, a spinner, a native
file picker.

The flag is read by ``ui.sim_walk_file.take_walk_flag``, which
``lumaviewpro.py`` calls before Kivy is imported; its refusals are run against
that function, so no test here can start the app.
"""

import json
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    sys.platform.startswith('linux') and not __import__('os').environ.get('DISPLAY'),
    reason='needs a display for a real Kivy window',
)

_HARNESS = r"""
import json, logging, os, pathlib, sys
os.environ['KIVY_NO_ARGS'] = '1'
os.environ['KIVY_NO_CONSOLELOG'] = '1'
sys.path.insert(0, sys.argv[1])
from kivy.config import Config
Config.set('graphics', 'window_state', 'hidden')
from kivy.base import EventLoop
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from ui import file_dialogs
from ui.sim_walk import SimWalk
from ui.sim_walk_file import parse_walk

EventLoop.ensure_window()
Window.size = (900, 700)

records = []
class _Collect(logging.Handler):
    def emit(self, record):
        records.append(record.getMessage())
logging.getLogger('LVP.gui_interactions').addHandler(_Collect())
logging.getLogger('LVP.gui_interactions').setLevel(logging.INFO)
walk_lines = []
class _CollectWalk(logging.Handler):
    def emit(self, record):
        walk_lines.append(record.getMessage())
logging.getLogger('LVP.sim_walk').addHandler(_CollectWalk())
logging.getLogger('LVP.sim_walk').setLevel(logging.INFO)

hits = []
frames = {}

from ui.ui_helpers import LoggedAccordionItem  # noqa: F401 -- registers the class the kv names
from kivy.factory import Factory

Builder.load_string('''
<Layer@BoxLayout>:
    Button:
        id: go
        text: 'go'
<Layers@BoxLayout>:
    size_hint: None, None
    size: 200, 40
    pos: 650, 600
    Layer:
        id: a
    Layer:
        id: b
<Drawers@Accordion>:
    orientation: 'vertical'
    size_hint: None, None
    size: 200, 300
    pos: 650, 250
    LoggedAccordionItem:
        id: one
        title: 'one'
        log_group: 'DRAWER'
        log_item: 'one'
        Label:
            text: 'first'
    LoggedAccordionItem:
        id: two
        title: 'two'
        log_group: 'DRAWER'
        log_item: 'two'
        Button:
            id: hidden
            text: 'hidden'
<Panel@FloatLayout>:
    Button:
        id: btn
        text: 'btn'
        size_hint: None, None
        size: 100, 40
        pos: 20, 640
    TextInput:
        id: field
        multiline: False
        input_filter: 'float'
        size_hint: None, None
        size: 150, 40
        pos: 140, 640
    ScrollView:
        size_hint: None, None
        size: 200, 100
        pos: 20, 500
        BoxLayout:
            orientation: 'vertical'
            size_hint_y: None
            height: 1000
            Widget:
            Button:
                id: deep
                text: 'deep'
                size_hint_y: None
                height: 40
    RelativeLayout:
        size_hint: None, None
        size: 200, 100
        pos: 300, 400
        Button:
            id: relbtn
            text: 'rel'
            size_hint: None, None
            size: 100, 40
            pos: 50, 30
    Spinner:
        id: spin
        text: '1x1'
        values: ['1x1', '2x2', '3x3']
        size_hint: None, None
        size: 100, 40
        pos: 20, 300
    Button:
        id: popbtn
        text: 'pop'
        size_hint: None, None
        size: 100, 40
        pos: 140, 300
    Button:
        id: modalbtn
        text: 'modal'
        size_hint: None, None
        size: 100, 40
        pos: 260, 300
    Button:
        id: filebtn
        text: 'file'
        size_hint: None, None
        size: 100, 40
        pos: 380, 300
''')

root = FloatLayout()
Window.add_widget(root)
panel = Factory.Panel()
layers = Factory.Layers()
drawers = Factory.Drawers()
for w in (panel, layers, drawers):
    root.add_widget(w)

def on(name):
    def handler(*_):
        hits.append(name)
        frames[name] = Clock.frames
    return handler

ids = panel.ids
ids.btn.bind(on_release=on('btn'))
ids.deep.bind(on_release=on('deep'))
ids.relbtn.bind(on_release=on('rel'))
layers.ids.a.ids.go.bind(on_release=on('a'))
layers.ids.b.ids.go.bind(on_release=on('b'))
drawers.ids.hidden.bind(on_release=on('hidden'))
ids.spin.bind(text=lambda s, t: hits.append('spin ' + t))

def field_focus(field, focused):
    if not focused:
        hits.append('commit ' + field.text)
        frames['commit'] = Clock.frames
ids.field.bind(focus=field_focus)

def open_popup(*_):
    ok = Button(text='OK')
    p = Popup(title='Notice', content=ok, size_hint=(0.3, 0.3), auto_dismiss=False)
    ok.bind(on_release=lambda *_: (hits.append('ok'), p.dismiss()))
    p.open()
ids.popbtn.bind(on_release=open_popup)

def open_modal(*_):
    Popup(title='Busy', content=Button(text='nothing'), size_hint=(0.3, 0.3), auto_dismiss=False).open()
ids.modalbtn.bind(on_release=open_modal)

def native():
    raise AssertionError('the native picker ran during a scripted walk')
ids.filebtn.bind(on_release=lambda b: file_dialogs._run_native_dialog_async(
    b, native, lambda path: hits.append('file ' + path),
    on_cancel=lambda: hits.append('file cancelled')))

def run(steps, shot_dir):
    hits.clear(); frames.clear(); records.clear(); walk_lines.clear()
    walk = SimWalk(parse_walk(json.dumps(steps), source='fragment'), source='fragment',
                   ready=lambda: True, shot_dir=pathlib.Path(shot_dir))
    walk.start()
    for _ in range(4000):
        EventLoop.idle()
        if walk.finished:
            break
    for p in [w for w in Window.children if isinstance(w, Popup)]:
        p.dismiss()
    for _ in range(10):
        EventLoop.idle()
    return {'outcome': walk.outcome, 'hits': list(hits), 'frames': dict(frames),
            'records': list(records), 'walk_lines': list(walk_lines), 'reads': walk.reads, 'shots': walk.shots,
            'two_collapsed': drawers.ids.two.collapse}

steps = json.loads(sys.argv[2])
print('RESULT ' + json.dumps(run(steps, sys.argv[3])))
"""


def _run(tmp_path, steps):
    script = tmp_path / 'harness.py'
    script.write_text(_HARNESS)
    r = subprocess.run(
        [sys.executable, str(script), str(REPO), json.dumps(steps), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    lines = [line for line in r.stdout.splitlines() if line.startswith('RESULT ')]
    assert lines, f'harness produced no result (exit {r.returncode}):\n{r.stdout}\n{r.stderr}'
    return json.loads(lines[-1][len('RESULT ') :])


def test_every_action_reaches_its_widget_as_a_touch_would(tmp_path):
    result = _run(
        tmp_path,
        [
            {'do': 'press', 'path': 'Panel/btn'},
            {'do': 'type', 'path': 'Panel/field', 'text': '12.5abc'},
            {'do': 'press', 'path': 'Panel/deep'},
            {'do': 'press', 'path': 'Panel/relbtn'},
            {'do': 'press', 'path': 'Layers/a/go'},
            {'do': 'press', 'path': 'Layers/b/go'},
            {'do': 'press', 'path': 'Drawers/two'},
            {'do': 'answer', 'button': 'OK', 'optional': True, 'timeout_s': 0.5},
            {'do': 'press', 'path': 'Panel/popbtn'},
            {'do': 'answer', 'button': 'OK'},
            {'do': 'select', 'path': 'Panel/spin', 'value': '3x3'},
            {'do': 'choose', 'path': 'Panel/filebtn', 'file': '/tmp/walk.tsv'},
            {'do': 'read', 'path': 'Panel/field', 'props': ['text', 'disabled']},
            {'do': 'shot', 'name': 'after'},
        ],
    )
    assert result['outcome'] == 'done', result
    assert result['hits'] == [
        'btn',
        'commit 12.5',
        'deep',
        'rel',
        'a',
        'b',
        'ok',
        'spin 3x3',
        'file /tmp/walk.tsv',
    ], result
    assert any(r.strip() == 'SELECT DRAWER two' for r in result['records']), result['records']
    assert any(r.startswith('WALK SCRIPTED fragment') for r in result['records']), result['records']
    # An optional answer no popup asked for says so once, and claims no press.
    step_8 = [line for line in result['walk_lines'] if ' step 8 ' in line]
    assert step_8 == ['[SIM WALK  ] step 8 answer OK: no popup asked; optional, nothing pressed'], (
        step_8
    )
    assert result['reads'] == [
        {'step': 13, 'path': 'Panel/field', 'text': '12.5', 'disabled': False}
    ]
    assert len(result['shots']) == 1 and pathlib.Path(result['shots'][0]).is_file()


def test_an_uncommitted_entry_commits_in_the_frame_of_the_next_press(tmp_path):
    result = _run(
        tmp_path,
        [
            {'do': 'type', 'path': 'Panel/field', 'text': '7', 'commit': False},
            {'do': 'press', 'path': 'Panel/btn'},
        ],
    )
    assert result['outcome'] == 'done', result
    assert result['hits'] == ['commit 7', 'btn'], result
    assert result['frames']['commit'] == result['frames']['btn'], result


def test_a_name_with_two_live_widgets_stops_the_walk(tmp_path):
    result = _run(
        tmp_path, [{'do': 'press', 'path': 'Layer/go'}, {'do': 'press', 'path': 'Panel/btn'}]
    )
    assert result['outcome'].startswith('stopped at step 1'), result
    assert '2 live widgets' in result['outcome'], result
    assert result['hits'] == [], result


def test_a_press_inside_a_collapsed_drawer_stops_and_opens_nothing(tmp_path):
    result = _run(tmp_path, [{'do': 'press', 'path': 'Drawers/hidden'}])
    assert result['outcome'].startswith('stopped at step 1'), result
    assert 'collapsed' in result['outcome'], result
    assert result['two_collapsed'] is True, result
    assert not any('SELECT' in r for r in result['records']), result['records']


def test_a_press_under_an_open_modal_stops(tmp_path):
    result = _run(
        tmp_path, [{'do': 'press', 'path': 'Panel/modalbtn'}, {'do': 'press', 'path': 'Panel/btn'}]
    )
    assert result['outcome'].startswith('stopped at step 2'), result
    assert 'popup' in result['outcome'], result
    assert 'btn' not in result['hits'], result


def test_a_missing_control_stops_the_walk_and_no_later_step_runs(tmp_path):
    result = _run(
        tmp_path, [{'do': 'press', 'path': 'Panel/nosuch'}, {'do': 'press', 'path': 'Panel/btn'}]
    )
    assert result['outcome'].startswith('stopped at step 1'), result
    assert 'nosuch' in result['outcome'], result
    assert result['hits'] == [], result


def test_a_required_popup_that_never_comes_stops_the_walk(tmp_path):
    result = _run(tmp_path, [{'do': 'answer', 'button': 'OK', 'timeout_s': 0.5}])
    assert result['outcome'].startswith('stopped at step 1'), result


@pytest.mark.parametrize(
    'args, simulate, says',
    [
        (['--sim-walk={walk}'], False, '--sim-walk needs --simulate'),
        (['--sim-walk={walk}', '--sim-walk={walk}'], True, 'give --sim-walk once'),
        (['--sim-walk={bad}'], True, 'unknown action'),
    ],
)
def test_the_flag_is_refused_and_taken_out_of_argv(tmp_path, args, simulate, says):
    from ui.sim_walk_file import WalkFileError, take_walk_flag

    walk = tmp_path / 'walk.json'
    walk.write_text(json.dumps([{'do': 'press', 'path': 'Panel/btn'}]))
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps([{'do': 'dance'}]))
    argv = ['lumaviewpro.py', *[a.format(walk=walk, bad=bad) for a in args], '--other']
    with pytest.raises(WalkFileError, match=__import__('re').escape(says)):
        take_walk_flag(argv, simulate=simulate)
    assert argv == ['lumaviewpro.py', '--other']


def test_the_flag_gives_the_walk_and_leaves_argv_for_kivy(tmp_path):
    from ui.sim_walk_file import take_walk_flag

    walk = tmp_path / 'walk.json'
    walk.write_text(json.dumps([{'do': 'quit'}]))
    argv = ['lumaviewpro.py', f'--sim-walk={walk}']
    assert take_walk_flag(argv, simulate=True) == (walk.resolve(), [{'do': 'quit'}])
    assert argv == ['lumaviewpro.py']
    assert take_walk_flag(argv, simulate=True) is None


def test_reading_the_flag_imports_no_kivy():
    # lumaviewpro.py reads the flag before Kivy is imported, so a refusal
    # exits before a window opens.
    r = subprocess.run(
        [
            sys.executable,
            '-c',
            'import sys; import ui.sim_walk_file; '
            "print(sorted(m for m in sys.modules if m.split('.')[0] == 'kivy'))",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == '[]', r.stdout


def test_the_walk_file_parser_names_each_defect():
    from ui.sim_walk_file import WalkFileError, parse_walk

    for text, says in [
        ('{}', 'a list of steps'),
        ('[{"do": "press"}]', "step 1: 'press' needs path"),
        ('[{"do": "press", "path": "a", "colour": 1}]', "step 1: unknown key 'colour'"),
        ('[{"do": "choose", "path": "a"}]', "step 1: 'choose' needs file or cancel"),
        ('[{"do": "read", "path": "a", "props": "text"}]', 'props'),
    ]:
        with pytest.raises(WalkFileError, match=__import__('re').escape(says)):
            parse_walk(text, source='w.json')
    with pytest.raises(json.JSONDecodeError):
        parse_walk('not json', source='w.json')
