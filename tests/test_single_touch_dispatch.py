"""One click reaches a widget's handler once.

ShaderViewer sat beside the side panels in the widget tree and, on a
touch landing over one of them, dispatched that touch into the panel by
hand. The normal walk reaches those panels BEFORE it reaches
ShaderViewer, so the panel had already received the touch: every handler
beneath it ran twice per click.

Measured in the simulator 2026-09-14 with a probe in Stage.on_touch_down
-- four clicks produced eight deliveries, each pair carrying the same
touch uid and the same widget id, and the stage issued two identical
absolute moves per click ~3 ms apart. The second move restarted motion
while the first was still settling, so any position sampled in that
window was read in flight.

Kivy's ButtonBehavior opens with `if self in touch.ud: return False`, so
buttons absorbed the second delivery; the Stage, which turns a touch
into a move rather than a value, did not. That is why the defect showed
on the stage map and nowhere obvious.
"""

import ast
import inspect
from pathlib import Path

import ui.mod_slider
import ui.shader

UI = Path(__file__).resolve().parent.parent / 'ui'


def _method(module, class_name, method_name):
    """The AST of one method, read from the module's own source file."""
    tree = ast.parse(Path(inspect.getsourcefile(module)).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f'{class_name}.{method_name} not found')


def _dispatch_calls(fn_node):
    """Calls that hand a touch to another widget's touch handler.

    `w.on_touch_down(touch)` re-delivers; `super().on_touch_down(touch)`
    is this widget's own inherited behaviour and is not a re-delivery.
    """
    found = []
    for node in ast.walk(fn_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if not func.attr.startswith('on_touch_'):
            continue
        if isinstance(func.value, ast.Call) and getattr(func.value.func, 'id', None) == 'super':
            continue
        found.append(f'{ast.unparse(func)}(...)')
    return found


def test_shaderviewer_declines_a_panel_touch_instead_of_redelivering_it():
    """The panel already has the touch; dispatching again doubles it."""
    node = _method(ui.shader, 'ShaderViewer', 'on_touch_down')

    assert _dispatch_calls(node) == [], (
        'ShaderViewer.on_touch_down re-dispatches a touch into another '
        'widget. The tree reaches the side panels before this widget, so '
        'that delivers the touch a second time and every handler beneath '
        'runs twice per click.'
    )


def test_both_blocker_branches_resolve_the_same_way():
    """The scroll branch always declined correctly; the other did not.

    Two arms of one method answering one question two ways is what
    produced this, so the pin is that they agree -- not that one of them
    contains a particular line.
    """
    source = inspect.getsource(ui.shader.ShaderViewer.on_touch_down)

    blocker_loops = source.count('for w in ZOOM_BLOCKERS:')
    assert blocker_loops == 2, f'expected both branches, found {blocker_loops}'
    assert 'return w.on_touch_down(touch)' not in source


def test_modslider_does_not_run_its_parent_handler_twice_per_move():
    """One drag step ran Slider.on_touch_move twice.

    Harmless in effect -- a slider sets its value from the touch position,
    so repeating it lands on the same value -- which is exactly why it
    survived unnoticed. It is the same family as the ShaderViewer defect.
    """
    node = _method(ui.mod_slider, 'ModSlider', 'on_touch_move')

    super_moves = [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == 'on_touch_move'
        and isinstance(n.func.value, ast.Call)
        and getattr(n.func.value.func, 'id', None) == 'super'
    ]

    assert len(super_moves) == 1, (
        f'super().on_touch_move is called {len(super_moves)} times; one drag '
        f'step must run the parent handler once'
    )


def test_no_ui_touch_handler_redelivers_a_touch():
    """A census, so a new re-dispatch anywhere in ui/ lands here.

    A whole-directory sweep rather than a list of known sites: the defect
    is the PATTERN, and a list would go stale the moment someone adds a
    handler. Reads source rather than importing, so it costs nothing and
    needs no Kivy window.
    """
    offenders = []
    for path in sorted(UI.glob('*.py')):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if not isinstance(item, ast.FunctionDef):
                    continue
                if not item.name.startswith('on_touch_'):
                    continue
                for call in _dispatch_calls(item):
                    offenders.append(f'{path.name}:{node.name}.{item.name} -> {call}')

    assert offenders == [], (
        f'these handlers re-deliver a touch the normal walk already delivers: {offenders}'
    )


def test_the_sweep_would_actually_catch_a_re_dispatch():
    """The census only means something if it can fail.

    A sweep that silently matches nothing -- a renamed method prefix, a
    broken AST walk -- passes forever and proves nothing.
    """
    planted = ast.parse(
        'class W:\n    def on_touch_down(self, touch):\n        return other.on_touch_down(touch)\n'
    )
    fn = planted.body[0].body[0]

    assert _dispatch_calls(fn) == ['other.on_touch_down(...)']
