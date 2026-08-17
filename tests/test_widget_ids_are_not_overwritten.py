"""A Kivy `ids` entry is a widget, and assigning over it is always a bug.

`VerticalControl.load_objective_from_settings` did
`self.ids['objective_spinner2'] = settings['objective_id']`, replacing the
spinner widget with a plain string. Every other site in the tree assigns to a
property of the widget (`.text`, `.state`, `.value`), which is what the author
meant. It never fired only because nothing ever called the method -- an
unfiltered census of the whole repo, including the single .kv file, found the
definition and no callers at all.

That made it a landmine rather than a live defect: the name reads exactly like
something a settings-restore path would want to call, and the first caller to
wire it up would have got `'str' object has no attribute 'text'` out of the
next objective selection. The method is deleted; this pins the shape so it
cannot return anywhere in the UI.
"""

import ast

from tests.ast_seams import iter_package_modules


def test_no_ui_module_assigns_over_a_widget_in_the_ids_dict():
    """Structural guard over `self.ids[...] = ...`, however it is spelled.

    Walks assignment targets rather than matching text, so a differently
    formatted or multi-target assignment cannot slip past.
    """
    offenders = []
    for rel_path, tree in iter_package_modules(['ui']):
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Subscript):
                    continue
                container = target.value
                if (
                    isinstance(container, ast.Attribute)
                    and container.attr == 'ids'
                    and isinstance(container.value, ast.Name)
                    and container.value.id == 'self'
                ):
                    offenders.append(f'{rel_path}:{node.lineno}')

    assert not offenders, (
        'assignment over a Kivy ids entry at: '
        + ', '.join(offenders)
        + '. ids holds widgets -- assign to the widget property that was meant '
        '(.text / .state / .value), never over the widget itself, which leaves '
        'every later attribute read on that id raising AttributeError.'
    )
