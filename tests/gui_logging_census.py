# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The one census of the GUI control surface: every control that binds a user
event, and the ``gui_logger`` record it reaches.

Every census before this one read ``ui/lumaviewpro.kv`` ONLY. Kivy also accepts
rules from ``Builder.load_string`` blocks inside ``.py`` modules, and those were
invisible to all of them -- the stated blind spot of the three passes behind
``GUI_LOGGING_GAP_FILL_PLAN_2026-09-08.md``. Both remaining unlogged controls
were hiding in it. :func:`kv_sources` is the closure: it enumerates the kv file
AND every production ``load_string`` block, so a census cannot silently shrink
back to one file.

The suite mocks Kivy, so ``kivy.lang.parser`` is not importable here. This reads
the kv as text and the handlers as AST -- narrower than a real parse tree, but
it pins the structure that regresses when someone moves or drops a line.

Resolution is the part a naive scan gets wrong, and each step below exists
because the rev-1 instrument was falsified without it:

- a handler may live on a BASE class, so methods resolve up the MRO;
- an emitter may sit behind a helper the handler calls, so calls are followed
  transitively (depth-limited) rather than read only at the top frame;
- a custom widget (``FolderChooseBTN``) emits from inside the widget class, not
  from the rule root, so ``self.x()`` resolves against the WIDGET and
  ``root.x()`` against the rule root.

Identity: a control is ``Class.id`` where it has an id, else
``Class.Widget(handlers:args)``. The literal argument matters -- two
``FileChooseBTN`` in ``CellCountControls`` both call ``choose`` and are
distinguishable only by the context they pass.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict

from tests.ast_seams import REPO_ROOT

# Kivy events only a person raises. Excludes lifecycle hooks (on_open,
# on_dismiss) and layout callbacks (on_size), which fire without anyone
# touching a control.
PERSON_EVENTS = (
    'on_release',
    'on_press',
    'on_text_validate',
    'on_focus',
    # PickSpinner's: raised only by a person choosing from the dropdown,
    # never by a write to the spinner's text.
    'on_pick',
)

# Kivy events a value change raises, whoever made it: a person operating the
# control, or the app writing to it. A person record reached from one of these
# says a person did what the app may have done.
VALUE_EVENTS = (
    'on_text',
    'on_active',
    'on_value',
    'on_state',
)

USER_EVENTS = PERSON_EVENTS + VALUE_EVENTS

EMITTERS = (
    'button',
    'toggle',
    'slider',
    'select',
    'frame_size',
    'protocol_action',
    'text_input',
    'notification',
    'popup_response',
    'window_event',
)

# The records that say a person did something to a control.
PERSON_RECORDS = ('button', 'toggle', 'slider', 'select', 'text_input')

# Production modules holding a Builder.load_string block. A new one must be
# added here or test_no_new_kv_source_escapes_the_census fails -- that test is
# what keeps the blind spot this census was built to close from reopening.
LOAD_STRING_MODULES = (
    'ui/advanced_settings.py',
    'ui/progress_popup.py',
    'ui/popup_close.py',
    'ui/range_slider.py',
)

KV_FILE = 'ui/lumaviewpro.kv'

# Classes whose methods can own an emitter: the rule roots and widget classes.
SEARCH_PACKAGES = ('ui',)
SEARCH_MODULES = ('lumaviewpro.py',)


def _indent(line: str) -> int:
    """Indentation width as Kivy's parser measures it (tab -> 4 spaces)."""
    prefix = line[: len(line) - len(line.lstrip(' \t'))]
    return len(prefix.replace('\t', '    '))


def _is_code(line: str) -> bool:
    """A kv line that carries structure -- not blank, not comment-only.

    The rev-1 instrument walked up to the nearest shallower line to find a
    widget header and landed on comments, which put four jog buttons under a
    header named ``Button: # COARSE LEFT``.
    """
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith('#')


def kv_sources():
    """``(label, text)`` for the kv file and every production load_string block."""
    sources = [(KV_FILE, (REPO_ROOT / KV_FILE).read_text())]
    for rel in LOAD_STRING_MODULES:
        src = (REPO_ROOT / rel).read_text()
        for match in re.finditer(r'Builder\.load_string\(\s*(?:"""|\'\'\')', src):
            quote = '"""' if src[match.end() - 3 : match.end()] == '"""' else "'''"
            sources.append((rel, src[match.end() : src.index(quote, match.end())]))
    return sources


def _class_index():
    """``name -> (relpath, ClassDef)`` for every class a kv rule can name."""
    index = {}
    paths = [REPO_ROOT / m for m in SEARCH_MODULES]
    for package in SEARCH_PACKAGES:
        paths += sorted((REPO_ROOT / package).glob('*.py'))
    for path in paths:
        rel = path.relative_to(REPO_ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover - a broken module fails elsewhere
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                index.setdefault(node.name, (rel, node, tree))
    return index


_CLASSES = _class_index()
_DEF_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)


def _mro(name, seen=None):
    """Class names in resolution order. A handler may live on a base class."""
    seen = seen or []
    if name in seen or name not in _CLASSES:
        return seen
    seen = [*seen, name]
    cls = _CLASSES[name][1]
    for base in cls.bases:
        if isinstance(base, ast.Name):
            seen = _mro(base.id, seen)
        elif isinstance(base, ast.Attribute):
            seen = _mro(base.attr, seen)
    return seen


def _resolve(cls_name, method):
    """``(relpath, FunctionDef)`` for ``method`` on ``cls_name`` or a base."""
    for name in _mro(cls_name):
        rel, cls, _tree = _CLASSES[name]
        for node in cls.body:
            if isinstance(node, _DEF_TYPES) and node.name == method:
                return rel, node
    return None, None


def _module_functions(rel):
    for _name, (owner_rel, _cls, tree) in _CLASSES.items():
        if owner_rel == rel:
            return {n.name: n for n in tree.body if isinstance(n, _DEF_TYPES)}
    return {}


def _record_name(call):
    """The record name a gui_logger call writes, or a marker for a computed one.

    ``<derived>`` is not a defect: a layer control names its record with its own
    channel suffix, and a jog button with its own direction. What the name must
    never be is another control's.
    """
    if not call.args:
        return '<derived>'
    first = call.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    if isinstance(first, ast.JoinedStr):
        return ''.join(
            part.value if isinstance(part, ast.Constant) else '{}' for part in first.values
        )
    return '<derived>'


def _direct_emitters(fn):
    found = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_gui_logger_call = (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == 'gui_logger'
            and func.attr in EMITTERS
        )
        if is_gui_logger_call:
            found.append(_record_name(node))
    return found


# How far to follow a handler's own calls looking for an emitter. Four hops
# reaches every emitter in the tree today; it is a bound on the walk, not a
# claim about the code.
_MAX_DEPTH = 4


def _reached_emitters(cls_name, method, depth=_MAX_DEPTH, seen=None):
    """Record names reachable from ``method``, following its calls."""
    seen = seen or set()
    key = (cls_name, method)
    if depth < 0 or key in seen:
        return []
    seen.add(key)
    rel, fn = _resolve(cls_name, method)
    if fn is None:
        return []
    names = list(_direct_emitters(fn))
    module_funcs = _module_functions(rel)
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == 'self'
        ):
            names += _reached_emitters(cls_name, func.attr, depth - 1, seen)
        elif isinstance(func, ast.Name) and func.id in module_funcs:
            names += _direct_emitters(module_funcs[func.id])
    return names


def _rule_root(lines, index):
    """The ``<Class>:`` rule the line at ``index`` belongs to."""
    for i in range(index, -1, -1):
        match = re.match(r'^<([A-Za-z_]\w*)(@\w+)?>:', lines[i])
        if match:
            return match.group(1)
    return None


def _blocks(label, text):
    """Every widget block in ``text`` that binds a user event."""
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines):
        match = re.match(r'^[ \t]*(on_\w+):\s*(.+)$', line)
        if not match or match.group(1) not in USER_EVENTS:
            continue
        prop_depth = _indent(line)
        header = i - 1
        while header >= 0 and (not _is_code(lines[header]) or _indent(lines[header]) >= prop_depth):
            header -= 1
        widget = lines[header].strip().rstrip(':') if header >= 0 else '?'
        widget = widget.split('#')[0].strip()
        header_depth = _indent(lines[header]) if header >= 0 else -1
        cursor, block = header + 1, []
        while cursor < len(lines):
            if _is_code(lines[cursor]) and _indent(lines[cursor]) <= header_depth:
                break
            block.append(lines[cursor])
            cursor += 1
        siblings = [b for b in block if _indent(b) == prop_depth]
        control_id = None
        for sibling in siblings:
            id_match = re.match(r'^[ \t]*id:\s*(\w+)', sibling)
            if id_match:
                control_id = id_match.group(1)
                break
        handler = match.group(2).strip()
        out.append(
            {
                'file': label,
                'id': control_id,
                'widget': widget,
                'root': _rule_root(lines, i),
                'event': match.group(1),
                'handler': handler,
                'line': i + 1,
                'root_methods': re.findall(r'root\.(\w+)\s*\(', handler),
                'self_methods': re.findall(r'self\.(\w+)\s*\(', handler),
                'args': re.findall(r"\.\w+\(\s*'([^']+)'", handler),
            }
        )
    return out


def census():
    """``identity -> {'static', 'derived', 'unresolved', 'file'}`` for every control.

    ``static`` is the set of literal record names the control reaches;
    ``derived`` the helpers it reaches that compute a name; ``unresolved`` any
    handler no class in the tree defines -- which must always be empty, since a
    kv binding naming a method that does not exist is silently dead.
    """
    bound = defaultdict(list)
    for label, text in kv_sources():
        for block in _blocks(label, text):
            args = ':'.join(sorted(set(block['args'])))
            if block['id']:
                identity = f'{block["root"]}.{block["id"]}'
            else:
                handlers = '+'.join(sorted(set(block['root_methods'] + block['self_methods'])))
                suffix = f':{args}' if args else ''
                identity = f'{block["root"]}.{block["widget"]}({handlers}{suffix})'
            bound[identity].append(block)

    out = {}
    for identity, blocks in bound.items():
        static, derived, unresolved = set(), set(), set()
        for block in blocks:
            pairs = [(block['root'], m) for m in block['root_methods']]
            pairs += [(block['widget'], m) for m in block['self_methods']]
            for owner, method in pairs:
                if owner is None or _resolve(owner, method)[1] is None:
                    unresolved.add(f'{owner}.{method}')
                    continue
                for name in _reached_emitters(owner, method):
                    if re.fullmatch(r'[A-Z][A-Z0-9_]*', name):
                        static.add(name)
                    else:
                        derived.add(method)
        out[identity] = {
            'static': sorted(static),
            'derived': sorted(derived),
            'unresolved': sorted(unresolved),
            'file': blocks[0]['file'],
        }
    out.update(accordion_items())
    out.update(pick_spinners())
    return out


# An accordion item binds no user event in the kv: Kivy expands it from its own
# touch handler, so the event scan above never saw one, and a drawer a person
# opened left no record. LoggedAccordionItem writes the record from that
# handler, under the group and item names its kv block sets.
LOGGED_ACCORDION_ITEM = 'LoggedAccordionItem'


def _accordion_kind(name):
    """'logged', 'plain' (an AccordionItem that records nothing), or None."""
    if name == 'AccordionItem':
        return 'plain'
    mro = _mro(name)
    if LOGGED_ACCORDION_ITEM in mro:
        return 'logged'
    for cls_name in mro:
        for base in _CLASSES[cls_name][1].bases:
            if getattr(base, 'id', getattr(base, 'attr', None)) == 'AccordionItem':
                return 'plain'
    return None


def _properties(lines, header):
    """``name -> value`` for the properties one level under the header line."""
    header_depth = _indent(lines[header])
    props, depth = {}, None
    for line in lines[header + 1 :]:
        if not _is_code(line):
            continue
        indent = _indent(line)
        if indent <= header_depth:
            break
        depth = depth or indent
        match = re.match(r'^[ \t]*(\w+):\s*(.+?)\s*$', line)
        if indent == depth and match:
            props[match.group(1)] = match.group(2).strip('\'"')
    return props


def accordion_items():
    """Every accordion item in the kv, as a control whose record is its group.

    A ``LoggedAccordionItem`` that sets both ``log_group`` and ``log_item``
    writes ``SELECT <log_group> <log_item>``; any other accordion item writes
    nothing and is reported unlogged, so the ratchet fails on it.
    """
    out = {}
    for label, text in kv_sources():
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if not _is_code(line):
                continue
            rule = re.match(r'^<([A-Za-z_]\w*)>:\s*$', line)
            widget = re.match(r'^[ \t]+([A-Za-z_]\w*):\s*$', line)
            name = rule.group(1) if rule else widget.group(1) if widget else None
            kind = _accordion_kind(name) if name else None
            if kind is None:
                continue
            props = _properties(lines, i)
            owner = name if rule else _rule_root(lines, i)
            identity = f'{owner}.{props.get("id", "<rule>")}'
            logged = kind == 'logged' and props.get('log_group') and props.get('log_item')
            out[identity] = {
                'static': [props['log_group']] if logged else [],
                'derived': [],
                'unresolved': [],
                'file': label,
            }
    return out


# A PickSpinner writes its own record from the dropdown's pick, the one path a
# person's choice takes, under the name its kv block sets; its handlers then
# act on the pick without recording it.
PICK_SPINNER = 'PickSpinner'


def pick_spinners():
    """Every PickSpinner in the kv, as a control whose record is its ``log_name``.

    One that sets no ``log_name`` writes no record from its pick and is
    reported unlogged, so the ratchet fails on it.
    """
    out = {}
    for label, text in kv_sources():
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if not _is_code(line):
                continue
            widget = re.match(r'^[ \t]+([A-Za-z_]\w*):\s*$', line)
            if not widget or widget.group(1) != PICK_SPINNER:
                continue
            props = _properties(lines, i)
            identity = f'{_rule_root(lines, i)}.{props.get("id", PICK_SPINNER)}'
            out[identity] = {
                'static': [props['log_name']] if props.get('log_name') else [],
                'derived': [],
                'unresolved': [],
                'file': label,
            }
    return out


def _person_record_calls(fn):
    """The ``gui_logger`` calls in ``fn`` that say a person did something."""
    return [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == 'gui_logger'
        and node.func.attr in PERSON_RECORDS
    ]


def _is_self_initializing(test):
    return (
        isinstance(test, ast.Attribute)
        and isinstance(test.value, ast.Name)
        and test.value.id == 'self'
        and test.attr == '_initializing'
    )


def _returns_while_initializing(fn):
    """Whether ``fn`` returns under ``if self._initializing:`` before any person record.

    The layer panel writes its own sliders with ``_initializing`` set, so a
    handler that returns on it records only what a person moved.
    """
    for stmt in fn.body:
        if (
            isinstance(stmt, ast.If)
            and _is_self_initializing(stmt.test)
            and any(isinstance(s, ast.Return) for s in stmt.body)
        ):
            return True
        if _person_record_calls(stmt):
            return False
    return False


def _reaches_person_record(cls_name, method, depth=_MAX_DEPTH, seen=None):
    """Whether ``method`` writes a person record, itself or through its own calls."""
    seen = seen or set()
    key = (cls_name, method)
    if depth < 0 or key in seen:
        return False
    seen.add(key)
    rel, fn = _resolve(cls_name, method)
    if fn is None:
        return False
    if _person_record_calls(fn):
        return True
    module_funcs = _module_functions(rel)
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id == 'self' and _reaches_person_record(
                cls_name, func.attr, depth - 1, seen
            ):
                return True
        elif (
            isinstance(func, ast.Name)
            and func.id in module_funcs
            and _person_record_calls(module_funcs[func.id])
        ):
            return True
    return False


def person_records_on_value_changes():
    """Every kv binding on a value change whose handler writes a person record.

    A value change fires for the app's write as much as for a person's, so a
    record written from one says a person did what the app may have done.
    ``file:line event: Class.method`` for each. Reads kv bindings only: a
    Python ``bind(text=...)`` and a helper in another module behind the
    handler are not followed.
    """
    out = []
    for label, text in kv_sources():
        for block in _blocks(label, text):
            if block['event'] not in VALUE_EVENTS:
                continue
            pairs = [(block['root'], m) for m in block['root_methods']]
            pairs += [(block['widget'], m) for m in block['self_methods']]
            for owner, method in pairs:
                fn = _resolve(owner, method)[1] if owner else None
                if fn is None or _returns_while_initializing(fn):
                    continue
                if _reaches_person_record(owner, method):
                    out.append(f'{label}:{block["line"]} {block["event"]}: {owner}.{method}')
    return sorted(out)


def _python_paths():
    paths = [REPO_ROOT / m for m in SEARCH_MODULES]
    for package in SEARCH_PACKAGES:
        paths += sorted((REPO_ROOT / package).glob('*.py'))
    return paths


# A widget's own touch handlers run only when a person touches it.
_TOUCH_HANDLERS = ('on_touch_down', 'on_touch_up', 'on_touch_move')


def _person_bound_methods():
    """Names of the methods a kv binding on a person-only event calls."""
    names = set()
    for label, text in kv_sources():
        for block in _blocks(label, text):
            if block['event'] in PERSON_EVENTS:
                names.update(block['root_methods'] + block['self_methods'])
    return names


# Receivers that are never a person's handler: the Session's members answer
# to the same names as the panels' handlers (``select_labware``,
# ``apply_tiling``), and ``gui_logger``'s own records (``frame_size``).
_NOT_A_HANDLER_RECEIVER = ('session', 'gui_logger')


def _receiver_names(func):
    names = []
    node = func.value if isinstance(func, ast.Attribute) else None
    while node is not None:
        if isinstance(node, ast.Attribute):
            names.append(node.attr)
            node = node.value
        elif isinstance(node, ast.Name):
            names.append(node.id)
            node = None
        else:
            node = getattr(node, 'value', None) or getattr(node, 'func', None)
    return names


def _top_level_functions(tree):
    """Module functions and class methods; a closure is part of the function it is in."""
    for node in tree.body:
        if isinstance(node, _DEF_TYPES):
            yield node
        elif isinstance(node, ast.ClassDef):
            yield from (n for n in node.body if isinstance(n, _DEF_TYPES))


def person_handlers_called_from_python():
    """Every call of a person's handler from code no person's touch reached.

    A function that writes a person record is a person's handler; called from
    anywhere but a person's own path -- a kv binding on a person-only event,
    another person's handler, a widget's touch handler -- its record says a
    person did what the app did. Matched by name, so a new caller fails
    closed. ``file:line caller -> name`` for each. A handler passed by
    reference or reached through ``getattr`` is not seen.
    """
    writers = set()
    for path in _python_paths():
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, _DEF_TYPES) and _person_record_calls(node):
                writers.add(node.name)
    person_paths = _person_bound_methods() | set(_TOUCH_HANDLERS)
    out = []
    for path in _python_paths():
        rel = path.relative_to(REPO_ROOT).as_posix()
        for caller in _top_level_functions(ast.parse(path.read_text())):
            if caller.name in person_paths:
                continue
            for node in ast.walk(caller):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, 'id', None)
                if name not in writers:
                    continue
                if set(_receiver_names(func)) & set(_NOT_A_HANDLER_RECEIVER):
                    continue
                out.append(f'{rel}:{node.lineno} {caller.name} -> {name}')
    return sorted(set(out))


def _text_written_ids(fn):
    """The ids whose ``.text`` ``fn`` assigns: ``<anything>.ids['x'].text = ...``."""
    ids = set()
    for node in ast.walk(fn):
        targets = node.targets if isinstance(node, ast.Assign) else []
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and target.attr == 'text'
                and isinstance(target.value, ast.Subscript)
                and isinstance(target.value.value, ast.Attribute)
                and target.value.value.attr == 'ids'
                and isinstance(target.value.slice, ast.Constant)
            ):
                ids.add(target.value.slice.value)
    return ids


def handlers_run_twice_by_a_write():
    """Every function that writes a widget's text and also calls that widget's ``on_text`` handler.

    The write dispatches the handler when the text changes, so the call runs
    it a second time. ``file:line function: id -> handler`` for each.
    """
    on_text = defaultdict(set)
    for label, text in kv_sources():
        for block in _blocks(label, text):
            if block['event'] == 'on_text' and block['id']:
                on_text[block['id']].update(block['root_methods'])
    out = []
    for path in _python_paths():
        rel = path.relative_to(REPO_ROOT).as_posix()
        for fn in _top_level_functions(ast.parse(path.read_text())):
            called = {
                node.func.attr
                for node in ast.walk(fn)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            }
            for widget_id in _text_written_ids(fn):
                for handler in sorted(on_text.get(widget_id, set()) & called):
                    out.append(f'{rel}:{fn.lineno} {fn.name}: {widget_id} -> {handler}')
    return sorted(out)
