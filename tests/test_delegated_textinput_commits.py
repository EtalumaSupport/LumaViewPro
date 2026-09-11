"""A delegated TextInput must reach the store before any button can read it.

The display-only migration points `modules/` at the settings store instead of
the Kivy widget tree. For a SPINNER that is safe: its `on_text` fires the
moment the user picks a value, so the store is current by the time anything
reads it.

A TextInput is different, and the difference is a real defect rather than a
style question. Kivy dispatches a button's handler and the focus-loss commit
in this order (`kivy/core/window/__init__.py`):

    elif etype == 'end':
        self.dispatch('on_touch_up', me)             # the button's on_release
        FocusBehavior._handle_post_on_touch_up(me)   # the TextInput unfocus

So a field that commits only on enter or focus loss is still holding the
user's typed value when the button handler runs. Delegating the read to the
store without an `on_text` commit means Run Protocol, Save Protocol and New
Protocol each read the PREVIOUS schedule while the screen shows the new one --
and Save and New Protocol persist it.

These tests pin the invariant, not the gesture: the binding exists, and the
commit it names writes the store without notifying. A Kivy touch cannot be
simulated in this suite, so asserting on the wiring is what is available --
and it is the half that gets forgotten when the next TextInput is delegated.
"""

import ast
import re

import tests.ast_seams as ast_seams

KV = 'ui/lumaviewpro.kv'
PROTOCOL_SETTINGS = 'ui/protocol_settings.py'

# Widget id -> the handler its on_text must call. Every entry is a TextInput
# whose value a store-reading getter consumes.
DELEGATED_TEXT_INPUTS = {
    'capture_period': 'commit_period',
    'capture_dur': 'commit_duration',
}


def _kv_block(widget_id: str) -> str:
    """The kv lines belonging to one widget id, up to the next id: line."""
    src = (ast_seams.REPO_ROOT / KV).read_text()
    start = src.index(f'id: {widget_id}\n')
    nxt = src.find('id: ', start + 1)
    return src[start : nxt if nxt != -1 else len(src)]


class TestTheBindingExists:
    def test_each_delegated_text_input_commits_on_text(self):
        for widget_id, handler in DELEGATED_TEXT_INPUTS.items():
            block = _kv_block(widget_id)
            assert re.search(rf'on_text:\s*root\.{handler}\(\)', block), (
                f'{widget_id} has no on_text commit, so a button handler can read '
                f'the store before the typed value reaches it'
            )

    def test_the_commit_is_not_the_reporting_handler(self):
        # Pointing on_text at update_period would fire its notifications per
        # keystroke. Worse than noise: the notification bus dedups on
        # (category, title), so a spurious "not a number" from a transient
        # empty field takes the slot and SWALLOWS the real sub-second clamp
        # warning that follows it.
        for widget_id in DELEGATED_TEXT_INPUTS:
            block = _kv_block(widget_id)
            assert not re.search(r'on_text:\s*root\.update_', block), (
                f'{widget_id} commits through its reporting handler'
            )


class TestTheCommitIsSilentAndWritesTheStore:
    @staticmethod
    def _commit_def(name):
        return ast_seams.find_def(PROTOCOL_SETTINGS, name, class_name='ProtocolSettings')

    def test_each_commit_writes_the_settings_store(self):
        for handler in DELEGATED_TEXT_INPUTS.values():
            node = self._commit_def(handler)
            assert node is not None, f'{handler} does not exist'
            subscripts = [
                n
                for n in ast.walk(node)
                if isinstance(n, ast.Assign)
                for t in n.targets
                if isinstance(t, ast.Subscript)
            ]
            assert subscripts, f'{handler} never writes the store'

    def test_each_commit_notifies_nobody(self):
        # The enter / focus-loss path still reports a value that never parses.
        for handler in DELEGATED_TEXT_INPUTS.values():
            node = self._commit_def(handler)
            called = [
                n.func.attr
                for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            ]
            assert 'warning' not in called and 'error' not in called, (
                f'{handler} notifies; per-keystroke warnings consume the dedup '
                f'slot and suppress the legitimate one'
            )

    def test_the_store_write_is_outside_the_parse_guard(self):
        # An absent `protocol` container is a broken configuration, not a
        # typing error. The template ships the key and the merge fills it, so
        # swallowing a KeyError here would hide a real one.
        for handler in DELEGATED_TEXT_INPUTS.values():
            node = self._commit_def(handler)
            for handler_node in ast.walk(node):
                if not isinstance(handler_node, ast.Try):
                    continue
                writes = [
                    n
                    for n in ast.walk(handler_node)
                    if isinstance(n, ast.Assign)
                    for t in n.targets
                    if isinstance(t, ast.Subscript)
                ]
                assert writes == [], (
                    f'{handler} writes the store inside its parse guard, so a '
                    f'missing container is swallowed as a typing error'
                )
