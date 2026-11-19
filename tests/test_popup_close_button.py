# Copyright Etaluma, Inc.
"""Regression guards for the app-wide popup close (X) button.

Kivy's stock Popup ships no close affordance. ui/popup_close.py injects an X
into every popup's title row from a global <Popup> on_open rule. The runtime
behavior (rule fires on open, X injected idempotently, X calls dismiss) was
verified against a real Kivy Popup outside the suite -- the suite mocks kivy
(see tests/conftest.py), so these are source-level guards on the invariants
that would silently regress.

The load-bearing guard is test_does_not_use_crashing_import_form: the global
rule MUST reference the module form (`#:import popup_close ui.popup_close`),
not a dotted attribute path (`#:import x ui.popup_close.add_popup_close`).
The dotted-attribute form hard-crashes the interpreter when resolved during
the module's own import -- no traceback, the process just dies.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
POPUP_CLOSE_SRC = (REPO_ROOT / 'ui' / 'popup_close.py').read_text()


class TestInjectorInvariants:
    def test_defines_injector(self):
        assert 'def add_popup_close(' in POPUP_CLOSE_SRC

    def test_idempotency_guard_present(self):
        # on_open fires on every open; the injector must no-op after the first.
        assert '_etaluma_close_added' in POPUP_CLOSE_SRC

    def test_close_button_dismisses(self):
        assert 'on_release' in POPUP_CLOSE_SRC
        assert 'popup.dismiss()' in POPUP_CLOSE_SRC

    def test_uses_ascii_x_glyph(self):
        # Rule 24: ASCII only. The close glyph is a plain capital X.
        assert "text='X'" in POPUP_CLOSE_SRC


class TestGlobalRuleWiring:
    def test_global_popup_rule_registered(self):
        assert '<Popup>:' in POPUP_CLOSE_SRC
        assert 'on_open:' in POPUP_CLOSE_SRC

    def test_does_not_use_crashing_import_form(self):
        # The dotted-attribute #:import form crashes the interpreter when
        # resolved during ui.popup_close's own import. Must use module form.
        assert 'ui.popup_close.add_popup_close' not in POPUP_CLOSE_SRC
        assert '#:import popup_close ui.popup_close' in POPUP_CLOSE_SRC

    def test_imported_at_startup_for_side_effect(self):
        src = (REPO_ROOT / 'lumaviewpro.py').read_text()
        assert 'import ui.popup_close' in src


class _ContainerProbe:
    """A popup's internal container that records whether the injector reached it."""

    def __init__(self):
        self.reached = False

    @property
    def parent(self):
        self.reached = True
        return None


class TestAPopupThatMustBeAnswered:
    def test_a_popup_that_must_be_answered_gets_no_x(self):
        from types import SimpleNamespace

        from ui.popup_close import add_popup_close

        probe = _ContainerProbe()
        popup = SimpleNamespace(must_answer=True, _container=probe)

        add_popup_close(popup)

        assert not probe.reached
        assert not getattr(popup, '_etaluma_close_added', False)

    def test_the_closing_popup_gets_no_x(self, monkeypatch):
        """The app's Closing popup cannot be dismissed while the close runs.

        Dismissed, it left the window looking live, accepting settings,
        while the close went on unseen and every later window close was
        ignored.
        """
        from ui import notification_popup
        from ui.popup_close import add_popup_close

        probe = _ContainerProbe()

        # Stand-ins by design: the suite's mocked Kivy widgets have no bind,
        # add_widget or open. The popup opens as the global <Popup> on_open
        # rule does, through the real injector.
        class _Widget:
            def __init__(self, **kw):
                self.__dict__.update(kw)
                self.parent = None

            def bind(self, **_handlers):
                pass

            def add_widget(self, widget):
                widget.parent = self

            def remove_widget(self, widget):
                widget.parent = None

        class _Popup(_Widget):
            def __init__(self, **kw):
                super().__init__(**kw)
                self._container = probe

            def open(self):
                add_popup_close(self)

        for name in ('BoxLayout', 'Label', 'Button'):
            monkeypatch.setattr(notification_popup, name, _Widget)
        monkeypatch.setattr(notification_popup, 'Popup', _Popup)

        notification_popup.show_blocking_progress_popup(
            title='Closing',
            message='Closing LumaViewPro...',
            action_text='Discard Remaining Video Frames',
            on_action=lambda: None,
            action_shown=False,
        )

        assert not probe.reached, 'the Closing popup was given an X that hides the close'

    def test_an_ordinary_popup_still_goes_on_to_get_its_x(self):
        from types import SimpleNamespace

        from ui.popup_close import add_popup_close

        probe = _ContainerProbe()
        add_popup_close(SimpleNamespace(_container=probe))

        assert probe.reached
