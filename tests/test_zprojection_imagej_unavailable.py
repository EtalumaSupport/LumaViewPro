# Copyright Etaluma, Inc.
"""Regression: Z-Projection names ImageJ/Java as the root cause when ImageJ
is unavailable.

init_ij ALWAYS returns an ImageJHelper -- even when Java is absent, the
helper just has no live ImageJ gateway (helper.available is False). The old
handling only checked `ij_helper is None` (which init_ij never produces), so
an unavailable helper sailed through and ran the projection; the user got a
generic "Failed to create Z-Projection" with no cause named, and the earlier
init-None branch additionally advised "try again" with no notification.

A live --simulate run on a Java-less machine confirmed the unavailable-helper
path, not the is-None path, is the one users hit. The fix adds
ImageJHelper.available and gates the projection on it, surfacing the real
cause (ImageJ could not start; Java likely missing) in both the popup and a
notification (Rule 14).

modules/imagej_helper.py and ui/post_processing.py import kivy / heavy deps,
so these are source-level guards on the gate, the wording, and the property.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PP_SRC = (REPO_ROOT / 'ui' / 'post_processing.py').read_text()
IJ_SRC = (REPO_ROOT / 'modules' / 'imagej_helper.py').read_text()


def _unavailable_gate_block():
    """The body of the post-init 'ImageJ unavailable' failure branch."""
    anchor = PP_SRC.index('ImageJ is not available')
    start = PP_SRC.rindex('if ctx.ij_helper is None', 0, anchor)
    end = PP_SRC.index('return', anchor) + len('return')
    return PP_SRC[start:end]


class TestAvailabilityFlag:
    def test_imagej_helper_exposes_available_property(self):
        # init_ij always returns a helper; callers need a way to ask whether
        # ImageJ actually initialized.
        assert 'def available(' in IJ_SRC
        assert 'self._ij is not None' in IJ_SRC


class TestUnavailableGate:
    def test_gate_checks_availability_not_just_none(self):
        # The is-None check alone never fires (init_ij never returns None);
        # the load-bearing condition is `not ctx.ij_helper.available`.
        block = _unavailable_gate_block()
        assert 'not ctx.ij_helper.available' in block

    def test_names_imagej_and_java_as_root_cause(self):
        block = _unavailable_gate_block()
        assert 'ImageJ could not start' in block
        assert 'Java is not installed' in block

    def test_fires_user_notification(self):
        # Rule 14: a failure that aborts the operation must notify.
        block = _unavailable_gate_block()
        assert 'notifications.error(' in block

    def test_misleading_try_again_message_removed(self):
        assert 'Failed to initialize ImageJ. Please try again.' not in PP_SRC
