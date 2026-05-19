"""Regression tests for the set_exposure_time WARNING threshold.

Bug shape: ``modules/lumascope_api/_lumascope.py::set_exposure_time``
emitted a WARNING at threshold ``t < 0.1`` ms saying "image will be
nearly black. Value should be in milliseconds." But Pylon physical
ExposureTime minimum is 10-35 us across Basler USB3 sensors, and
bright-field captures legitimately use 0.03 ms (30 us). The warning
fired on every BF capture and every protocol BF step (multiple times
per scan in Chris's beta11 logs), generating user-visible log noise
for fully valid values.

Rule 20 (logs accurate): the "nearly black" wording was wrong --
Pylon silently clamps below the sensor's physical minimum; the image
is at-minimum, not zero.

Fix shape: lower threshold to 0.005 ms (5 us), below any Basler
sensor physical minimum. The warning now fires only for genuinely
impossible values (unit-confusion bugs). Wording corrected to name
the actual behavior (clamping) and prompt the user to verify units.

These tests lock the threshold and wording so a future bump back to
0.1 ms or a revert of the wording trips immediately.
"""

from __future__ import annotations

import ast
import pathlib


def _set_exposure_time_source() -> str:
    # Phase 4d relocated set_exposure_time's body from _lumascope.py to imaging.py.
    src_path = (pathlib.Path(__file__).resolve().parent.parent
                / "modules" / "lumascope_api" / "imaging.py")
    source = src_path.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "set_exposure_time":
            text = ast.get_source_segment(source, node)
            assert text is not None
            return text
    raise AssertionError("set_exposure_time not found in imaging.py")


class TestSetExposureTimeWarningThreshold:
    """Lock the lower-than-100us threshold so a regression to 0.1 ms
    (which fires on every legitimate bright-BF capture) trips."""

    def test_threshold_is_at_or_below_physical_minimum(self):
        body = _set_exposure_time_source()
        # Confirm the warning gate is at 0.005 ms or tighter.
        # Higher thresholds would re-introduce false positives on
        # bright-BF values (0.03 ms is real-hardware bench-validated
        # via Chris's beta11 logs).
        tree = ast.parse(body)
        thresholds = []
        for node in ast.walk(tree):
            # Walk every Compare in the function body, find any
            # `t < <Num>` form
            if isinstance(node, ast.Compare):
                if (len(node.ops) == 1
                        and isinstance(node.ops[0], ast.Lt)
                        and isinstance(node.left, ast.Name)
                        and node.left.id == "t"
                        and isinstance(node.comparators[0], ast.Constant)
                        and isinstance(node.comparators[0].value, (int, float))):
                    thresholds.append(node.comparators[0].value)
        assert thresholds, (
            "set_exposure_time must contain a `t < <threshold>` "
            "comparison for the sanity-check warning gate."
        )
        # The new threshold must be no higher than 0.01 ms (10 us).
        # Pylon physical min is 10-35 us across Basler USB3 sensors;
        # 0.03 ms is legitimate bright-BF. 0.1 ms (the old threshold)
        # fires on legitimate values.
        for thresh in thresholds:
            assert thresh <= 0.01, (
                f"set_exposure_time warning threshold is {thresh} ms; "
                f"must be <= 0.01 ms (10 us) to avoid false positives "
                f"on bright-BF values like 0.03 ms (30 us) that the "
                f"camera handles natively."
            )

    def test_warning_no_longer_claims_image_will_be_nearly_black(self):
        body = _set_exposure_time_source()
        # The old wording "image will be nearly black" was factually
        # wrong: Pylon silently clamps below the sensor minimum, so
        # the captured image is at minimum exposure, not zero. Lock
        # the corrected wording so a revert is caught.
        assert "nearly black" not in body, (
            "set_exposure_time warning must not say 'nearly black' -- "
            "Pylon silently clamps below the physical minimum, so the "
            "captured image is at the sensor minimum, not zero. Rule 20."
        )

    def test_warning_text_describes_clamping_behavior(self):
        body = _set_exposure_time_source()
        # New wording must mention the clamping behavior so the user
        # understands what actually happens at impossible values.
        assert "clamp" in body.lower(), (
            "set_exposure_time warning should describe the clamping "
            "behavior so the user understands what happens at "
            "below-minimum values."
        )
