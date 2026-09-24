"""The camera gain cap enters the app without a float tail.

A GenICam float node reports its maximum in continuous units, and that value
can carry a tail past the last real step -- a 48 dB gain node reports
48.00000004350822. The driver publishes it raw whenever the node declares no
fixed increment, deliberately: inventing a step would narrow a range the
camera did not narrow, and a test in test_pylon_published_ranges.py pins that
behaviour.

Raw, that value becomes the gain slider's maximum, the ceiling a typed entry
is clamped to, and -- when a layer's stored gain is reconciled down to the cap
-- the number written into the store. Observed on a Basler daA3840-45um:
typing 5757 into the gain box applied 48.00000004350822, which is what reaches
current.json and the box.

camera_max_gain_for_ui is the single point the whole chain passes through, so
it is where the value is normalised, at the precision the existing owner
already declares for gain rather than at a second number chosen locally.
"""

from types import SimpleNamespace

from modules import common_utils
from modules.config_helpers import DEFAULT_MAX_GAIN_DB, camera_max_gain_for_ui

# The value a real Basler daA3840-45um published for a 48 dB gain node.
REPORTED_TAIL_CAP = 48.00000004350822


class TestTheGainCapCarriesNoFloatTail:
    def test_a_reported_tail_does_not_reach_the_caller(self):
        imaging = SimpleNamespace(max_gain_db_cached=REPORTED_TAIL_CAP)

        assert camera_max_gain_for_ui(imaging) == 48.0

    def test_the_normalised_cap_is_not_above_what_the_camera_published(self):
        """A bound must stay a value the camera takes, so normalising may move
        it inward but never outward past the reported ceiling by anything the
        camera could notice."""
        imaging = SimpleNamespace(max_gain_db_cached=REPORTED_TAIL_CAP)

        resolved = camera_max_gain_for_ui(imaging)

        # The driver short-circuits a gain write within 1e-3 dB, so any
        # residual overshoot has to sit far below that to be unobservable.
        assert resolved - REPORTED_TAIL_CAP < 1e-4

    def test_a_cap_already_at_precision_is_untouched(self):
        imaging = SimpleNamespace(max_gain_db_cached=24.0)

        assert camera_max_gain_for_ui(imaging) == 24.0

    def test_the_no_camera_default_still_applies_and_is_clean(self):
        imaging = SimpleNamespace(max_gain_db_cached=None)

        assert camera_max_gain_for_ui(imaging) == DEFAULT_MAX_GAIN_DB

    def test_the_precision_comes_from_the_existing_owner(self):
        """Not a second constant. If the owner's precision for gain changes,
        this resolver changes with it rather than disagreeing with it."""
        imaging = SimpleNamespace(max_gain_db_cached=REPORTED_TAIL_CAP)

        expected = round(REPORTED_TAIL_CAP, common_utils.max_decimal_precision('gain'))

        assert camera_max_gain_for_ui(imaging) == expected
