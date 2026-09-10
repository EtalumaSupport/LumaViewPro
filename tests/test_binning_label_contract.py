"""The binning label is parsed, and a label that is not a factor REFUSES.

``modules/binning.py`` used to hold a hardcoded ``{'1x1': 1, '2x2': 2,
'4x4': 4}`` whitelist and answer 1 for every label outside it. That was a
second source of truth for which binning factors exist -- and the narrower
one. The offered labels come from the camera: the selector renders
``imaging.get_available_binning_sizes()`` as ``f'{s}x{s}'``, and an
unrecognized IDS body has its list widened live from the nodemap to
``(1, 2, 4, 8, 16)``.

So ``'8x8'`` was offerable on real hardware, converted silently to 1, passed
the support check against the already-defaulted integer, and was stored
verbatim -- an 8x8 selection captured UNBINNED while the screen read 8x8.

Two halves are pinned here:

1. The parse accepts a positive SQUARE label and refuses everything else as
   ``ConfigError`` -- and as nothing else. The refusals are not academic:
   the retired table was silently containing ``'0x0'`` (a divide-by-zero in
   ``native_to_displayed``), ``'-1x-1'`` (a negative native ROI that
   ``_store_native_roi`` would persist as the source of truth) and ``'2x4'``
   (not a single factor at all).
2. The exception TYPE is a contract, not an implementation detail. Callers
   boundary on ``ConfigError`` alone, so a leaked ``ValueError`` from a bare
   ``int()`` or an ``AttributeError`` from ``.split`` on a non-string would
   walk straight past every one of them into the host's re-raise -- and the
   container shape check lets non-string values through, so that is
   reachable rather than theoretical.
"""

import ast

import pytest

import modules.binning as binning
import tests.ast_seams as ast_seams
from modules.exceptions import ConfigError


class TestTheParseAcceptsWhatTheCameraCanOffer:
    def test_every_label_the_selector_can_render_converts(self):
        # The selector builds its values as f'{s}x{s}' over the camera's own
        # list. These are the factors the curated profiles and the generic
        # IDS widening between them can produce.
        for factor in (1, 2, 3, 4, 8, 16):
            assert binning.binning_size_str_to_int(f'{factor}x{factor}') == factor

    def test_the_three_labels_the_retired_table_carried_are_unchanged(self):
        # Whatever else moves, the parse must agree with the retired map on
        # the entire set the map actually held -- that is what makes this a
        # fix rather than a behaviour change for existing users.
        assert binning.binning_size_str_to_int('1x1') == 1
        assert binning.binning_size_str_to_int('2x2') == 2
        assert binning.binning_size_str_to_int('4x4') == 4


class TestTheParseRefusesWhatIsNotAFactor:
    @pytest.mark.parametrize(
        'label',
        [
            '0x0',  # native_to_displayed would raise ZeroDivisionError
            '-1x-1',  # a negative native ROI, persisted as the source of truth
            '2x4',  # not square, so not a single binning factor
            'Select',  # the kv placeholder
            '',
            '2x2x2',
        ],
    )
    def test_a_label_that_is_not_a_positive_square_raises_config_error(self, label):
        with pytest.raises(ConfigError):
            binning.binning_size_str_to_int(label)

    @pytest.mark.parametrize('value', [4, None, 2.0, ['2x2']])
    def test_a_non_string_raises_config_error_not_attribute_error(self, value):
        # _check_container_shape compares container KIND only, so a
        # "size": 4 or a "size": null in current.json reaches the converter
        # intact. A bare .split() here would leak AttributeError past every
        # except ConfigError boundary in the app.
        with pytest.raises(ConfigError):
            binning.binning_size_str_to_int(value)

    @pytest.mark.parametrize('label', ['0x0', '-1x-1', '2x4', 'Select', '', '2x2x2', 4, None, 2.0])
    def test_no_other_exception_type_escapes(self, label):
        # The type IS the contract. Assert it directly rather than relying on
        # pytest.raises(ConfigError) above, which would also pass for a
        # subclass raised from somewhere unintended.
        try:
            binning.binning_size_str_to_int(label)
        except ConfigError:
            pass
        except Exception as exc:
            pytest.fail(f'{label!r} leaked {type(exc).__name__}, not ConfigError: {exc}')
        else:
            pytest.fail(f'{label!r} was accepted; it is not a positive square label')


class TestTheFormatterDoesNotRaise:
    def test_it_formats_every_factor_including_ones_no_table_carried(self):
        assert [binning.binning_size_int_to_str(v) for v in (1, 2, 3, 4, 8, 16)] == [
            '1x1',
            '2x2',
            '3x3',
            '4x4',
            '8x8',
            '16x16',
        ]

    def test_it_round_trips_with_the_parse(self):
        for factor in (1, 2, 3, 4, 8, 16):
            label = binning.binning_size_int_to_str(factor)
            assert binning.binning_size_str_to_int(label) == factor


class TestTheWhitelistIsGone:
    def test_no_hardcoded_label_table_remains(self):
        # A second source of truth for which factors exist is the defect
        # itself; software binning is about to add 3x3, and a table here is
        # exactly what would make that a hand-edit.
        assert not hasattr(binning, 'BINNING_SIZE_MAP'), (
            'the label whitelist is back -- the camera owns which factors exist'
        )


class TestTheHostBoundsTheFailure:
    """``ConfigError`` reaches the GUI host, and the host comes up anyway.

    ``ScopeInitConfig.from_settings`` converts the stored label during
    ``ScopeSession.create``, which ``build()`` calls with nothing above it
    that catches -- its only handler logs and RE-RAISES. ``ConfigError``
    derives from ``Exception``, not ``ValueError``, so ``settings_init``'s
    corrupt-current.json fallback does not catch it either. Without a
    boundary here a malformed stored label does not degrade the app, it
    stops the app from launching at all.
    """

    @staticmethod
    def _build_def():
        return ast_seams.find_def('lumaviewpro.py', 'build', class_name='LumaViewProApp')

    def _config_error_handler(self):
        build = self._build_def()
        assert build is not None
        return next(
            h
            for h in ast.walk(build)
            if isinstance(h, ast.ExceptHandler)
            and isinstance(h.type, ast.Name)
            and h.type.id == 'ConfigError'
        )

    def test_build_catches_config_error(self):
        build = self._build_def()
        assert build is not None
        caught = [
            handler
            for handler in ast.walk(build)
            if isinstance(handler, ast.ExceptHandler)
            and isinstance(handler.type, ast.Name)
            and handler.type.id == 'ConfigError'
        ]
        assert caught, (
            'build() must bound ConfigError around the session compose; '
            'without it a malformed stored value stops the app launching'
        )

    def test_the_recovery_delegates_to_the_one_settings_policy(self):
        # The host must not restate the recovery. settings_init owns it, and
        # owns the half that is easy to miss: marking the session provisional
        # so nothing is saved over the only copy of the user's configuration.
        # A hand-rolled template read here would come up on the right values
        # with saving still enabled.
        handler = self._config_error_handler()
        called = [
            call.func.id
            for call in ast.walk(handler)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        ]
        assert 'fall_back_to_template' in called, (
            'the recovery delegates to the settings policy; the host renders, '
            f'it does not decide. Calls found: {called}'
        )

    def test_the_recovery_does_not_rebind_the_settings_store(self):
        # Every module that imported the settings dict holds its own name
        # bound to the same object. Rebinding the name here would leave all
        # of them on the rejected values while only this scope saw the
        # template -- and it is a reference-before-assignment besides, since
        # the name is a module global read earlier in this same function.
        handler = self._config_error_handler()
        rebound = [
            target.id
            for node in ast.walk(handler)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and target.id == 'settings'
        ]
        assert rebound == [], 'the recovery rebinds `settings` instead of republishing the store'

    def test_the_recovery_writes_nothing(self):
        handler = self._config_error_handler()
        writers = [
            call.func.attr
            for call in ast.walk(handler)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr in ('save_settings', 'write', 'dump', 'save')
        ]
        assert writers == [], f'the recovery path persists something: {writers}'
