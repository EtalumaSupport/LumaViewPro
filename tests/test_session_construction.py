# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The Session composes and decomposes the instrument; no host keeps a copy.

AST pins over ``tests.ast_seams`` (never a direct source read: the
source-pin ratchet counts those). Each pin names one copy of the
bring-up or the teardown that used to live in a host and the reason it
must not come back.
"""

import tests.ast_seams as ast_seams


class TestNoHostCopyOfTheBringUp:
    def test_microscope_settings_has_no_reconnect_handler(self):
        # The handler was an unbound 131-line copy of the settings-to-scope
        # bring-up: it took the registry's unvalidated explicit-name path
        # and stopped the display thread last, after the scope swap. The
        # scope-swap seams it called (set_scope, rebind) stay for the
        # auto-reconnect item; a new handler would be a second bring-up.
        node = ast_seams.find_def(
            'ui/microscope_settings.py', 'reconnect', class_name='MicroscopeSettings'
        )
        assert node is None, f'MicroscopeSettings.reconnect is back at line {node.lineno}'
