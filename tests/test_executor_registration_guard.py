# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression: registering executors over a live scope's existing,
different handles must refuse instead of silently swapping.

The executor-backed dispatch (scope.X_async / X_sync) reads the
registered handles per call. A second registration with different
executors -- e.g. a hosting environment creating a session against an
already-composed scope -- would silently reroute all dispatch and
produce no symptom until a protocol fence is bypassed on the wrong
lane. Re-registering the SAME handles stays idempotent; a deliberate
rewire says replace=True.
"""

from unittest.mock import MagicMock

import pytest

from modules.lumascope_api import Lumascope


@pytest.fixture
def scope():
    s = Lumascope(simulate=True)
    yield s
    s.disconnect()


class TestExecutorRegistrationGuard:
    def test_first_registration_is_plain(self, scope):
        scope.register_executors(
            camera_executor=MagicMock(), io_executor=MagicMock(), file_io_executor=MagicMock()
        )

    def test_same_handles_reregister_idempotently(self, scope):
        camera, io, file_io = MagicMock(), MagicMock(), MagicMock()
        scope.register_executors(camera_executor=camera, io_executor=io, file_io_executor=file_io)
        scope.register_executors(camera_executor=camera, io_executor=io, file_io_executor=file_io)

    def test_different_handles_refuse_without_replace(self, scope):
        scope.register_executors(
            camera_executor=MagicMock(), io_executor=MagicMock(), file_io_executor=MagicMock()
        )
        with pytest.raises(RuntimeError, match=r'already\s+registered'):
            scope.register_executors(
                camera_executor=MagicMock(),
                io_executor=MagicMock(),
                file_io_executor=MagicMock(),
            )

    def test_replace_true_rewires_deliberately(self, scope):
        scope.register_executors(
            camera_executor=MagicMock(), io_executor=MagicMock(), file_io_executor=MagicMock()
        )
        new_io = MagicMock()
        scope.register_executors(
            camera_executor=MagicMock(),
            io_executor=new_io,
            file_io_executor=MagicMock(),
            replace=True,
        )
        assert scope._io_executor is new_io
