# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression: PylonCamera._has_node treats a missing node as a quiet False.

pypylon's GetNode RAISES genicam.LogicalErrorException for a node that is not
in the map -- it does not return None. Probing an optional Basler feature
(BslConversionGainMode, BslLineNoiseReduction*) on a camera that lacks it --
a Basler dart has no Bsl* nodes -- is the EXPECTED "unsupported" case and must
report False without logging an error or letting the exception escape.

Before the fix, the capability probes did ``GetNode(name) is not None``, which
assumed None-on-missing; the raise instead fell through to a broad
``except Exception`` that logged an ERROR + full traceback on every startup of
such a camera (and the matching set-path guards, with no try at all, would have
raised uncaught). _has_node centralizes the correct handling so every probe
shares it.
"""

import pytest

from pypylon import genicam

from drivers.pyloncamera import PylonCamera


class _AbsentNodeMap:
    """GetNode raises for any name, like a camera that lacks the node."""

    def GetNode(self, name):
        raise genicam.LogicalErrorException('Node not existing')


class _PresentNodeMap:
    def GetNode(self, name):
        return object()


class _CommsFailNodeMap:
    def GetNode(self, name):
        raise genicam.RuntimeException('camera communication failure')


def test_has_node_absent_returns_false():
    """A node the camera does not expose reports False, no exception raised."""
    assert PylonCamera._has_node(_AbsentNodeMap(), 'BslConversionGainMode') is False


def test_has_node_present_returns_true():
    assert PylonCamera._has_node(_PresentNodeMap(), 'BslConversionGainMode') is True


def test_has_node_comms_failure_propagates():
    """A real comms failure is NOT the absent-node case: it must propagate so
    the probe's caller can mark the camera disconnected."""
    with pytest.raises(genicam.RuntimeException):
        PylonCamera._has_node(_CommsFailNodeMap(), 'BslConversionGainMode')
