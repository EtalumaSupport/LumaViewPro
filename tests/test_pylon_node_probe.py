# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression: the Pylon lookup answers a missing node as None, quietly.

pypylon's GetNode answers a placeholder for a node that is not in the map --
it does not return None -- and the map's Contains says whether it holds the
node. Probing an optional Basler feature
(BslConversionGainMode, BslLineNoiseReduction*) on a camera that lacks it --
a Basler dart has no Bsl* nodes -- is the EXPECTED "unsupported" case and must
report absence without logging an error or letting the exception escape.

Before the fix, the capability probes did ``GetNode(name) is not None``, which
assumed None-on-missing; the raise instead fell through to a broad
``except Exception`` that logged an ERROR + full traceback on every startup of
such a camera (and the matching set-path guards, with no try at all, would have
raised uncaught). ``PylonCamera._node_in`` is the one place absence is decided,
so every probe shares it.
"""

import pytest

from pypylon import genicam

from _lumascope.drivers.pyloncamera import PylonCamera
from tests.pypylon_stub import NodeMap


class _CommsFailNodeMap(NodeMap):
    """A map holding the node, whose access the camera does not answer."""

    def GetNode(self, name):
        raise genicam.RuntimeException('camera communication failure')


def test_an_absent_node_is_none():
    """A node the camera does not expose is None, no exception raised."""
    assert PylonCamera._node_in(NodeMap(), 'BslConversionGainMode') is None


def test_a_present_node_is_the_node():
    node = object()
    nodemap = NodeMap({'BslConversionGainMode': node})
    assert PylonCamera._node_in(nodemap, 'BslConversionGainMode') is node


def test_a_comms_failure_propagates():
    """A real comms failure is NOT the absent-node case: it must propagate so
    the probe's caller can answer it as its own failure."""
    with pytest.raises(genicam.RuntimeException):
        PylonCamera._node_in(
            _CommsFailNodeMap({'BslConversionGainMode': object()}), 'BslConversionGainMode'
        )
