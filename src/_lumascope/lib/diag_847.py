# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""[DIAG-847] NOT FOR MERGE: the bench's two #847 conditions, set from current.json.

``ScopeSession.create`` sets them from ``diag_847_frame_rate_cap`` and
``diag_847_link_limit_mb`` before the scope is built; the Pylon driver reads them.
"""

# [DIAG-847] False: every start skips the frame-rate cap.
FRAME_RATE_CAP = True
# [DIAG-847] A number of MB/s: the connect writes the link limit On at it,
# after the Default user set loads. None: the camera's own.
LINK_LIMIT_MB = None
