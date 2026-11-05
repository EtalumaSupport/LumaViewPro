# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Cross-layer shared utilities.

Modules in src/_lumascope/lib/ are dependency-free relative to src/_lumascope/drivers/, src/_lumascope/modules/, and ui/ --
they may be imported from any layer. Use src/_lumascope/lib/ for instrumentation, generic
helpers, and primitives that don't belong to a specific layer.
"""
