# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Out-of-process CPU profiling instrument for a running LumaViewPro.

Samples the live process with py-spy (no in-process overhead, immune to the
scheduler-tick quantization that defeats duration timers) and joins the
per-function sample shares with the process's total CPU to report ABSOLUTE
per-function CPU, ranked, with statistical error bars.
"""
