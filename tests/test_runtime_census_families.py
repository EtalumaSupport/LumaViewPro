# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The support-bundle censuses must see every C-runtime family that ships.

These censuses exist to answer one question: WHICH copy of a runtime won
the loader search, since an app-local DLL shadows System32 for the whole
process. A family that ships but is absent from the census pattern is
invisible in the exact report meant to catch shadowing -- the failure is
silent, and it reads as a clean census rather than an incomplete one.

VCOMP140.DLL is the worked example: it ships app-local, the build
transcript lists it, and both runtime censuses filtered it out.

The filenames below are the real shipped set, read off a build
transcript, rather than invented examples.
"""

from __future__ import annotations

import re

from modules.app_environment import (
    C_RUNTIME_DLL_FAMILIES,
    crt_dll_pattern,
)

# Every C-runtime file a real build reported shipping, at app root and in
# package subfolders.
SHIPPED_CRT_FILENAMES = (
    'MSVCP140_ATOMIC_WAIT.dll',
    'VCOMP140.DLL',
    'VCRUNTIME140.dll',
    'VCRUNTIME140_1.dll',
    'msvcp140.dll',
    'vcruntime140.dll',
    'vcruntime140_1.dll',
    'msvcp140-a4c2229bdc2a2a630acdc095b4d86008.dll',
)

# Present on the machine even when not shipped app-local; the census
# reports them because whether they resolve app-local or from System32
# is the finding.
SYSTEM_CRT_FILENAMES = (
    'MSVCP140.dll',
    'CONCRT140.dll',
    'ucrtbase.dll',
)


class TestOnDiskCensusPattern:
    def test_every_shipped_runtime_is_visible(self):
        pattern = crt_dll_pattern()
        missed = [n for n in SHIPPED_CRT_FILENAMES if not pattern.match(n)]
        assert not missed, (
            f'These ship but the on-disk census cannot see them: {missed}. '
            f'A runtime missing from the census reads as absent from the '
            f'install, which is the opposite of the truth.'
        )

    def test_system_runtimes_are_visible(self):
        pattern = crt_dll_pattern()
        missed = [n for n in SYSTEM_CRT_FILENAMES if not pattern.match(n)]
        assert not missed, f'census cannot see: {missed}'

    def test_pattern_is_version_agnostic(self):
        """A toolset bump must not silently empty the census."""
        pattern = crt_dll_pattern()
        for future in ('msvcp150.dll', 'vcruntime150.dll', 'vcomp150.dll'):
            assert pattern.match(future), (
                f'{future} would vanish from the census after a toolset bump; '
                f'the pattern must match the family, not one version.'
            )

    def test_ucrt_stubs_stay_excluded(self):
        """~40 OS-provided stubs that never shadow -- excluded on purpose.

        Pinned so a later widening is a deliberate edit rather than a
        side effect of adding a family.
        """
        pattern = crt_dll_pattern()
        for stub in ('api-ms-win-crt-runtime-l1-1-0.dll', 'api-ms-win-crt-math-l1-1-0.dll'):
            assert not pattern.match(stub), f'{stub} should not be in the census'

    def test_non_runtime_dlls_are_not_swept_in(self):
        pattern = crt_dll_pattern()
        for other in ('python313.dll', 'ids_peak.dll', 'PylonBase_v11.dll', 'tbb12.dll'):
            assert not pattern.match(other), f'{other} is not a C runtime'


class TestLoadedModuleCensusPattern:
    """The process-resident census shares the family list, so the two
    reports in one bundle cannot disagree about what counts."""

    def test_shares_the_runtime_families(self):
        from modules import app_environment

        interesting = re.compile(
            '|'.join(app_environment._CAMERA_STACK_FAMILIES + C_RUNTIME_DLL_FAMILIES),
            re.IGNORECASE,
        )
        for name in SHIPPED_CRT_FILENAMES + SYSTEM_CRT_FILENAMES:
            assert interesting.search(name), (
                f'{name} is visible to the on-disk census but not the '
                f'loaded-module census; one bundle would contradict itself.'
            )

    def test_camera_stack_still_covered(self):
        from modules import app_environment

        interesting = re.compile(
            '|'.join(app_environment._CAMERA_STACK_FAMILIES + C_RUNTIME_DLL_FAMILIES),
            re.IGNORECASE,
        )
        for name in (
            'ids_peak.dll',
            'tbb12.dll',
            'GenApi_MD_VC141_v3_4.dll',
            'GCBase_MD_VC141_v3_4.dll',
            'PylonBase_v11.dll',
            'NodeMapData_MD_VC141_v3_4.dll',
            'XmlParser_MD_VC141_v3_4.dll',
            'MathParser_MD_VC141_v3_4.dll',
            'log4cpp_MD_VC142_v3_5_Basler_pylon_v1.dll',
            'python313.dll',
        ):
            assert interesting.search(name), f'{name} dropped from the census'
