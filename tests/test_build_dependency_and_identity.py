"""Build-chain guards: dependency discovery and build identity.

Two defects these pin, both found on the 2026-08-17 Windows bench trip:

1. IDS ships a download PACKAGE (``ids-peak-win-<variant>-setup-...exe``,
   hyphens) wrapping the real installer (``ids_peak_<ver>.exe``,
   underscores), and only the inner one accepts the silent-install
   switches. Dropping the wrapper into ``dependencies\\`` produced a
   build that reported success and an installer whose IDS package exited
   ``0x80042000``; dropping nothing produced a build that also reported
   success and an app failing every IDS connect with
   ``GENICAM_GENTLN_PATH environment variable not found``. Both silent.

2. ``version.txt`` line 4 is written per COMMIT by the pre-commit hook but
   was labelled ``BuildGUID``. Three builds of one SHA produced
   byte-identical banners, so a rebuild that changed only bundled inputs
   could not be told apart from its predecessor.

The selector tests read the pattern OUT of build.ps1 rather than
restating it, so a drift in the shipped literal reds these rather than
passing against a copy that agrees with itself.

Engine note. PowerShell's regex engine and Python's ``re`` are different
implementations, so matching the extracted literal here is not automatically
proof about the build host. That was checked rather than assumed: on
2026-08-17 both patterns were extracted and run through ``pwsh`` against all
eleven fixtures below -- selector, wrapper detector, and the junk that must
match neither -- and both engines agreed on every case.

What remains true: that was a one-time desk check, and CI does not re-run
it (no PowerShell in the test environment). If the pattern is ever changed
to use a construct where the engines diverge -- named groups, lookbehind,
Unicode classes -- re-run the pwsh check by hand. The current pattern uses
only anchors, a character class and ``.*``, which are identical in both.
"""

import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILD_PS1 = REPO_ROOT / 'scripts' / 'appBuild' / 'build.ps1'
MIN_VERSION_FILE = REPO_ROOT / 'scripts' / 'appBuild' / 'MIN_BUILD_SCRIPT_VERSION'
LVP_LOGGER = REPO_ROOT / 'lvp_logger.py'


def _build_ps1_text() -> str:
    # pin-justified: build.ps1 is PowerShell. There is no Python AST seam
    # to assert against, and no PowerShell parser in the test environment.
    assert BUILD_PS1.exists(), f'build script missing at {BUILD_PS1}'
    return BUILD_PS1.read_text(encoding='utf-8-sig')


def _lvp_logger_text() -> str:
    # pin-justified: conftest replaces lvp_logger in sys.modules with a
    # MagicMock, so the banner cannot be called and its behaviour cannot be
    # observed. See TestBannerIdentityContract for why importing the real
    # module under an alias was rejected. One read site serves every
    # assertion in that class rather than one per test.
    return LVP_LOGGER.read_text()


def _ids_selector() -> re.Pattern:
    """The IDS installer selector, extracted from the shipped build script."""
    text = _build_ps1_text()
    m = re.search(r"\$IDS_EXE_PATTERN\s*=\s*'([^']+)'", text)
    assert m, (
        'build.ps1 defines no $IDS_EXE_PATTERN. The IDS installer selector must '
        'be a named, extractable literal so this test pins what actually ships.'
    )
    return re.compile(m.group(1))


# IDS ships two executables. Only the INNER one (underscores) accepts the
# silent-install switches; the outer download package (hyphens) does not.
# Per the IDS peak 26.06 readme: "The required silent setup command line
# parameters can only be processed by the ids_peak_<version>.exe."
VENDOR_IDS_INSTALLERS = [
    'ids_peak_2.21.0.0-630_full_64.exe',
    'ids_peak_2.18.1.0-183_full_64.exe',
    'ids_peak_2.9.0.0.exe',
    'ids_peak_64-26.06.1.exe',
]

# The wrapper is the specific artifact that shipped a bundle whose IDS
# package exited 0x80042000 twice on 2026-08-17. It must NOT be selected.
IDS_DOWNLOAD_WRAPPERS = [
    'ids-peak-win-runtime-step-64.26.06.1.exe',
    'ids-peak-win-standard-setup-64-2.21.0.0.exe',
]

NOT_IDS_INSTALLERS = [
    'vc_redist.x64.exe',
    'pylon_USB_Camera_Driver.msi',
    'setup.iss',
    'ids_peak_2.9.0.0.exe.bak',
    'ids_peak_notes.txt',
]


@pytest.mark.parametrize('filename', VENDOR_IDS_INSTALLERS)
def test_ids_selector_accepts_real_vendor_filenames(filename):
    """The selector must match the INNER installer -- the driveable one."""
    assert _ids_selector().match(filename), (
        f'{filename!r} is a real IDS Peak installer spelling and the selector '
        f'rejects it. A rejected installer is not bundled, and the build still '
        f'succeeds -- which is exactly the 2026-08-17 failure.'
    )


@pytest.mark.parametrize('filename', NOT_IDS_INSTALLERS)
def test_ids_selector_rejects_non_installers(filename):
    """Breadth must not become a wildcard: .bak / .txt / other vendors stay out."""
    assert not _ids_selector().match(filename), (
        f'{filename!r} is not an IDS Peak installer but the selector accepts it. '
        f'A parking copy or unrelated file would be chained into the installer.'
    )


@pytest.mark.parametrize('filename', IDS_DOWNLOAD_WRAPPERS)
def test_ids_selector_rejects_the_download_wrapper(filename):
    """The wrapper must never be selected -- it cannot be driven silently.

    Widening the selector to accept hyphens looks like the obvious fix for
    "my file was not found" and is the opposite of correct: it bundles an
    ExePackage that rejects /s /f1 at install time, and because the IDS
    package is vital, that rolls the whole product install back.
    """
    assert not _ids_selector().match(filename), (
        f'{filename!r} is the IDS download package, not the installer. '
        f'Selecting it bundles an exe that cannot be driven silently.'
    )


def test_near_miss_guard_names_the_wrapper_case():
    """Rejecting the wrapper silently would just relocate the original bug."""
    text = _build_ps1_text()
    assert '$IDS_WRAPPER_PATTERN' in text, (
        'build.ps1 has no wrapper pattern, so a rejected wrapper is reported '
        'as "no IDS runtime found" -- the message that cost a bench trip'
    )
    guard = text.split('$near_misses = @()', 1)[1]
    assert 'extract' in guard.lower(), (
        'the wrapper near-miss does not tell the builder to extract the '
        'package, which is the one thing they need to do'
    )


def test_near_miss_guard_exists_and_can_fail_the_build():
    """A file meant to be bundled that was not must stop the build.

    Absence of an optional dependency is legal -- most builds ship without
    IDS on purpose. What must be unconstructible is the silent miss: a
    file placed in dependencies\\ that the selector skipped, leaving an
    installer with no driver and a build log that reads as success.
    """
    text = _build_ps1_text()
    assert '$near_misses' in text, (
        'build.ps1 has no near-miss guard. Without it a selector mismatch is '
        'indistinguishable from a deliberate omission.'
    )
    guard = text.split('$near_misses = @()', 1)[1]
    assert 'Exit 1' in guard, (
        'the near-miss guard does not fail the build; a warning would repeat the '
        'original failure, which was a message nobody saw in a 620-line log.'
    )
    # The guard must inspect the directory unconditionally. Running it only
    # inside the "a match was already found" branch is the drafting error
    # this assertion exists to catch: that is precisely when it is not needed.
    ids_block = guard.split('if (-not $ids_files)', 1)
    assert len(ids_block) == 2, (
        'the IDS near-miss check must run when NO strict match was found. '
        'Gating it behind a successful match makes it dead code.'
    )


def test_build_script_version_gate_bumped_for_the_build_id_file():
    """A branch whose logger reads build_id.txt must refuse a builder that never writes it.

    The gate already exists; this pins that it was actually used. A v2
    build.ps1 rewrites version.txt with ``Set-Content -Encoding UTF8``,
    which under Windows PowerShell 5.1 means UTF-8 WITH BOM -- that is
    what shipped in 4.0.0-beta29 and made the version string
    ``LumaViewPro <BOM>4.0.0-beta29``: garbled title bar, a fresh
    Documents folder that orphaned the user's settings, and a TIFF
    Software tag that failed ``TIFF strings must be 7-bit ASCII`` on
    every single save. A v2 copy must not be allowed to build this
    branch.
    """
    text = _build_ps1_text()
    m = re.search(r'\$script_version\s*=\s*(\d+)', text)
    assert m, 'build.ps1 declares no $script_version'
    script_version = int(m.group(1))
    min_version = int(MIN_VERSION_FILE.read_text().strip())
    assert script_version >= 3, (
        'build.ps1 no longer rewrites version.txt, which is a load-bearing '
        'change; $script_version must be bumped so older copies are refused.'
    )
    assert min_version >= 3, (
        f'MIN_BUILD_SCRIPT_VERSION is {min_version}; this branch requires a '
        f'builder that leaves version.txt alone and writes build_id.txt, so '
        f'it must be >= 3.'
    )
    assert script_version >= min_version, (
        f'build.ps1 is v{script_version} but the branch requires '
        f'v{min_version} -- this build script cannot build its own branch.'
    )


def test_build_ps1_writes_the_build_id_to_its_own_file():
    """The build must write build_id.txt; nothing else in the chain does."""
    text = _build_ps1_text()
    assert '$build_id' in text, 'build.ps1 computes no build ID'
    assert 'build_id.txt' in text, (
        'build.ps1 never writes build_id.txt, so the build ID cannot reach the exe'
    )


def test_build_ps1_never_writes_version_txt():
    """version.txt has ONE author: the pre-commit hook.

    This is the 4.0.0-beta29 regression, pinned at its structural root.
    The build ID is a DIAGNOSTIC; version.txt line 1 is LOAD-BEARING --
    it names the user's Documents folder and is stamped into the TIFF
    Software tag of every saved image. Writing the diagnostic into the
    load-bearing file is what let a PowerShell encoding default (5.1
    ``-Encoding UTF8`` emits a BOM) take out image capture entirely.

    Reads are fine; the assertion is that no WRITE targets version.txt.
    """
    text = _build_ps1_text()
    writers = [
        line.strip()
        for line in text.splitlines()
        if not line.lstrip().startswith('#')  # prose may discuss the old writer
        and 'version.txt' in line
        and re.search(r'Set-Content|Out-File|WriteAllLines|WriteAllText|>\s*"', line)
    ]
    assert not writers, (
        'build.ps1 writes version.txt: '
        + ' | '.join(writers)
        + ' -- the build must not author the file that names the data folder '
        'and the TIFF Software tag. Write build_id.txt instead.'
    )


class TestBannerIdentityContract:
    """The banner must separate commit identity from build identity.

    These assert on SOURCE TEXT, which the source-pin ratchet rightly
    discourages -- and it was tried the better way first. The banner
    cannot be exercised: ``tests/conftest.py`` replaces ``lvp_logger`` in
    ``sys.modules`` with a MagicMock, so ``log_environment_banner`` is a
    no-op under pytest and every behavioural assertion passed vacuously
    against an empty capture.

    Importing the real module under an alias to get around that was
    rejected: ``lvp_logger`` installs a global ``sys.excepthook`` at
    import, and a bench log has already been polluted once by an
    out-of-app script inheriting that hook. Trading a suite-wide hazard
    for four tests is a bad deal.

    The sibling ``test_lvp_logger_marker_lookup.py`` pins this module the
    same way for the same reason.
    """

    def test_line_four_is_reported_as_a_commit_identity(self):
        """Line 4 is a commit fingerprint and must not be labelled a build one.

        The pre-commit hook that writes it says so itself: "random per
        commit". Labelling it BuildGUID is what let three builds of one
        SHA read as the same artifact on 2026-08-17.
        """
        emitted = re.findall(r'\[LVP Main\s*\]\s*(BuildGUID|CommitGUID):', _lvp_logger_text())
        assert 'CommitGUID' in emitted, 'the banner does not emit a CommitGUID line'
        assert 'BuildGUID' not in emitted, 'the banner still calls a per-COMMIT value a build ID'

    def test_build_id_is_reported_separately_from_the_commit_guid(self):
        """Two builds of one commit must be tellable apart."""
        text = _lvp_logger_text()
        assert re.search(r'\[LVP Main\s*\]\s*BuildID:', text), (
            'the banner emits no BuildID line, so two builds of one commit stay '
            'indistinguishable -- the defect this line exists to fix'
        )
        assert 'build_id.txt' in text, (
            'build_id.txt is never read, so a stamped build ID is dropped on the floor'
        )

    def test_absent_build_id_distinguishes_installed_exe_from_source_run(self):
        """Absent line 5 means two different things and must not share a message.

        On a source run there was no build, which is honest. On an
        installed exe it means the builder predates the stamp -- saying
        "source / dev" there would be a lying log line introduced by the
        very fix meant to stop misattribution.
        """
        text = _lvp_logger_text()
        _, _, tail = text.partition('_build_id_str')
        assert tail, 'lvp_logger derives no build-ID display string'
        branch = tail[:600]
        assert 'lvp_installed' in branch, (
            'the absent-build-ID path does not branch on lvp_installed, so an '
            'installed exe and a source run would report the same thing'
        )
        assert 'build script' in branch, 'the stale-builder case is never named'
        assert 'source / dev' in branch, 'the source-run case is never named'
