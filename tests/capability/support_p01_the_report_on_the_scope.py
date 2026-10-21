"""Support P01: the support report made on a scope, and what its fan and LED steps say.

The report the Settings panel's support-report button makes
(`TechSupportReport.generate`), made here through the scope the session
brought up, as the button makes it. On the simulator it shows the probe runs;
on the connected scope it is the error-parity plan's bench check (2): on a
healthy board each of the fan and LED steps reads the board, and a board
without the feature says so in its own words, never a bare error.

    python tests/capability/support_p01_the_report_on_the_scope.py             # a simulated LS850
    python tests/capability/support_p01_the_report_on_the_scope.py --hardware  # the connected scope

The report homes the stage (its homing test) and lights each LED for its
self-test, as the button does. The ZIP is kept in the probe's scratch folder
and named in the verdict; the fan and LED files are printed whole.
"""

import sys
import traceback
import zipfile

from harness import HARDWARE, check, figure, hardware_session, make_session, probe_dir, report

# The files the fan and LED steps write, inside the report's hardware_checks/.
_STEP_FILES = ('fan_test.txt', 'led_leakage.txt')


def _make(session):
    from modules.tech_support_report import TechSupportReport

    zip_path = TechSupportReport(scope=session.scope).generate(output_dir=probe_dir('support_p01'))
    figure('report', str(zip_path))
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        for file_name in _STEP_FILES:
            inside = [n for n in names if n.endswith(f'hardware_checks/{file_name}')]
            check(f'the report holds {file_name}', len(inside) == 1, inside)
            if len(inside) != 1:
                continue
            text = z.read(inside[0]).decode('utf-8', errors='replace')
            print(f'--- {file_name} ---\n{text}', flush=True)
            figure(file_name, text)
            # A step that could not read the board writes 'Error:' with no
            # reading; a board without the feature writes its refusal's words.
            check(f'{file_name} reads the board or names why it cannot', 'Error:' not in text, text)


def main():
    if HARDWARE:
        with hardware_session() as (session, _runner):
            _make(session)
        return
    session, _live = make_session('probe_support_report', microscope='LS850')
    try:
        _make(session)
    finally:
        session.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        check('probe completed without an unexpected raise', False)
    sys.exit(report())
