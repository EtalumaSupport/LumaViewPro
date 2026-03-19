"""Find the best Python 3.11-3.13 installation on Windows.

Called by install_windows.bat. Prints ONLY the command to invoke
the best Python found (e.g., "py -3.13") to stdout. Nothing else.
Prints "ERROR" if no suitable Python is found.
"""

import shutil
import subprocess
import sys

SUPPORTED = (11, 12, 13)


def _check_python(cmd):
    """Try to run a Python command. Returns minor version or None."""
    try:
        result = subprocess.run(
            cmd + ['-c', 'import sys; print(sys.version_info.minor)'],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            minor = int(result.stdout.strip())
            if minor in SUPPORTED:
                return minor
    except Exception:
        pass
    return None


def main():
    best_cmd = None
    best_minor = 0

    # Try py launcher with specific versions (best approach on Windows)
    if shutil.which('py'):
        for minor in reversed(SUPPORTED):  # 13, 12, 11
            result = _check_python(['py', f'-3.{minor}'])
            if result and result > best_minor:
                best_minor = result
                best_cmd = f'py -3.{minor}'

    # Try python on PATH if py launcher didn't find anything
    if best_cmd is None:
        for cmd_name in ('python', 'python3'):
            if shutil.which(cmd_name):
                result = _check_python([cmd_name])
                if result and result > best_minor:
                    best_minor = result
                    best_cmd = cmd_name

    if best_cmd:
        print(best_cmd)
    else:
        print('ERROR')


if __name__ == '__main__':
    main()
