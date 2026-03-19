"""Find the best Python 3.11-3.13 installation on Windows.

Called by install_windows.bat. Prints the command to invoke the
best Python found (e.g., "py -3.13" or "C:\\Python313\\python.exe").
Prints "ERROR" if no suitable Python is found.
"""

import shutil
import subprocess
import sys


SUPPORTED = (11, 12, 13)


def _check_python(cmd):
    """Try to run a Python command and return (minor_version, full_cmd) or None."""
    try:
        result = subprocess.run(
            cmd + ['-c', 'import sys; print(sys.version_info.minor)'],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            minor = int(result.stdout.strip())
            if minor in SUPPORTED:
                # Get full version for display
                ver_result = subprocess.run(
                    cmd + ['-c',
                           'import sys; '
                           'print(f"{sys.version_info.major}.'
                           f'{sys.version_info.minor}.'
                           f'{sys.version_info.micro}")'],
                    capture_output=True, text=True, timeout=10,
                )
                ver = ver_result.stdout.strip() if ver_result.returncode == 0 else f'3.{minor}'
                return minor, ver
    except Exception:
        pass
    return None


def main():
    candidates = []

    # Try py launcher with specific versions (best approach on Windows)
    if shutil.which('py'):
        for minor in reversed(SUPPORTED):  # 13, 12, 11 — prefer newest
            result = _check_python(['py', f'-3.{minor}'])
            if result:
                candidates.append((result[0], result[1], f'py -3.{minor}'))

    # Try python on PATH
    for cmd_name in ('python', 'python3'):
        if shutil.which(cmd_name):
            result = _check_python([cmd_name])
            if result:
                # Avoid duplicates (py launcher might find the same one)
                cmd_str = cmd_name
                if not any(c[0] == result[0] for c in candidates):
                    candidates.append((result[0], result[1], cmd_str))

    if not candidates:
        print('ERROR')
        return

    # Show what we found
    print(f'Found {len(candidates)} Python installation(s):', file=sys.stderr)
    for i, (minor, ver, cmd) in enumerate(candidates):
        print(f'  [{i+1}] Python {ver}  ({cmd})', file=sys.stderr)

    # If multiple, let user choose
    if len(candidates) > 1:
        print('', file=sys.stderr)
        print(f'Press Enter for recommended [Python {candidates[0][1]}]',
              file=sys.stderr)
        try:
            choice = input(f'Choose [1-{len(candidates)}] or Enter: ').strip()
        except EOFError:
            choice = ''
        if choice and choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(candidates):
                print(candidates[idx][2])
                return
        # Default: first (newest)
        print(candidates[0][2])
    else:
        print(candidates[0][2])


if __name__ == '__main__':
    main()
