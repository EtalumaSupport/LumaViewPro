# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
# The board's console as its firmware's input() meets it. Runs INSIDE the
# MicroPython child, imported by the launch before the firmware, and
# replaces the unix runtime's input() with the board's where they differ.
#
# On an rp2 board, input() is shared/readline/readline.c. In 1.19 a line
# ends on a carriage return only; a newline matches no branch and is
# dropped; a printable byte is echoed as it is typed; and a control byte on
# an empty line returns at once (Ctrl-C raising KeyboardInterrupt, Ctrl-D
# EOFError, Ctrl-A / B / E an empty line). The unix runtime ends a line on
# a newline, so without this a host that sends 'Y\n' answers a prompt the
# board would still be waiting on. What readline does besides (cursor
# movement, history, completion, escape sequences) is not modelled, and a
# byte that would reach it is refused rather than guessed.
#
# Later runtimes end a line on either byte; no firmware that calls input()
# runs on one, so only 1.19 is modelled and any other runtime keeps its own.

import builtins
import sys

CR = '\r'
NL = '\n'
CTRL_A, CTRL_B, CTRL_C, CTRL_D, CTRL_E = '\x01', '\x02', '\x03', '\x04', '\x05'


def _input(prompt: str = '') -> str:
    sys.stdout.write(prompt)
    line = ''
    while True:
        c = sys.stdin.read(1)
        if not c:
            # The host closed the board's input: there is no board to type on.
            raise EOFError
        if c == CR:
            sys.stdout.write('\r\n')
            return line
        if c == NL:
            continue
        if ' ' <= c <= '~':
            line += c
            sys.stdout.write(c)
            continue
        if c == CTRL_C:
            raise KeyboardInterrupt
        if not line and CTRL_A <= c <= CTRL_E:
            if c == CTRL_D:
                raise EOFError
            return line
        if c in (CTRL_D, CTRL_E):
            # Delete-at-cursor and end-of-line, with the cursor at the end.
            continue
        raise ValueError(f'console: byte {repr(c)} would reach line editing, not modelled')


if sys.implementation.version[:2] == (1, 19):
    builtins.input = _input
