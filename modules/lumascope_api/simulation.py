# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""SimulationAPI -- the faults a simulated scope presents, switched on by name.

A real board or camera fails in ways the API reads through its drivers: a
reply that does not come, a home the firmware fails, a cable pulled, a
camera that leaves the bus or refuses a write. A simulated scope presents
the same faults through the members here, so a test, the contract harness
or an integrator developing without hardware drives the production
driver path and asserts what the API answers; nothing reaches a simulated
board privately.

Each member is answered by the simulated board the scope built -- the
capability probe is the board itself -- and refused by name on a board
that cannot produce the fault, never a silent no-op. Which board answers
which member: ``docs/LumascopeSkills.md`` "Simulated mode". The members are
in-process only: the faults belong to the simulator, not to a wire.

A fault is armed for the next command, not the one in flight.
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

from drivers.sim_wire.port import EmulatedBoard
from drivers.simulated_camera import SimulatedCamera, SimulatedStall
from drivers.simulated_fx2 import SimulatedFX2
from drivers.simulated_ledboard import SimulatedLEDBoard
from drivers.simulated_motorboard import SimulatedMotorBoard
from drivers.simulated_tmcm6110 import SimulatedTmcm6110
from modules.api_surface import api, api_fields
from modules.exceptions import ArgumentRefusedError
from modules.finite_number import refuse_unless_finite_number
from modules.lumascope_api._constants import refuse_unknown_axis

if TYPE_CHECKING:
    from modules.lumascope_api._lumascope import Lumascope

MOTOR = 'motor'
LED = 'led'

# How a home fails, as the EL-0940 firmware's simulated hardware takes it.
HOME_FAULTS = ('switch_never_trips', 'stall', 'absent')


@api_fields('member', 'arguments')
@dataclasses.dataclass(frozen=True)
class InjectedFault:
    """One fault injected on a simulated scope, as it was asked for.

    Attributes:
        member: The ``scope.simulation`` member that injected it.
        arguments: The arguments it was called with, by name.
    """

    member: str
    arguments: dict[str, str | int | float | bool | None]


def _family(board: object) -> str:
    """The simulated board ``board`` is, in words, for a refusal."""
    if isinstance(board, SimulatedMotorBoard):
        return 'the fast simulated motor board'
    if isinstance(board, SimulatedLEDBoard):
        return 'the fast simulated LED board'
    if isinstance(board, EmulatedBoard):
        return f'the simulated EL-0940 board {board.label}'
    if isinstance(board, SimulatedTmcm6110):
        return 'the simulated TMCM-6110'
    if isinstance(board, SimulatedFX2):
        return 'the simulated FX2'
    return 'the simulated camera'


class SimulationAPI:
    """Fault injection on a simulated scope; every member refuses on a real one.

    Holds the simulated boards the scope built, which are the drivers
    themselves on the fast tier, the boards behind the production drivers
    on the firmware tiers, and the devices behind them on a Classic model.
    """

    def __init__(
        self,
        scope: Lumascope,
        *,
        motor: object | None = None,
        led: object | None = None,
        camera: SimulatedCamera | SimulatedFX2 | None = None,
    ) -> None:
        self._scope = scope
        self._boards = {
            name: board for name, board in ((MOTOR, motor), (LED, led)) if board is not None
        }
        self._camera = camera
        # What is injected, oldest first, each beside what takes it back.
        self._injected: list[tuple[InjectedFault, Callable[[], None]]] = []

    # --- refusals ---

    def _refuse_unless_simulated(self, member: str) -> None:
        if not self._scope._simulated:
            raise ArgumentRefusedError('needs_simulated_scope', argument=member, value=None)

    def _board(self, member: str, board: str) -> object:
        """The simulated board named ``board``, on a simulated scope.

        Raises:
            ArgumentRefusedError: ``'needs_simulated_scope'`` on a real
                scope; ``'fault_unknown'`` for a name that is not one of
                the boards this scope has.
        """
        self._refuse_unless_simulated(member)
        if board not in self._boards:
            raise ArgumentRefusedError(
                'fault_unknown', argument='board', value=board, offered=tuple(self._boards)
            )
        return self._boards[board]

    def _camera_board(self, member: str) -> SimulatedCamera | SimulatedFX2:
        self._refuse_unless_simulated(member)
        if self._camera is None:
            raise ArgumentRefusedError(
                'not_injectable_on_board',
                argument=member,
                value=None,
                kind='a scope with no camera',
            )
        return self._camera

    @staticmethod
    def _not_injectable(member: str, board: object) -> ArgumentRefusedError:
        return ArgumentRefusedError(
            'not_injectable_on_board', argument=member, value=None, kind=_family(board)
        )

    @staticmethod
    def _bound(argument: str, value: object, words: str, holds: bool) -> None:
        refuse_unless_finite_number(value, argument)
        if not holds:
            raise ArgumentRefusedError('fault_bound', argument=argument, value=value, kind=words)

    def _record(self, member: str, undo: Callable[[], None], **arguments: object) -> None:
        self._injected.append((InjectedFault(member, arguments), undo))

    # --- the boards' replies and links ---

    @api(in_process=True)
    def drop_replies(self, board: str, count: int | None = 1, command: str | None = None) -> None:
        """The board answers nothing to its next ``count`` commands.

        Args:
            board: ``'motor'`` or ``'led'``.
            count: How many replies go missing, 1 or more; None for every
                one until ``clear``, as a register the board stops reporting
                is missing from every poll. Taken by the fast motor board
                only: a firmware board's link drops a counted few.
            command: Only commands with this first word, such as
                ``'ACTUAL_RZ'`` or ``'STOP'``; any command when None. Taken
                by the fast motor board only: a firmware board's link loses
                whichever reply comes next.

        Raises:
            ArgumentRefusedError: ``'needs_simulated_scope'``;
                ``'fault_unknown'`` for a board this scope lacks;
                ``'fault_bound'`` for a count below 1;
                ``'argument_not_on_board'`` for a count of None on a
                firmware board;
                ``'not_injectable_on_board'`` on the fast LED board, whose
                missing reply the API never reads, and on a Classic model's
                boards, which have ``silence``; ``'argument_not_on_board'``
                for a command on a firmware board.
        """
        target = self._board('drop_replies', board)
        if count is not None:
            self._bound('count', count, 'a number of replies, 1 or more', count >= 1)
        if isinstance(target, SimulatedMotorBoard):
            target.drop_replies(command, count)
            self._record(
                'drop_replies',
                lambda: target.stop_dropping(command),
                board=board,
                count=count,
                command=command,
            )
            return
        if isinstance(target, EmulatedBoard):
            if command is not None:
                raise ArgumentRefusedError(
                    'argument_not_on_board',
                    argument='command',
                    value=command,
                    kind=f'{_family(target)}, whose link drops the next reply whatever its command',
                )
            if count is None:
                raise ArgumentRefusedError(
                    'argument_not_on_board',
                    argument='count',
                    value=count,
                    kind=f'{_family(target)}, whose link drops a counted few replies; '
                    'a board that answers nothing is unplug',
                )
            target.drop_next_reply(count)
            self._record(
                'drop_replies', target.disarm_reply, board=board, count=count, command=None
            )
            return
        raise self._not_injectable('drop_replies', target)

    @api(in_process=True)
    def silence(self, board: str) -> None:
        """The board answers no command from now on, until ``unsilence``.

        Raises:
            ArgumentRefusedError: ``'needs_simulated_scope'``;
                ``'fault_unknown'``; ``'not_injectable_on_board'`` on the
                fast LED board and a firmware board, whose silence is
                ``unplug``.
        """
        target = self._board('silence', board)
        if isinstance(target, SimulatedMotorBoard):
            target.drop_replies(None, None)
            undo = lambda: target.stop_dropping(None)
        elif isinstance(target, SimulatedTmcm6110):
            target.silent = True
            undo = lambda: setattr(target, 'silent', False)
        elif isinstance(target, SimulatedFX2):
            target.transport.go_silent()
            undo = _does_not_resume
        else:
            raise self._not_injectable('silence', target)
        self._record('silence', undo, board=board)

    @api(in_process=True)
    def unsilence(self, board: str) -> None:
        """The board answers again after ``silence``.

        Raises:
            ArgumentRefusedError: ``'needs_simulated_scope'``;
                ``'fault_unknown'``; ``'not_injectable_on_board'`` where
                ``silence`` is, and on the FX2, whose stream does not
                resume.
        """
        target = self._board('unsilence', board)
        if isinstance(target, SimulatedMotorBoard):
            target.stop_dropping(None)
        elif isinstance(target, SimulatedTmcm6110):
            target.silent = False
        else:
            raise self._not_injectable('unsilence', target)

    @api(in_process=True)
    def unplug(self, board: str) -> None:
        """The board's cable is pulled: its port fails from the next command.

        Raises:
            ArgumentRefusedError: ``'needs_simulated_scope'``;
                ``'fault_unknown'``; ``'not_injectable_on_board'`` on the
                fast LED board, whose lost link the API never reads.
        """
        target = self._board('unplug', board)
        if isinstance(target, (SimulatedMotorBoard, EmulatedBoard)):
            target.unplug()
            undo = target.replug
        elif isinstance(target, SimulatedTmcm6110):
            target.plugged = False
            undo = lambda: setattr(target, 'plugged', True)
        elif isinstance(target, SimulatedFX2):
            target.transport.unplug()
            undo = _does_not_resume
        else:
            raise self._not_injectable('unplug', target)
        self._record('unplug', undo, board=board)

    @api(in_process=True)
    def replug(self, board: str) -> None:
        """The board's cable goes back in: it is found again.

        Raises:
            ArgumentRefusedError: ``'needs_simulated_scope'``;
                ``'fault_unknown'``; ``'not_injectable_on_board'`` where
                ``unplug`` is, and on the FX2, which re-enumerates on its
                own.
        """
        target = self._board('replug', board)
        if isinstance(target, (SimulatedMotorBoard, EmulatedBoard)):
            target.replug()
        elif isinstance(target, SimulatedTmcm6110):
            target.plugged = True
        else:
            raise self._not_injectable('replug', target)

    @api(in_process=True)
    def reboot(self, board: str, booting_for_s: float = 0.0) -> None:
        """The board's firmware restarts and forgets everything it knew.

        Its port fails at once, as USB re-enumerates, and the board is found
        again after ``booting_for_s`` seconds.

        Raises:
            ArgumentRefusedError: ``'needs_simulated_scope'``;
                ``'fault_unknown'``; ``'not_a_number'`` or ``'fault_bound'``
                for a negative boot; ``'not_injectable_on_board'`` on every
                board but a firmware board, the only one with firmware to
                restart.
        """
        target = self._board('reboot', board)
        self._bound('booting_for_s', booting_for_s, '0 s or more', booting_for_s >= 0)
        if not isinstance(target, EmulatedBoard):
            raise self._not_injectable('reboot', target)
        target.reboot(booting_for_s)
        self._record('reboot', _over, board=board, booting_for_s=booting_for_s)

    # --- the stage ---

    @api(in_process=True)
    def fail_home(self, axis: str, how: str) -> None:
        """``axis``'s home fails in the simulated hardware, until ``clear``.

        Args:
            axis: The axis whose home fails.
            how: ``'switch_never_trips'`` (its home switch never closes),
                ``'stall'`` (the motor stalls) or ``'absent'`` (no motor).

        Raises:
            ArgumentRefusedError: ``'needs_simulated_scope'``;
                ``'axis_unknown'``; ``'fault_unknown'`` for ``how``, or for a
                scope with no motor board; ``'not_injectable_on_board'`` on
                the fast motor board, which runs no firmware to fail a home;
                ``'argument_not_on_board'`` for a ``how`` the TMCM-6110
                cannot show (it has ``'switch_never_trips'``).
        """
        self._refuse_unless_simulated('fail_home')
        refuse_unknown_axis(axis)
        if how not in HOME_FAULTS:
            raise ArgumentRefusedError(
                'fault_unknown', argument='how', value=how, offered=HOME_FAULTS
            )
        target = self._board('fail_home', MOTOR)
        if isinstance(target, EmulatedBoard):
            target.inject(axis, how)
            undo = lambda: target.clear(axis, how)
        elif isinstance(target, SimulatedTmcm6110):
            if how != 'switch_never_trips':
                raise ArgumentRefusedError(
                    'argument_not_on_board',
                    argument='how',
                    value=how,
                    kind=f'{_family(target)}, whose home fails by its switch',
                )
            target.break_right_switch(axis)
            undo = lambda: target.break_right_switch(axis, broken=False)
        else:
            raise self._not_injectable('fail_home', target)
        self._record('fail_home', undo, axis=axis, how=how)

    @api(in_process=True)
    def hold_axis(self, axis: str, at_fraction: float = 0.5) -> threading.Event:
        """``axis``'s next move halts part of the way and never arrives, until ``clear``.

        A stop, a home or a new target on the axis ends the move it holds,
        and with it the hold, as on a real stage.

        Args:
            axis: The axis whose next move is held.
            at_fraction: How far along its travel it halts, between 0 and 1.

        Returns:
            An event set once the move has halted at the hold, for a caller
            acting on a move while it travels: waited on, it replaces a
            guess with a clock at how far the move has got. ``clear``
            releases the move, which then travels the rest.

        Raises:
            ArgumentRefusedError: ``'needs_simulated_scope'``;
                ``'axis_unknown'``; ``'not_a_number'`` or ``'fault_bound'``
                for a fraction outside 0 to 1; ``'fault_unknown'`` for a
                scope with no motor board; ``'not_injectable_on_board'``
                on every board but the fast motor board: the firmware's
                axis reaches its target whatever is injected, and the
                TMCM-6110's arrives too.
        """
        self._refuse_unless_simulated('hold_axis')
        refuse_unknown_axis(axis)
        self._bound('at_fraction', at_fraction, 'between 0 and 1', 0 < at_fraction < 1)
        target = self._board('hold_axis', MOTOR)
        if not isinstance(target, SimulatedMotorBoard):
            raise self._not_injectable('hold_axis', target)
        hold = target.hold_travel(axis, at_fraction)
        self._record('hold_axis', hold.release, axis=axis, at_fraction=at_fraction)
        return hold.reached

    @api(in_process=True)
    def interlock(self, lid_open: bool | None = None, powered: bool | None = None) -> None:
        """The stage's interlock inputs read as given; None leaves one as it is.

        Raises:
            ArgumentRefusedError: ``'needs_simulated_scope'``;
                ``'fault_unknown'`` for a scope with no motor board;
                ``'not_injectable_on_board'`` on every board but the
                TMCM-6110, the only one with interlock inputs.
        """
        target = self._board('interlock', MOTOR)
        if not isinstance(target, SimulatedTmcm6110):
            raise self._not_injectable('interlock', target)
        if lid_open is not None:
            target.lid_open = lid_open
        if powered is not None:
            target.powered = powered

        def undo() -> None:
            target.lid_open = False
            target.powered = True

        self._record('interlock', undo, lid_open=lid_open, powered=powered)

    # --- the camera ---

    @api(in_process=True)
    def remove_camera(self) -> None:
        """The camera leaves the bus, as a real one does when removed.

        It stays removed for the session: ``clear`` does not bring it back,
        as nothing brings back a removed real camera but a new session.

        Raises:
            ArgumentRefusedError: ``'needs_simulated_scope'``;
                ``'not_injectable_on_board'`` on a scope with no camera.
        """
        target = self._camera_board('remove_camera')
        if isinstance(target, SimulatedFX2):
            target.transport.unplug()
        else:
            target.leave_bus()
        self._record('remove_camera', _does_not_resume)

    @api(in_process=True)
    def refuse_camera_write(self, node: str, answer: str) -> None:
        """The camera's next write of ``node`` answers ``answer`` and moves nothing.

        Args:
            node: The setting written: ``'gain'``, ``'exposure'``,
                ``'auto_exposure'``, ``'frame_size'``, ``'binning'``,
                ``'pixel_format'`` or ``'black_level'``.
            answer: What the node write answers: ``'false'``, ``'none'``
                (no confirmation), ``'true'`` (a bare True that confirms no
                value), ``'raise'``, or ``'removed'`` (False, with the camera
                gone from the bus during the write; it stays removed for the
                session, as after ``remove_camera``).

        Raises:
            ArgumentRefusedError: ``'needs_simulated_scope'``;
                ``'fault_unknown'`` for a node or an answer;
                ``'not_injectable_on_board'`` on the FX2 camera.
        """
        target = self._camera_board('refuse_camera_write')
        if not isinstance(target, SimulatedCamera):
            raise self._not_injectable('refuse_camera_write', target)
        for argument, value, offered in (
            ('node', node, tuple(SimulatedCamera.WRITABLE_NODES)),
            ('answer', answer, SimulatedCamera.REFUSED_WRITE_ANSWERS),
        ):
            if value not in offered:
                raise ArgumentRefusedError(
                    'fault_unknown', argument=argument, value=value, offered=offered
                )
        target.refuse_write(node, answer)
        self._record(
            'refuse_camera_write', lambda: target.allow_write(node), node=node, answer=answer
        )

    @api(in_process=True)
    def stall_stream(self, after_s: float, for_s: float) -> None:
        """The camera's stream delivers no frame for ``for_s`` seconds,
        starting ``after_s`` seconds from now; it stays connected and grabbing.

        Raises:
            ArgumentRefusedError: ``'needs_simulated_scope'``;
                ``'not_a_number'`` or ``'fault_bound'`` for a negative start
                or a stall of no length; ``'not_injectable_on_board'`` on
                the FX2, whose stream does not resume (``silence('led')``
                stops it).
        """
        target = self._camera_board('stall_stream')
        self._bound('after_s', after_s, '0 s or more', after_s >= 0)
        self._bound('for_s', for_s, 'more than 0 s', for_s > 0)
        if not isinstance(target, SimulatedCamera):
            raise self._not_injectable('stall_stream', target)
        target.hold_frames(SimulatedStall(after_s, for_s))
        self._record('stall_stream', target.release_frames, after_s=after_s, for_s=for_s)

    # --- what is injected ---

    @api(in_process=True)
    def injected(self) -> tuple[InjectedFault, ...]:
        """The faults injected since the last ``clear``, oldest first, as asked.

        A one-shot fault stays listed after it has taken its command.

        Raises:
            ArgumentRefusedError: ``'needs_simulated_scope'``.
        """
        self._refuse_unless_simulated('injected')
        return tuple(fault for fault, _ in self._injected)

    @api(in_process=True)
    def clear(self) -> None:
        """Every injected fault taken back, newest first: drops, holds,
        refused writes and stalls lifted, boards plugged back in and
        answering, faults out of the hardware. A board that found its cable
        pulled is plugged in but stays disconnected until it is reconnected,
        as a real one does. A removed camera, an FX2 unplugged or
        silenced, and a reboot under way are not undone, as on a real scope.

        Raises:
            ArgumentRefusedError: ``'needs_simulated_scope'``.
        """
        self._refuse_unless_simulated('clear')
        injected, self._injected = self._injected, []
        for _, undo in reversed(injected):
            undo()


def _over() -> None:
    """A fault that is over once injected: a reboot."""


def _does_not_resume() -> None:
    """A fault the simulated device does not come back from in a session."""
