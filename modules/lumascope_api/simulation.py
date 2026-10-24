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
from drivers.camera import Frame
from drivers.simulated_camera import SensorExposure, SimulatedCamera, SimulatedStall
from drivers.simulated_fx2 import SimulatedFX2
from drivers.simulated_ledboard import SimulatedLEDBoard
from drivers.simulated_motorboard import SimulatedMotorBoard
from drivers.simulated_tmcm6110 import SimulatedTmcm6110
from modules.api_surface import api, api_fields
from modules.exceptions import ArgumentRefusedError
from modules.finite_number import WAIT_S, refuse_unless_finite_number
from modules.axes import Axis

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


@api_fields('model')
@dataclasses.dataclass(frozen=True)
class SimulatedModelAnswer:
    """The model a simulated motor board names as its own (``answer_model``).

    A record rather than the bare model, so that naming no model -- None --
    is an answer a caller can ask for.

    Attributes:
        model: The model the board names, or None for none.
    """

    model: str | None


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
                ``'not_injectable_on_board'`` on the fast LED board, which
                has no link to drop a reply on, and on a Classic model's
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
    def delay_reply(self, board: str, delay_s: float) -> None:
        """The board's next reply comes ``delay_s`` seconds late.

        A reply later than the driver waits for is no reply to it: the
        board did what it was told, and the host was not told so.

        Args:
            board: ``'motor'`` or ``'led'``.
            delay_s: How late the reply comes, in seconds, above 0.

        Raises:
            ArgumentRefusedError: ``'needs_simulated_scope'``;
                ``'fault_unknown'`` for a board this scope lacks;
                ``'not_a_number'`` or ``'fault_bound'`` for a delay not
                above 0 or past 30 days, the longest wait every host can
                hold; ``'not_injectable_on_board'`` on every board but
                a firmware board, the only one with a link to delay.
        """
        target = self._board('delay_reply', board)
        self._bound(
            'delay_s', delay_s, 'a delay above 0 s, at most 30 days', 0 < delay_s <= WAIT_S[1]
        )
        if not isinstance(target, EmulatedBoard):
            raise self._not_injectable('delay_reply', target)
        target.delay_next_reply(float(delay_s))
        self._record('delay_reply', target.disarm_reply, board=board, delay_s=delay_s)

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
                fast LED board, which has no link to lose.
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
                for a negative boot or one past 30 days;
                ``'not_injectable_on_board'`` on every
                board but a firmware board, the only one with firmware to
                restart.
        """
        target = self._board('reboot', board)
        self._bound(
            'booting_for_s',
            booting_for_s,
            '0 s to 30 days',
            WAIT_S[0] <= booting_for_s <= WAIT_S[1],
        )
        if not isinstance(target, EmulatedBoard):
            raise self._not_injectable('reboot', target)
        target.reboot(booting_for_s)
        self._record('reboot', _over, board=board, booting_for_s=booting_for_s)

    # --- the stage ---

    @api(in_process=True)
    def fail_home(self, axis: Axis, how: str) -> None:
        """``axis``'s home fails in the simulated hardware, until ``clear``.

        Args:
            axis: The axis whose home fails.
            how: ``'switch_never_trips'`` (its home switch never closes),
                ``'stall'`` (the motor stalls) or ``'absent'`` (no motor).

        Raises:
            ArgumentRefusedError: ``'not_offered'`` for ``axis``;
                ``'needs_simulated_scope'``; ``'fault_unknown'`` for ``how``, or for a
                scope with no motor board; ``'not_injectable_on_board'`` on
                the fast motor board, which runs no firmware to fail a home;
                ``'argument_not_on_board'`` for a ``how`` the TMCM-6110
                cannot show (it has ``'switch_never_trips'``).
        """
        self._refuse_unless_simulated('fail_home')
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
    def hold_axis(self, axis: Axis, at_fraction: float = 0.5) -> threading.Event:
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
            ArgumentRefusedError: ``'not_offered'`` for ``axis``;
                ``'needs_simulated_scope'``; ``'not_a_number'`` or ``'fault_bound'``
                for a fraction outside 0 to 1; ``'fault_unknown'`` for a
                scope with no motor board; ``'not_injectable_on_board'``
                on every board but the fast motor board: the firmware's
                axis reaches its target whatever is injected, and the
                TMCM-6110's arrives too.
        """
        self._refuse_unless_simulated('hold_axis')
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

    @api(in_process=True)
    def answer_model(self, model: str | None) -> None:
        """The motor board names ``model`` as its own, until ``clear``; None names none.

        Its axes stay the ones it was built with, as a board whose firmware
        names a model its motors are not: ``diagnostics.get_motor_info``
        reports the answer, and bring-up reads it to decide the scope's
        model.

        Raises:
            ArgumentRefusedError: ``'needs_simulated_scope'``;
                ``'fault_unknown'`` for a scope with no motor board;
                ``'not_injectable_on_board'`` on every board but the fast
                motor board: the firmware boards and the TMCM-6110 answer
                with the model they were built as.
        """
        target = self._board('answer_model', MOTOR)
        if not isinstance(target, SimulatedMotorBoard):
            raise self._not_injectable('answer_model', target)
        self._record('answer_model', target.report_model(model), model=model)

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
    def refuse_camera_write(self, node: str, answer: str, count: int = 1) -> None:
        """The camera's next ``count`` writes of ``node`` answer ``answer`` and move nothing.

        Args:
            node: The setting written: ``'gain'``, ``'exposure'``,
                ``'auto_exposure'``, ``'frame_size'``, ``'binning'``,
                ``'pixel_format'``, ``'black_level'``, ``'auto_gain'``,
                ``'auto_gain_once'`` or ``'auto_gain_target_brightness'``.
            answer: What the node write answers: ``'false'``, ``'none'``
                (no confirmation), ``'true'`` (a bare True that confirms no
                value; not for ``'frame_size'``, whose write answers the
                window it delivers), ``'raise'``, or ``'removed'`` (False, with the camera
                gone from the bus during the write; it stays removed for the
                session, as after ``remove_camera``), or ``'no_answer'``
                (``CameraNoAnswerError``: the write timed out, and the
                camera stays on the bus).
            count: How many writes in a row meet it, 1 or more: the API
                sends a write the camera did not answer once more, so a
                second unanswered write takes 2.

        Raises:
            ArgumentRefusedError: ``'needs_simulated_scope'``;
                ``'fault_unknown'`` for a node, or an answer that node's
                write does not give;
                ``'fault_bound'`` for a count below 1;
                ``'not_injectable_on_board'`` on the FX2 camera.
        """
        target = self._camera_board('refuse_camera_write')
        if not isinstance(target, SimulatedCamera):
            raise self._not_injectable('refuse_camera_write', target)
        for argument, value, offered in (
            ('node', node, tuple(SimulatedCamera.WRITABLE_NODES)),
            ('answer', answer, SimulatedCamera.answers_for(node)),
        ):
            if value not in offered:
                raise ArgumentRefusedError(
                    'fault_unknown', argument=argument, value=value, offered=offered
                )
        self._bound('count', count, 'a number of writes, 1 or more', count >= 1)
        target.refuse_write(node, answer, count)
        self._record(
            'refuse_camera_write',
            lambda: target.allow_write(node),
            node=node,
            answer=answer,
            count=count,
        )

    @api(in_process=True)
    def refuse_camera_read(self, node: str, answer: str, count: int = 1) -> None:
        """The camera's next ``count`` reads of ``node`` fail as ``answer`` says.

        Args:
            node: The setting read: ``'gain'``, ``'exposure'``,
                ``'frame_size'``, ``'binning'``, ``'pixel_format'``,
                ``'black_level'``, ``'frame_rate'`` or ``'minimum_exposure'``
                (``camera.read_exposure_min_ms``).
            answer: How the read fails: ``'raise'``; ``'sentinel'``, the
                driver's failed-read value (-1 for the gain, the exposure
                and the binning, nothing for the frame size; not for the
                other nodes, whose drivers raise); or ``'removed'``, raising
                with the camera gone from the bus during the read (it stays
                removed for the session, as after ``remove_camera``).
            count: How many reads in a row meet it, 1 or more.

        Raises:
            ArgumentRefusedError: ``'needs_simulated_scope'``;
                ``'fault_unknown'`` for a node, or an answer that node's
                read does not give; ``'fault_bound'`` for a count below 1;
                ``'not_injectable_on_board'`` on the FX2 camera.
        """
        target = self._camera_board('refuse_camera_read')
        if not isinstance(target, SimulatedCamera):
            raise self._not_injectable('refuse_camera_read', target)
        for argument, value, offered in (
            ('node', node, tuple(SimulatedCamera.READABLE_NODES)),
            ('answer', answer, SimulatedCamera.read_answers_for(node)),
        ):
            if value not in offered:
                raise ArgumentRefusedError(
                    'fault_unknown', argument=argument, value=value, offered=offered
                )
        self._bound('count', count, 'a number of reads, 1 or more', count >= 1)
        target.refuse_read(node, answer, count)
        self._record(
            'refuse_camera_read',
            lambda: target.allow_read(node),
            node=node,
            answer=answer,
            count=count,
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

    @api(in_process=True)
    def delay_delivery(self, frames: int) -> None:
        """The camera's frames reach the host ``frames`` behind the camera from
        now on, as through a link or a driver queue that has fallen behind.

        Each frame arrives once ``frames`` more have been made after it, so
        a frame made before a change can arrive well after it, numbered and
        timed at its arrival. The backlog builds over the next ``frames``
        frames, in which none arrives; the first frame to arrive after this
        call says it stands. ``clear`` takes it back and the held frames
        arrive at once.

        Raises:
            ArgumentRefusedError: ``'needs_simulated_scope'``;
                ``'not_a_number'`` or ``'fault_bound'`` for fewer than 1
                frame; ``'not_injectable_on_board'`` on the FX2, whose
                frames come over its own link.
        """
        target = self._camera_board('delay_delivery')
        self._bound('frames', frames, 'a number of frames, 1 or more', frames >= 1)
        if not isinstance(target, SimulatedCamera):
            raise self._not_injectable('delay_delivery', target)
        target.delay_delivery(int(frames))
        self._record('delay_delivery', lambda: target.delay_delivery(0), frames=frames)

    @api(in_process=True)
    def refuse_stream_start(self, count: int = 1) -> None:
        """The camera refuses its next ``count`` stream starts and stays on the bus.

        A start the camera refuses raises its fault to whoever started the
        stream, and the camera is left connected and not streaming.

        Raises:
            ArgumentRefusedError: ``'needs_simulated_scope'``;
                ``'not_a_number'`` or ``'fault_bound'`` for fewer than 1
                start; ``'not_injectable_on_board'`` on the simulated
                camera, made like a Pylon body, whose driver logs a start
                that fails.
        """
        target = self._camera_board('refuse_stream_start')
        self._bound('count', count, 'a number of starts, 1 or more', count >= 1)
        if not isinstance(target, SimulatedFX2):
            raise self._not_injectable('refuse_stream_start', target)
        target.transport.refuse_start(int(count))
        self._record('refuse_stream_start', target.transport.allow_start, count=count)

    @api(in_process=True)
    def bind_camera_usb_driver(self, hardware_id: str, service: str | None) -> None:
        """Windows binds a USB identity of the simulated FX2 to ``service``.

        The simulated FX2 comes up on WinUSB, as Windows binds it with
        LumaViewPro's package installed; this rebinds one identity, as a
        Zadig run or LumaView Classic's package would. The camera already up
        keeps running: the binding is read by
        ``diagnostics.get_camera_usb_bindings`` and checked at the next
        connect.

        Args:
            hardware_id: ``USB\\VID_04B4&PID_8613`` (the bootloader) or
                ``USB\\VID_04B4&PID_EA17`` (the running firmware).
            service: The driver's service, as Windows names it
                (``'libusbK'``); None for no driver.

        Raises:
            ArgumentRefusedError: ``'needs_simulated_scope'``;
                ``'fault_unknown'`` for any other hardware ID;
                ``'not_injectable_on_board'`` on any camera but the FX2.
        """
        from drivers.usb_driver_binding import DEVICE_NAMES

        target = self._camera_board('bind_camera_usb_driver')
        if not isinstance(target, SimulatedFX2):
            raise self._not_injectable('bind_camera_usb_driver', target)
        if hardware_id not in DEVICE_NAMES:
            raise ArgumentRefusedError(
                'fault_unknown',
                argument='hardware_id',
                value=hardware_id,
                offered=tuple(DEVICE_NAMES),
            )
        was = target.transport.services[hardware_id]
        target.transport.bind(hardware_id, service)
        self._record(
            'bind_camera_usb_driver',
            lambda: target.transport.bind(hardware_id, was),
            hardware_id=hardware_id,
            service=service,
        )

    @api(in_process=True)
    def refuse_led_write(self, count: int = 1) -> None:
        """The LED board refuses its next ``count`` writes and stays on the bus.

        An LED command whose write is refused raises the LED board's
        ``BoardPortFailedError`` to its caller, and the LED stays connected
        for the next command. On the FX2 each LED command is three writes.

        Raises:
            ArgumentRefusedError: ``'needs_simulated_scope'``;
                ``'fault_unknown'`` on a scope with no LED board;
                ``'not_a_number'`` or ``'fault_bound'`` for fewer than 1
                write; ``'not_injectable_on_board'`` on any LED board but
                the FX2.
        """
        target = self._board('refuse_led_write', LED)
        self._bound('count', count, 'a number of writes, 1 or more', count >= 1)
        if not isinstance(target, SimulatedFX2):
            raise self._not_injectable('refuse_led_write', target)
        target.transport.refuse_led_write(int(count))
        self._record('refuse_led_write', target.transport.allow_led_write, count=count)

    @api(in_process=True, reads=True)
    def exposed_under(self, frame: Frame) -> SensorExposure:
        """What a frame the simulated camera delivered was exposed under:
        the settings its sensor applied when its exposure started, and Z
        when it ended.

        A frame's chunks name the register when it was tagged, which a
        change reaches before the frames already started show it; this is
        what the pixels show, so a test can judge a frame by what it was
        made under, never by its pixels or by the gate's own answer.

        Raises:
            ArgumentRefusedError: ``'needs_simulated_scope'``;
                ``'not_injectable_on_board'`` on the FX2, which keeps no
                record; ``'exposure_not_recorded'`` for a frame this camera
                did not deliver, or one older than the records it keeps
                (``SimulatedCamera.EXPOSURE_RECORDS_KEPT``).
        """
        target = self._camera_board('exposed_under')
        if not isinstance(target, SimulatedCamera):
            raise self._not_injectable('exposed_under', target)
        frame_id = (frame.chunks or {}).get('FrameID')
        exposure = None if frame_id is None else target.exposed_under(frame_id)
        if exposure is None:
            raise ArgumentRefusedError(
                'exposure_not_recorded',
                argument='frame',
                value=frame.seq,
                kind=f'it kept the last {SimulatedCamera.EXPOSURE_RECORDS_KEPT} it delivered',
            )
        return exposure

    # --- what is injected ---

    @api(in_process=True, reads=True)
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
