# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import csv
import datetime
import pathlib

import pandas as pd

from lvp_logger import logger
from modules.exceptions import RecordIncompleteError


class ProtocolExecutionRecord:
    FILE_HEADER = 'LumaViewPro Protocol Execution Record'
    CURRENT_VERSION = 3
    DEFAULT_FILENAME = 'protocol_record.tsv'
    # 'Scan Count' is the CONTRACT for the temporal ordinal everywhere a
    # post-processing dataframe carries the column: which pass through
    # the protocol produced this capture. Post-processing consumers
    # (stack builder, z-projector, stitcher, video builder) group and
    # sort on it as the T axis. The one deliberate extension: a
    # RECORDING'S OWN frames have no scan axis, so a per-recording
    # dataframe (the manual hyperstack build) carries the frame ordinal
    # in this column -- T = frame there, by the same "temporal ordinal"
    # meaning, never a mixture of the two in one dataframe. The column
    # name is an out-of-band identifier (this TSV is read by external
    # tooling); it does not change.
    COLUMNS = (
        'Filename',
        'Step Name',
        'Step Index',
        'Scan Count',
        'Timestamp',
        'Frame Count',
        'Duration (s)',
    )

    def __init__(
        self,
        protocol_file_loc: pathlib.Path,  # | None = None,
        outfile: pathlib.Path | None = None,
        records: pd.DataFrame | None = None,
    ):
        if (outfile is not None) and (records is not None):
            raise Exception('Specify only outfile OR records')

        if (outfile is None) and (records is None):
            raise Exception('Must specify outfile or records')

        self._protocol_file_loc = pathlib.Path(protocol_file_loc)

        # Every capture the protocol attempts should leave exactly one row
        # (success or failure). Counting attempts against rows actually
        # written lets complete() catch a capture that vanished without a row
        # -- or a row whose own disk write raised -- which is otherwise an
        # invisible hole in the record that post-processing silently skips.
        self._capture_attempts = 0
        self._rows_written = 0
        # Latched (never reset) when the save target is declared dead -- a
        # wedged file writer. Every add_step after that point would block the
        # calling thread against a filesystem that has already stopped
        # responding (bench-measured ~60 s per attempt on a dead SMB share,
        # on the ABORT path of all places), with no chance of landing a row.
        # The latch makes that write a loud no-op instead. complete() is NOT
        # latched: it does no filesystem I/O and its reconcile warning must
        # survive. New runs construct a new record, so the latch scope is
        # exactly one run's dead target.
        self._target_unresponsive = False

        if outfile is not None:
            self._mode = 'to_file'
            self._outfile = outfile
            self._initialize_outfile(outfile=outfile)
        else:
            self._mode = 'from_file'
            self._records = records

    def _initialize_outfile(self, outfile: pathlib.Path):
        """Create file with header. Each add_step will append separately."""
        with open(outfile, 'w', newline='') as fp:
            csv_writer = csv.writer(fp, delimiter='\t', lineterminator='\n')
            csv_writer.writerow([self.FILE_HEADER])
            csv_writer.writerow(['Version', self.CURRENT_VERSION])
            csv_writer.writerow(['Protocol File', str(self._protocol_file_loc)])
            csv_writer.writerow(self.COLUMNS)

    def protocol_file_loc(self) -> pathlib.Path:
        return self._protocol_file_loc

    def mark_target_unresponsive(self) -> None:
        """Declare the record's disk target dead; see the latch comment in
        __init__. Set-once, no lock: the two writer threads at worst let one
        already-started add_step finish -- harmless."""
        self._target_unresponsive = True

    def note_capture_attempt(self) -> None:
        """Record that the protocol is about to attempt one capture.

        Called once per capture the protocol dispatches, before the row is
        written. Paired with the per-row count so complete() can reconcile.
        """
        self._capture_attempts += 1

    def complete(self) -> None:
        """Finalize the record.

        Reconciled on every ending: a run writes every image it captured
        however it ends, so a shortfall is always a fault.

        Raises:
            RecordIncompleteError: fewer rows were written than captures
                attempted. Raised to the run's files completion, whose
                reporter tells the person once.
        """
        self._close_outfile()
        if self._mode == 'to_file':
            missing = self._capture_attempts - self._rows_written
            if missing > 0:
                raise RecordIncompleteError(
                    missing=missing,
                    attempted=self._capture_attempts,
                    record_name=self._outfile.name,
                )

    def _close_outfile(self):
        # Execution record is written in append mode; nothing to close
        pass

    def add_step(
        self,
        capture_result_file_name: pathlib.Path,
        step_name: str,
        step_index: int,
        scan_count: int,
        timestamp: datetime.datetime,
        frame_count: int = 1,
        duration_sec: float = 0.0,
    ):
        if self._mode != 'to_file':
            raise Exception(
                "add_step() can only be called when the instance is initialized with an 'outfile'."
            )

        if self._target_unresponsive:
            logger.warning(
                f'ProtocolExecutionRecord: not writing step {step_name} -- the '
                f'record target was declared unresponsive; the row is lost'
            )
            return

        try:
            with open(self._outfile, 'a', newline='') as fp:
                csv_writer = csv.writer(fp, delimiter='\t', lineterminator='\n')
                csv_writer.writerow(
                    [
                        capture_result_file_name,
                        step_name,
                        step_index,
                        scan_count,
                        timestamp,
                        frame_count,
                        duration_sec,
                    ]
                )
            # Counted only after a successful write: a row whose write raises
            # stays uncounted, so reconciliation in complete() flags the loss.
            self._rows_written += 1
        except Exception as e:
            logger.error(f'ProtocolExecutionRecord: Failed to write step {step_name}: {e}')

    def get_data_from_filename(self, file_path: str | pathlib.Path) -> dict | None:
        record = self._records.loc[self._records['Filename'] == str(file_path)]
        if len(record) != 1:
            return None

        first_row = record.iloc[0]
        return {
            'Step Index': first_row['Step Index'],
            'Scan Count': first_row['Scan Count'],
            'Timestamp': first_row['Timestamp'],
        }

    def num_records(self) -> int:
        return len(self._records)

    @classmethod
    def from_file(cls, file_path: pathlib.Path) -> 'ProtocolExecutionRecord':
        """The run record saved at ``file_path``.

        Raises:
            OSError: the file cannot be opened.
            ValueError: the file is not a run record this release reads: the
                wrong header or version, or a row cut short or malformed.
        """
        try:
            return cls._read(file_path)
        except (StopIteration, IndexError, csv.Error) as e:
            raise ValueError(f'{pathlib.Path(file_path).name} is cut short or malformed') from e

    @classmethod
    def _read(cls, file_path: pathlib.Path) -> 'ProtocolExecutionRecord':
        with open(file_path) as fp:
            csvreader = csv.reader(fp, delimiter='\t')
            header = next(csvreader)
            if header[0] != cls.FILE_HEADER:
                raise ValueError('Invalid protocol execution record')

            version = next(csvreader)
            if version[0] != 'Version':
                raise ValueError('Version key not found')

            if int(version[1]) not in (2, 3):  # Add 3 to supported versions
                raise ValueError('Unsupported protocol execution record version')

            protocol_file_loc_row = next(csvreader)
            if protocol_file_loc_row[0] != 'Protocol File':
                raise ValueError('Protocol file location not found in file')

            protocol_file_loc = protocol_file_loc_row[1]

            _ = next(csvreader)  # Column names

            records = []
            for row in csvreader:
                record_dict = {
                    'Filename': row[0],
                    'Step Name': row[1],
                    'Step Index': int(row[2]),
                    'Scan Count': int(row[3]),
                    'Timestamp': datetime.datetime.strptime(row[4], '%Y-%m-%d %H:%M:%S.%f'),
                }

                # Handle version 3 additions
                if len(row) > 5:
                    record_dict['Frame Count'] = int(row[5])
                    record_dict['Duration (s)'] = float(row[6])
                else:
                    # Default values for older versions
                    record_dict['Frame Count'] = 1
                    record_dict['Duration (s)'] = 0.0

                records.append(record_dict)

            df = pd.DataFrame(records)

            return ProtocolExecutionRecord(records=df, protocol_file_loc=protocol_file_loc)
