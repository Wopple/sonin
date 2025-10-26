from itertools import groupby
from typing import Any

from pydantic import BaseModel, Field

from sonin.model.hypercube import Hypercube, Vector
from sonin.model.signal import Signal, SignalCount, SignalProfile
from sonin.sonin_math import div

# cells start as receiving
RECEIVING = 0

# cells become sending after an iteration of receiving signals
SENDING = 1

# cells become stable when all adjacent cells are SENDING or STABLE
STABLE = 2


class StemCell(BaseModel):
    position: Vector
    state: int
    signals: dict[Signal, SignalCount] = Field(default_factory=dict)

    def add_signals(self, signal: Signal, count: SignalCount):
        new_count = self.signals.get(signal, 0) + count

        if new_count <= 0:
            if signal in self.signals:
                del self.signals[signal]
        else:
            self.signals[signal] = new_count


class Incubator(BaseModel):
    n_dimension: int
    dimension_size: int

    # a static set of signals in the environment, this seeds variation to avoid symmetry
    environment: list[tuple[Signal, SignalCount, Vector]]

    # defines the affinity between signals
    signal_profile: SignalProfile

    cells: Hypercube[StemCell] = None

    # the hyper-edits of adjacent cells
    adjacent_edits: list[list[int]] = None

    def model_post_init(self, context: Any, /):
        assert self.cells is None
        assert self.adjacent_edits is None

        self.cells = Hypercube(
            n_dimension=self.n_dimension,
            dimension_size=self.dimension_size,
        )

        self.adjacent_edits = []
        base = [0 for _ in range(self.n_dimension)]

        for idx in range(self.n_dimension):
            for e in (-1, 1):
                copy = base.copy()
                copy[idx] = e
                self.adjacent_edits.append(copy)

    def initialize(self, starting_signals: dict[Signal, SignalCount]):
        self.cells.initialize(lambda position: StemCell(position=position, state=RECEIVING))

        for cell in self.cells.center():
            cell.signals.update(starting_signals)
            cell.state = SENDING

    def state_of(self, position: Vector) -> int:
        return self.cells.get(position).state

    # consider handling each signal separately
    def incubate(self):
        """
        Signals push and pull on each other to move signals from SENDING cells to RECEIVING cells.
        After receiving signals, a cell transitions to SENDING (cell division). This happens even if the signal forces
            propagated zero signals to the receiving cell.
        When no more neighboring cells are RECEIVING, a cell transitions to STABLE (prunes them from scanning).
        Adjacent is defined as a city block distance of 1.
        Each iteration, the only receiving cells that receive signals are the ones tied for the greatest potential
            of number of signals received (sum of signals in neighboring sending cells). Sending cells then only send
            signals to that set of cells, not all of their receiving neighbors.
        Iteration stops when there are no more receiving cells.
        """

        sending_cells: list[StemCell] = self.cells.center()

        while True:
            # list of receiving positions and how much they can receive from a neighboring sending cell
            receiving_positions: list[tuple[Vector, SignalCount]] = []

            for send in sending_cells:
                signal_sum: SignalCount = sum(send.signals.values())

                for edit in self.adjacent_edits:
                    receive_position = send.position + edit

                    if not receive_position.out_of_bounds() and self.state_of(receive_position) == RECEIVING:
                        receiving_positions.append((receive_position, signal_sum))

            # no more cells can receive signals
            if not receiving_positions:
                break

            receiving_positions.sort(key=lambda t: t[0])

            # group by receiving cell and sum potential contributions from sending cells
            receiving_position_sums: list[tuple[Vector, SignalCount]] = [
                (position, sum(p[1] for p in pairs))
                for position, pairs in groupby(receiving_positions, key=lambda t: t[0])
            ]

            highest_sum: SignalCount = max(signal_sum for _, signal_sum in receiving_position_sums)

            # these cells are the ones that will receive signals this iteration
            highest_receiving_positions: list[Vector] = [
                position
                for position, signal_sum in receiving_position_sums
                if signal_sum == highest_sum
            ]

            # build list of sending cells and the receiving cells they need to transfer to
            open_channels: list[tuple[Vector, Vector]] = []

            for receive in highest_receiving_positions:
                for edit in self.adjacent_edits:
                    send_position = receive + edit

                    if not send_position.out_of_bounds() and self.state_of(send_position) == SENDING:
                        open_channels.append((send_position, receive))

            open_channels.sort(key=lambda t: t[0])

            sending_cell_channels: list[tuple[StemCell, list[Vector]]] = [
                (self.cells.get(send_position), [p[1] for p in pairs])
                for send_position, pairs in groupby(open_channels, key=lambda t: t[0])
            ]

            all_signals: list[tuple[Signal, SignalCount, Vector]] = self.environment + [
                (signal, count, cell.position)
                for cell in self.cells
                for signal, count in cell.signals.items()
            ]

            # from, to, signal, count
            transfers: list[tuple[StemCell, Vector, Signal, SignalCount]] = []

            # redistribute signals based on signal interactions
            for send, receives in sending_cell_channels:
                # get adjacent receiving position if it exists in the direction indicated by `positive`
                def receive_by_dimension(dimension: int, positive: bool) -> Vector | None:
                    edit: list[int] = [0 for _ in range(self.n_dimension)]
                    edit[dimension] = 1 if positive else -1
                    adjacent_position = send.position + edit

                    if adjacent_position in receives:
                        return adjacent_position
                    else:
                        return None

                for send_signal, send_count in send.signals.items():
                    # get the forces on the sending cell's signals
                    forces: list[Vector] = [
                        self.signal_profile.attraction_force(
                            send_signal,
                            signal,
                            send.position,
                            position,
                            2 ** 8 * count
                        )
                        for signal, count, position in all_signals
                    ]

                    # For each dimension, we can get the component forces to split the signals. There are two
                    # possibilities, either there is a receiving cell on both sides in which case they split the
                    # signals, or there is a receiving cell on only one side in which case the sending cell and the
                    # receiving cell split the signals (sending cell signals do not move).
                    total_force_all_dimensions = sum(abs(v) for f in forces for v in f.value)

                    if total_force_all_dimensions > 0:
                        for dimension in range(self.n_dimension):
                            total_positive_force: int = sum(f.value[dimension] for f in forces if f.value[dimension] > 0)
                            total_negative_force: int = sum(-f.value[dimension] for f in forces if f.value[dimension] < 0)
                            positive_target = receive_by_dimension(dimension, positive=True)
                            negative_target = receive_by_dimension(dimension, positive=False)

                            if positive_target is not None:
                                split_count = div(send_count * total_positive_force, total_force_all_dimensions)
                                transfers.append((send, positive_target, send_signal, split_count))

                            if negative_target is not None:
                                split_count = div(send_count * total_negative_force, total_force_all_dimensions)
                                transfers.append((send, negative_target, send_signal, split_count))

            # process the transfers at a later step to avoid mutations during force calculations
            for from_cell, to_position, signal, count in transfers:
                from_cell.add_signals(signal, -count)
                self.cells.get(to_position).add_signals(signal, count)

            # transition cells to SENDING
            for position in highest_receiving_positions:
                self.cells.get(position).state = SENDING

            # transition cells to STABLE when there are no adjacent receivers
            for send in sending_cells:
                for edit in self.adjacent_edits:
                    position = send.position + edit

                    if not position.out_of_bounds() and self.state_of(position) == RECEIVING:
                        break
                else:
                    send.state = STABLE

            # filter out stable cells and append new sending cells
            sending_cells = [c for c in sending_cells if c.state == SENDING]
            sending_cells += [self.cells.get(position) for position in highest_receiving_positions]
