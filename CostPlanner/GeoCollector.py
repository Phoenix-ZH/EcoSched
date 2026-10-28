"""Exact non-preemptive GPU capacity calendars with interval backfilling.

Each pool contains interchangeable devices. Reservations consume an integer
number of devices over half-open intervals [start, finish). Maintaining the
aggregate occupancy at every change point is sufficient to enforce the pool's
capacity; individual device identifiers can be assigned when execution starts.
Unlike an append-only device queue, an interval calendar retains idle gaps
before future reservations and can safely place shorter workloads into them.
"""

import math
from bisect import bisect_left, bisect_right


__all__ = ["CapacityCalendar"]


def _count(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("{} must be an integer".format(name))
    if value < minimum:
        raise ValueError("{} must be >= {}".format(name, minimum))
    return value


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError("{} must be a finite number".format(name))
    if not math.isfinite(value):
        raise ValueError("{} must be finite".format(name))
    return float(value)


class CapacityCalendar:
    __slots__ = ("capacity", "_times", "_loads", "_reservations", "_next_token",
                 "_windows", "_areas")

    def __init__(self, capacity):
        self.capacity = _count(capacity, "capacity", 0)
        self._times = [-math.inf]
        self._loads = [0]
        self._reservations = {}
        self._next_token = 0
        self._windows = {}
        self._areas = None

    def clone(self):
        """Return an independent calendar with the same reservations and tokens."""
        result = CapacityCalendar(self.capacity)
        result._times = self._times.copy()
        result._loads = self._loads.copy()
        result._reservations = self._reservations.copy()
        result._next_token = self._next_token
        # Cached values are immutable tuples. Each copy owns its cache mapping
        # and invalidates it independently when its reservations change.
        result._windows = self._windows.copy()
        result._areas = self._areas
        return result

    @property
    def event_count(self):
        """Return the number of finite occupancy change points."""
        return len(self._times) - 1

    @property
    def reservation_count(self):
        """Return the number of active reservation tokens."""
        return len(self._reservations)

    def free_at(self, time):
        """Return the number of unreserved devices at the specified instant."""
        time = _finite(time, "time")
        return self.capacity - self._loads[bisect_right(self._times, time) - 1]

    def earliest_start(self, earliest, runtime, count, deadline):
        """Find the first feasible start whose complete execution meets deadline.

        Consecutive occupancy segments with sufficient capacity form one free
        interval. Repeated queries for the same gang size reuse those intervals
        and skip past expired ones by binary search. Ending exactly when
        another reservation begins is feasible.
        """
        earliest = _finite(earliest, "earliest")
        runtime = _finite(runtime, "runtime")
        deadline = _finite(deadline, "deadline")
        count = _count(count, "count")
        if runtime <= 0:
            raise ValueError("runtime must be positive")
        if count > self.capacity or earliest + runtime > deadline:
            return None
        windows = self._windows.get(count)
        if windows is None:
            windows = self._free_windows(count)
            self._windows[count] = windows
        starts, ends = windows
        index = bisect_right(ends, earliest)
        while index < len(starts):
            start = max(earliest, starts[index])
            finish = start + runtime
            if finish > deadline:
                return None
            if finish <= ends[index]:
                return start
            index += 1
        return None

    def _free_windows(self, count):
        limit = self.capacity - count
        starts, ends = [], []
        opening = None
        for time, load in zip(self._times, self._loads):
            if load <= limit:
                if opening is None:
                    opening = time
            elif opening is not None:
                starts.append(opening)
                ends.append(time)
                opening = None
        if opening is not None:
            starts.append(opening)
            ends.append(math.inf)
        return tuple(starts), tuple(ends)

    def available_device_time(self, start, finish):
        """Integrate residual capacity using cached change-point prefix areas."""
        start = _finite(start, "start")
        finish = _finite(finish, "finish")
        if finish <= start:
            return 0.0
        times, loads = self._times, self._loads
        left = bisect_right(times, start) - 1
        right = bisect_left(times, finish) - 1
        if left == right:
            return (finish - start) * (self.capacity - loads[left])
        if self._areas is None:
            self._build_areas()
        # Integrate partial boundary segments directly. Prefix subtraction is
        # confined to complete interior segments, avoiding cancellation when
        # a short query lies far from the start of the calendar.
        leading = (times[left + 1] - start) * (self.capacity - loads[left])
        trailing = (finish - times[right]) * (self.capacity - loads[right])
        interior = self._areas[right] - self._areas[left + 1]
        return max(0.0, math.fsum((leading, interior, trailing)))

    def _build_areas(self):
        areas = [0.0] * len(self._times)
        total = correction = 0.0
        for index in range(2, len(self._times)):
            piece = ((self._times[index] - self._times[index - 1])
                     * (self.capacity - self._loads[index - 1]))
            adjusted = piece - correction
            value = total + adjusted
            correction = (value - total) - adjusted
            total = value
            areas[index] = total
        self._areas = tuple(areas)

    def reserve(self, start, finish, count):
        """
        An infeasible reservation raises ValueError without changing occupancy.
        Capacity is checked independently of any preceding placement query, so
        proposals made against stale snapshots cannot overbook the pool.
        """
        start = _finite(start, "start")
        finish = _finite(finish, "finish")
        count = _count(count, "count")
        if finish <= start:
            raise ValueError("finish must be greater than start")
        if count > self.capacity:
            raise ValueError("reservation exceeds pool capacity")
        left = bisect_right(self._times, start) - 1
        right = bisect_left(self._times, finish)
        if any(load + count > self.capacity for load in self._loads[left:right]):
            raise ValueError("reservation overlaps occupied devices")
        self._change(start, finish, count)
        token = self._next_token
        self._next_token += 1
        self._reservations[token] = (start, finish, count)
        return token

    def release(self, token):
        """Cancel exactly one reservation; unknown tokens leave state unchanged."""
        _count(token, "token", 0)
        start, finish, count = self._reservations[token]
        self._change(start, finish, -count)
        del self._reservations[token]

    def _split(self, time):
        index = bisect_left(self._times, time)
        if index == len(self._times) or self._times[index] != time:
            self._times.insert(index, time)
            self._loads.insert(index, self._loads[index - 1])
        return index

    def _change(self, start, finish, delta):
        self._windows.clear()
        self._areas = None
        left = self._split(start)
        right = self._split(finish)
        for index in range(left, right):
            self._loads[index] += delta
        # Interior differences are unchanged because every segment receives
        # the same delta; only the two window boundaries can become redundant.
        if self._loads[right] == self._loads[right - 1]:
            del self._times[right]
            del self._loads[right]
        if self._loads[left] == self._loads[left - 1]:
            del self._times[left]
            del self._loads[left]
