from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar

T = TypeVar("T")


@dataclass
class Schedule:
    """
    Schedules the value of a varaible over time.
    """

    name: str
    start_value: float
    end_value: float
    _t: int
    n_steps: int

    def __init__(self, start_value: float, end_value: float, n_steps: int):
        self.start_value = start_value
        self.end_value = end_value
        self.n_steps = n_steps
        self.name = self.__class__.__name__
        self._t = 0
        self._current_value = self.start_value

    def update(self, step: Optional[int] = None):
        """Update the value of the schedule. Force a step if given."""
        if step is not None:
            self._t = step
        else:
            self._t += 1
        if self._t >= self.n_steps:
            self._current_value = self.end_value
        else:
            self._current_value = self._compute()

    @abstractmethod
    def _compute(self) -> float:
        """Compute the value of the schedule"""

    @property
    def value(self) -> float:
        """Returns the current value of the schedule"""
        return self._current_value

    @staticmethod
    def constant(value: float):
        return ConstantSchedule(value)

    @staticmethod
    def linear(start_value: float, end_value: float, n_steps: int):
        return LinearSchedule(start_value, end_value, n_steps)

    @staticmethod
    def exp(start_value: float, end_value: float, n_steps: int):
        return ExpSchedule(start_value, end_value, n_steps)

    @staticmethod
    def arbitrary(func: Callable[[int], float], n_steps: Optional[int] = None):
        if n_steps is None:
            n_steps = 0
        return ArbitrarySchedule(func, n_steps)

    def rounded(self, n_digits: int = 0) -> "RoundedSchedule":
        return RoundedSchedule(self, n_digits)

    # Operator overloading
    def __mul__(self, other: T) -> T:
        return self.value * other  # type: ignore

    def __rmul__(self, other: T) -> T:
        return self.value * other  # type: ignore

    def __pow__(self, exp: float) -> float:
        return self.value**exp

    def __rpow__(self, other: T) -> T:
        return other**self.value  # type: ignore

    def __add__(self, other: T) -> T:
        return self.value + other  # type: ignore

    def __radd__(self, other: T) -> T:
        return self.value + other  # type: ignore

    def __neg__(self):
        return -self.value

    def __pos__(self):
        return +self.value

    def __sub__(self, other: T) -> T:
        return self.value - other  # type: ignore

    def __rsub__(self, other: T) -> T:
        return other - self.value  # type: ignore

    def __div__(self, other: T) -> T:
        return self.value // other  # type: ignore

    def __rdiv__(self, other: T) -> T:
        return other // self.value  # type: ignore

    def __truediv__(self, other: T) -> T:
        return self.value / other  # type: ignore

    def __rtruediv__(self, other: T) -> T:
        return other / self.value  # type: ignore

    def __lt__(self, other) -> bool:
        return self.value < other

    def __le__(self, other) -> bool:
        return self.value <= other

    def __gt__(self, other) -> bool:
        return self.value > other

    def __ge__(self, other) -> bool:
        return self.value >= other

    def __eq__(self, other) -> bool:
        if isinstance(other, Schedule):
            if self.start_value != other.start_value:
                return False
            if self.end_value != other.end_value:
                return False
            if self.n_steps != other.n_steps:
                return False
            if type(self) is not type(other):
                return False
        return self.value == other

    def __ne__(self, other) -> bool:
        return not (self.__eq__(other))

    def __float__(self):
        return self.value

    def __int__(self) -> int:
        return int(self.value)


@dataclass(eq=False)
class LinearSchedule(Schedule):
    a: float
    b: float

    def __init__(self, start_value: float, end_value: float, n_steps: int):
        super().__init__(start_value, end_value, n_steps)
        self._current_value = self.start_value
        # y = ax + b
        self.a = (self.end_value - self.start_value) / self.n_steps
        self.b = self.start_value

    def _compute(self):
        return self.a * (self._t) + self.b

    @property
    def value(self) -> float:
        return self._current_value


@dataclass(eq=False)
class ExpSchedule(Schedule):
    """Exponential schedule. After n_steps, the value will be min_value.

    Update formula is next_value = start_value * (min_value / start_value) ** (step / (n - 1))
    """

    n_steps: int
    base: float
    last_update_step: int

    def __init__(self, start_value: float, min_value: float, n_steps: int):
        super().__init__(start_value, min_value, n_steps)
        self.base = self.end_value / self.start_value
        self.last_update_step = self.n_steps - 1

    def _compute(self):
        return self.start_value * (self.base) ** (self._t / (self.n_steps - 1))

    @property
    def value(self) -> float:
        return self._current_value


@dataclass(eq=False)
class ConstantSchedule(Schedule):
    def __init__(self, value: float):
        super().__init__(value, value, 0)
        self._value = value

    def update(self, step=None):
        return

    @property
    def value(self) -> float:
        return self._value


@dataclass(eq=False)
class RoundedSchedule(Schedule):
    def __init__(self, schedule: Schedule, n_digits: int):
        super().__init__(schedule.start_value, schedule.end_value, schedule.n_steps)
        self.schedule = schedule
        self.n_digits = n_digits

    def update(self, step: int | None = None):
        return self.schedule.update(step)

    def _compute(self) -> float:
        return self.schedule._compute()

    @property
    def value(self) -> float:
        return round(self.schedule.value, self.n_digits)


@dataclass(eq=False)
class MultiSchedule(Schedule):
    def __init__(self, schedules: dict[int, Schedule]):
        ordered_schedules, ordered_steps = MultiSchedule._verify(schedules)
        n_steps = ordered_steps[-1] + ordered_schedules[-1].n_steps
        super().__init__(ordered_schedules[0].start_value, ordered_schedules[-1].end_value, n_steps)
        self.schedules = iter(ordered_schedules)
        self.current_schedule = next(self.schedules)
        self.offset = 0
        self.current_end = ordered_steps[1]

    @staticmethod
    def _verify(schedules: dict[int, Schedule]):
        sorted_steps = sorted(schedules.keys())
        sorted_schedules = [schedules[t] for t in sorted_steps]
        if sorted_steps[0] != 0:
            raise ValueError("First schedule must start at t=0")
        current_step = 0
        for i in range(len(sorted_steps)):
            # Artificially set the end step of ConstantSchedules to the next step
            if isinstance(sorted_schedules[i], ConstantSchedule):
                if i + 1 < len(sorted_steps):
                    sorted_schedules[i].n_steps = sorted_steps[i + 1] - sorted_steps[i]
            if sorted_steps[i] != current_step:
                raise ValueError(f"Schedules are not contiguous at t={current_step}")
            current_step += sorted_schedules[i].n_steps
        return sorted_schedules, sorted_steps

    def update(self, step: int | None = None):
        if step is not None:
            raise NotImplementedError("Cannot update MultiSchedule with a specific step")
        super().update(step)
        # If we reach the end of the current schedule, update to the next one
        # except if we are at the end.
        if self._t == self.current_end and self._t < self.n_steps:
            self.current_schedule = next(self.schedules)
            self.offset = self._t
            self.current_end += self.current_schedule.n_steps
        self.current_schedule.update(self.relative_step)

    @property
    def relative_step(self):
        return self._t - self.offset

    def _compute(self) -> float:
        return self.current_schedule._compute()

    @property
    def value(self):
        return self.current_schedule.value


@dataclass(eq=False)
class ArbitrarySchedule(Schedule):
    def __init__(self, fn: Callable[[int], float], n_steps: int):
        super().__init__(fn(0), fn(n_steps), n_steps)
        self._func = fn

    def _compute(self) -> float:
        return self._func(self._t)
