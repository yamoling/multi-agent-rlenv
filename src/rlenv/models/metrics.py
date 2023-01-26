from dataclasses import dataclass
from typing import Iterable, Union
import numpy as np


@dataclass
class Measurement:
    """
    The `Measurement` class is defined by a value, a unit and a print format. It provides
    arithmetic operators for usability and is mainly used for logging purposes.
    """
    value: float
    fmt: str | None
    unit: str | None

    def __init__(self, value, fmt=None, unit="") -> None:
        self.value = value
        self.fmt = fmt
        self.unit = unit
        if fmt is None and isinstance(value, (int, float)):
            self.fmt = ".3f"

    def __add__(self, other: Union["Measurement", float]) -> "Measurement":
        fmt = self.fmt
        unit = self.unit
        if isinstance(other, Measurement):
            assert self.unit is None or other.unit is None or self.unit == other.unit, "Cannot add two measurements with different units"
            if fmt is None:
                fmt = other.fmt
            if unit is None:
                unit = other.unit
            other_value = other.value
        else:
            other_value = other
        return Measurement(self.value + other_value, fmt, unit)

    def __str__(self)-> str:
        return format(self.value, self.fmt) + self.unit

    def __lt__(self, other: Union["Measurement", float]) -> bool:
        if isinstance(other, Measurement):
            assert self.unit == other.unit, "Can not compare measurements with different units"
            value = other.value
        elif isinstance(other, float):
            value = other
        else:
            raise NotImplementedError(f"Comparison not implemented against type {type(other)}")
        return self.value < value

    def __le__(self, other: Union["Measurement", float]) -> bool:
        if isinstance(other, Measurement):
            assert self.unit == other.unit, "Can not compare measurements with different units"
            value = other.value
        elif isinstance(other, float):
            value = other
        else:
            raise NotImplementedError(f"Comparison not implemented against type {type(other)}")
        return self.value <= value

    def __gt__(self, other: Union["Measurement", float]) -> bool:
        if isinstance(other, Measurement):
            assert self.unit == other.unit, "Can not compare measurements with different units"
            value = other.value
        elif isinstance(other, float):
            value = other
        else:
            raise NotImplementedError(f"Comparison not implemented against type {type(other)}")
        return self.value > value

    def __ge__(self, other: Union["Measurement", float]) -> bool:
        if isinstance(other, Measurement):
            assert self.unit == other.unit, "Can not compare measurements with different units"
            value = other.value
        elif isinstance(other, float):
            value = other
        else:
            raise NotImplementedError(f"Comparison not implemented against type {type(other)}")
        return self.value >= value

    def __eq__(self, other: Union["Measurement", float]) -> bool:
        if isinstance(other, Measurement):
            assert self.unit == other.unit, "Can not compare measurements with different units"
            value = other.value
        elif isinstance(other, float):
            value = other
        else:
            raise NotImplementedError(f"Comparison not implemented against type {type(other)}")
        return self.value == value

    def __truediv__(self, divider: float):
        return Measurement(self.value/divider, self.fmt, self.unit)


class Metrics(dict):
    """Metrics are just a dictionary of type [str, Measurement] with facilities such as adding, dividing and averaging methods."""

    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if not isinstance(value, Measurement):
                kwargs[key] = Measurement(value)
        super().__init__(**kwargs)

    def __truediv__(self, divider: float) -> "Metrics":
        res = Metrics(**self)
        for key, value  in self.items():
            res[key] = value / divider
        return res

    def __add__(self, other: "Metrics") -> "Metrics":
        """Add all measurements pairwise."""
        res = Metrics(**self)
        for key in other:
            if key not in res:
                res[key] = other[key]
            else:
                res[key] += other[key]
        return res

    def __getitem__(self, key: str) -> Measurement:
        # pylint: disable = W0235
        """Just for type hinting"""
        return super().__getitem__(key)

    def __setitem__(self, key: str, value: Measurement | float | int) -> None:
        if not isinstance(value, Measurement):
            value = Measurement(value)
        return super().__setitem__(key, value)

    def items(self) -> Iterable[tuple[str, Measurement]]:
        """Just for type hinting"""
        return super().items()

    @staticmethod
    def agregate(all_metrics: list["Metrics"]) -> "Metrics":
        """Aggregate a list of metrics into min, max, avg and std."""
        all_values: dict[str, list[Measurement]] = {}
        for metrics in all_metrics:
            for key, value in metrics.items():
                if key not in all_values:
                    all_values[key] = []
                all_values[key].append(value)
        res = Metrics()
        for key, values in all_values.items():
            unit = values[0].unit
            fmt = values[0].fmt
            values = np.array([v.value for v in values])
            res[f"avg_{key}"] = Measurement(np.average(values), fmt, unit)
            res[f"std_{key}"] = Measurement(np.std(values), fmt, unit)
            res[f"min_{key}"] = Measurement(values.min(), fmt, unit)
            res[f"max_{key}"] = Measurement(values.max(), fmt, unit)
        return res

    @property
    def score(self) -> float:
        """Score"""
        try:
            return self["score"].value
        except KeyError:
            return self["avg_score"].value
