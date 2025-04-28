from __future__ import annotations

import itertools
import sys
from dataclasses import MISSING, dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar, overload

from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.datacollection import DataCollection
from eitprocessing.datahandling.eitdata import EITData
from eitprocessing.datahandling.intervaldata import IntervalData
from eitprocessing.datahandling.mixins.equality import Equivalence
from eitprocessing.datahandling.mixins.slicing import SelectByTime
from eitprocessing.datahandling.sparsedata import SparseData

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np
    from typing_extensions import Self

    from eitprocessing.parameters import DataContainer

T = TypeVar("T", bound=Any)


@dataclass(eq=False)
class Sequence(Equivalence, SelectByTime):
    """Sequence of timepoints containing respiratory data.

    A Sequence object is a representation of data points over time. These data can consist of any combination of EIT
    frames (`EITData`), waveform data (`ContinuousData`) from different sources, or individual events (`SparseData`)
    occurring at any given timepoint. A Sequence can consist of an entire measurement, a section of a measurement, a
    single breath, or even a portion of a breath. A Sequence can consist of multiple sets of each type of data from the
    same time-points or can be a single measurement from just one source.

    A Sequence can be split up into separate sections of a measurement or multiple (similar) Sequence objects can be
    merged together to form a single Sequence.

    Args:
        label: Computer readable naming of the instance.
        name: Human readable naming of the instance.
        description: Human readable extended description of the data.
        eit_data: Collection of one or more sets of EIT data frames.
        continuous_data: Collection of one or more sets of continuous data points.
        sparse_data: Collection of one or more sets of individual data points.
    """  # TODO: check that docstring is up to date

    label: str | None = field(default=None, compare=False)
    name: str | None = field(default=None, compare=False, repr=False)
    description: str = field(default="", compare=False, repr=False)
    eit_data: DataCollection[EITData] = field(default_factory=lambda: DataCollection(EITData), repr=False)
    continuous_data: DataCollection[ContinuousData] = field(
        default_factory=lambda: DataCollection(ContinuousData),
        repr=False,
    )
    sparse_data: DataCollection[SparseData] = field(default_factory=lambda: DataCollection(SparseData), repr=False)
    interval_data: DataCollection[IntervalData] = field(
        default_factory=lambda: DataCollection(IntervalData),
        repr=False,
    )

    def __post_init__(self):
        if not self.label:
            self.label = f"Sequence_{id(self)}"
        self.name = self.name or self.label

    @property
    def time(self) -> np.ndarray:
        """Time axis from either EITData or ContinuousData."""
        if len(self.eit_data):
            return self.eit_data["raw"].time
        if len(self.continuous_data):
            return next(iter(self.continuous_data.values())).time

        msg = "Sequence has no timed data"
        raise AttributeError(msg)

    def __len__(self):
        return len(self.time)

    def __add__(self, other: Sequence) -> Sequence:
        return self.concatenate(self, other)

    @classmethod  # TODO: why is this a class method? In other cases it's instance method
    def concatenate(
        cls,
        a: Sequence,
        b: Sequence,
        newlabel: str | None = None,
    ) -> Sequence:
        """Create a merge of two Sequence objects."""
        # TODO: rewrite

        concat_eit = a.eit_data.concatenate(b.eit_data)
        concat_continuous = a.continuous_data.concatenate(b.continuous_data)
        concat_sparse = a.sparse_data.concatenate(b.sparse_data)
        concat_interval = a.interval_data.concatenate(b.interval_data)

        newlabel = newlabel or a.label
        # TODO: add concatenation of other attached objects

        return a.__class__(
            eit_data=concat_eit,
            continuous_data=concat_continuous,
            sparse_data=concat_sparse,
            interval_data=concat_interval,
            label=newlabel,
        )

    def _sliced_copy(self, start_index: int, end_index: int, newlabel: str) -> Self:  # noqa: ARG002
        if start_index >= len(self.time):
            msg = "start_index larger than length of time axis"
            raise ValueError(msg)
        time = self.time[start_index:end_index]

        sliced_eit = DataCollection(EITData)
        for value in self.eit_data.values():
            sliced_eit.add(value[start_index:end_index])

        sliced_continuous = DataCollection(ContinuousData)
        for value in self.continuous_data.values():
            sliced_continuous.add(value[start_index:end_index])

        sliced_sparse = DataCollection(SparseData)
        for value in self.sparse_data.values():
            sliced_sparse.add(value.t[time[0] : time[-1]])

        sliced_interval = DataCollection(IntervalData)
        for value in self.interval_data.values():
            sliced_interval.add(value.t[time[0] : time[-1]])

        return self.__class__(
            label=self.label,  # newlabel gives errors
            name=f"Sliced copy of <{self.name}>",
            description=f"Sliced copy of <{self.description}>",
            eit_data=sliced_eit,
            continuous_data=sliced_continuous,
            sparse_data=sliced_sparse,
            interval_data=sliced_interval,
        )

    def select_by_time(
        self,
        start_time: float | None = None,
        end_time: float | None = None,
        start_inclusive: bool = True,
        end_inclusive: bool = False,
        label: str | None = None,
        name: str | None = None,
        description: str = "",
    ) -> Self:
        """Return a sliced version of the Sequence.

        See `SelectByTime.select_by_time()`.
        """
        if not label:
            label = f"copy_of_<{self.label}>"
        if not name:
            f"Sliced copy of <{self.name}>"

        return self.__class__(
            label=label,
            name=name,
            description=description,
            # perform select_by_time() on all four data types
            **{
                key: getattr(self, key).select_by_time(
                    start_time=start_time,
                    end_time=end_time,
                    start_inclusive=start_inclusive,
                    end_inclusive=end_inclusive,
                )
                for key in ("eit_data", "continuous_data", "sparse_data", "interval_data")
            },
        )

    @property
    def data(self) -> _DataAccess:
        """Shortcut access to data stored in collections inside a sequence.

        This allows all data objects stored in a collection inside a sequence to be accessed.
        Instead of `sequence.continuous_data["global_impedance"]` you can use
        `sequence.data["global_impedance"]`. This works for getting (`sequence.data["label"]` or
        `sequence.data.get("label")`) and adding data (`sequence.data["label"] = obj` or
        `sequence.data.add(obj)`).

        Other dict-like behaviour is also supported:

        - `label in sequence.data` to check whether an object with a label exists;
        - `del sequence.data[label]` to remove an object from the sequence based on the label;
        - `for label in sequence.data` to iterate over the labels;
        - `sequence.data.items()` to retrieve a list of (label, object) pairs, especially useful for iteration;
        - `sequence.data.labels()` or `sequence.data.keys()` to get a list of data labels;
        - `sequence.data.objects()` or `sequence.data.values()` to get a list of data objects.

        This interface only works if the labels are unique among the data collections. An attempt
        to add a data object with an exiting label will result in a KeyError.
        """
        return _DataAccess(self)


@dataclass
class _DataAccess:
    _sequence: Sequence

    def __post_init__(self):
        for a, b in itertools.combinations(self._collections, 2):
            if duplicates := set(a) & set(b):
                msg = f"Duplicate labels ({', '.join(sorted(duplicates))}) found in {a} and {b}."
                exc = KeyError(msg)
                if sys.version_info >= (3, 11):
                    exc.add_note(
                        "You can't use the `data` interface with duplicate labels. "
                        "Use the explicit data collections (`eit_data`, `continuous_data`, `sparse_data`, "
                        "`interval_data`) instead."
                    )
                raise exc

    @property
    def _collections(self) -> tuple[DataCollection, ...]:
        return (
            self._sequence.continuous_data,
            self._sequence.interval_data,
            self._sequence.sparse_data,
            self._sequence.eit_data,
        )

    @overload
    def get(self, label: str) -> DataContainer: ...

    @overload
    def get(self, label: str, default: T) -> DataContainer | T: ...

    def get(self, label: str, default: object = MISSING) -> DataContainer | object:
        """Get a DataContainer object by label.

        Example:
        ```
        if filtered_data := sequence.data.get("filtered data", None):
            print(filtered_data.values.mean())
        else:
            print("No filtered data was found.")

        ```

        Args:
            label (str): label of the object to retrieve.
            default (optional): a default value that is returned if the object is not found.
                Defaults to MISSING.

        Raises:
            KeyError: if the object is not found, and no default was set.

        Returns:
            DataContainer: the requested DataContainer.
        """
        for collection in self._collections:
            if label in collection:
                return collection[label]

        if default is not MISSING:
            return default

        msg = f"No object with label {label} was found."
        raise KeyError(msg)

    def __getitem__(self, key: str) -> DataContainer:
        return self.get(key)

    def add(self, *obj: DataContainer) -> None:
        """Add a DataContainer object to the sequence.

        Adds the object to the appropriate data collection. The label of the object must be unique
        among all data collections, otherwise a KeyError is raised.

        Args:
            obj (DataContainer): the object to add to the Sequence.

        Raises:
            KeyError: if the label of the object already exists in any of the data collections.
        """
        for object_ in obj:
            if self.get(object_.label, None):
                msg = f"An object with the label {object_.label} already exists in this sequence."
                exc = KeyError(msg)
                if sys.version_info >= (3, 11):
                    exc.add_note(
                        "You can't add an object with the same label through the `data` interface. "
                        "Use the explicit data collections (`eit_data`, `continuous_data`, `sparse_data`, "
                        "`interval_data`) instead."
                    )
                raise exc

            match object_:
                case ContinuousData():
                    self._sequence.continuous_data.add(object_)
                case IntervalData():
                    self._sequence.interval_data.add(object_)
                case SparseData():
                    self._sequence.sparse_data.add(object_)
                case EITData():
                    self._sequence.eit_data.add(object_)

    def __setitem__(self, label: str, obj: DataContainer):
        if obj.label != label:
            msg = f"Label {label} does not match object label {obj.label}."
            raise KeyError(msg)
        return self.add(obj)

    def __contains__(self, label: str) -> bool:
        return any(label in container for container in self._collections)

    def __delitem__(self, label: str) -> None:
        for container in self._collections:
            if label in container:
                del container[label]
                return

        msg = f"Object with label {label} was not found."
        raise KeyError(msg)

    def __iter__(self) -> Iterator[str]:
        return itertools.chain(*[collection.keys() for collection in self._collections])

    def items(self) -> list[tuple[str, DataContainer]]:
        """Return all data items (`(label, object)` pairs)."""
        return list(itertools.chain(*[collection.items() for collection in self._collections]))

    def keys(self) -> list[str]:
        """Return a list of all labels."""
        return list(self.__iter__())

    labels = keys

    def values(self) -> list[DataContainer]:
        """Return all data objects."""
        return list(itertools.chain(*[collection.values() for collection in self._collections]))

    objects = values
