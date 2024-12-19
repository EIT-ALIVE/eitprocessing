from __future__ import annotations

from collections import UserDict
from typing import TYPE_CHECKING, Generic, TypeVar

from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.eitdata import EITData
from eitprocessing.datahandling.intervaldata import IntervalData
from eitprocessing.datahandling.mixins.equality import Equivalence
from eitprocessing.datahandling.mixins.slicing import HasTimeIndexer
from eitprocessing.datahandling.sparsedata import SparseData

if TYPE_CHECKING:
    from typing_extensions import Self


V = TypeVar("V", EITData, ContinuousData, SparseData, IntervalData)
V_classes = V.__constraints__


class DataCollection(Equivalence, UserDict, HasTimeIndexer, Generic[V]):
    """A collection of a single type of data with unique labels.

    A DataCollection functions largely as a dictionary, but requires a data_type argument, which must be one of the data
    containers existing in this package. When adding an item to the collection, the type of the value must match the
    data_type of the collection. Furthermore, the key has to match the attribute 'label' attached to the value.

    The convenience method `add()` adds an item by setting the key to `value.label`.

    Args:
        data_type: the data container stored in this collection.
    """

    data_type: type

    def __init__(self, data_type: type[V], *args, **kwargs):
        if not any(issubclass(data_type, cls) for cls in V_classes):
            msg = f"Type {data_type} not expected to be stored in a DataCollection."
            raise TypeError(msg)
        self.data_type = data_type
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: str, value: V, /) -> None:
        self._check_item(value, key=key)
        return super().__setitem__(key, value)

    def add(self, *item: V, overwrite: bool = False) -> None:
        """Add one or multiple item(s) to the collection using the item label as the key."""
        for item_ in item:
            self._check_item(item_, overwrite=overwrite)
            super().__setitem__(item_.label, item_)

    def _check_item(
        self,
        item: V,
        key: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """Check whether the item can be added to the collection.

        In order to be added to the collection, the data type of the item has to match the data type set in the
        collection. They key that is used to store the item in the collection has to match the label of the item itself.
        By default, existing keys can not be overridden.

        Args:
            item: Object to be added to the collection.
            key: Key of the item. Has to match `item.label`.
            overwrite: If False, the key can not already exist in the collection. Set to True to allow overwriting an
            existing object in the collection.

        Raises:
            TypeError: If the type of the item does not match the type set in the collection.
            KeyError: If the key does not match `item.label`, or when the key already exists in de collection and
            overwrite is set to False.
        """
        if not isinstance(item, self.data_type):
            msg = f"Type of `data` is {type(item)}, not '{self.data_type}'"
            raise TypeError(msg)

        if key and key != item.label:
            msg = f"'{key}' does not match label '{item.label}'. Keys to Collection items must match their labels."
            raise KeyError(msg)

        if not key:
            key = item.label

        if not overwrite and key in self:
            # Generally it is not expected one would want to overwrite existing data with different/derived data.
            # One should probably change the label instead over overwriting existing data.
            # To force an overwrite, use __setitem__ and set `overwrite=True`.
            msg = (
                f"Item with label {key} already exists and cannot be overwritten. Please use a different label instead."
            )
            raise KeyError(msg)

    def get_loaded_data(self) -> dict[str, V]:
        """Return all data that was directly loaded from disk."""
        return {k: v for k, v in self.items() if v.loaded}

    def get_data_derived_from(self, obj: V) -> dict[str, V]:
        """Return all data that was derived from a specific source."""
        return {k: v for k, v in self.items() if any(obj is item for item in v.derived_from)}

    def get_derived_data(self) -> dict[str, V]:
        """Return all data that was derived from any source."""
        return {k: v for k, v in self.items() if v.derived_from}

    def concatenate(self: Self, other: Self) -> Self:
        """Concatenate this collection with an equivalent collection.

        Each item of self of concatenated with the item of other with the same key.
        """
        self.isequivalent(other, raise_=True)

        concatenated = self.__class__(self.data_type)
        for key in self:
            concatenated.add(self[key].concatenate(other[key]))

        return concatenated

    def select_by_time(
        self,
        start_time: float | None,
        end_time: float | None,
        start_inclusive: bool = True,
        end_inclusive: bool = False,
    ) -> DataCollection:
        """Return a DataCollection containing sliced copies of the items."""
        if self.data_type is IntervalData:
            return DataCollection(
                self.data_type,
                **{
                    k: v.select_by_time(
                        start_time=start_time,
                        end_time=end_time,
                    )
                    for k, v in self.items()
                },
            )

        return DataCollection(
            self.data_type,
            **{
                k: v.select_by_time(
                    start_time=start_time,
                    end_time=end_time,
                    start_inclusive=start_inclusive,
                    end_inclusive=end_inclusive,
                )
                for k, v in self.items()
            },
        )
