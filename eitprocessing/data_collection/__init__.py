from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

from eitprocessing.continuous_data import ContinuousData
from eitprocessing.eit_data import EITData
from eitprocessing.mixins.equality import Equivalence
from eitprocessing.sparse_data import SparseData

if TYPE_CHECKING:
    from typing_extensions import Self

V = TypeVar("V", EITData, ContinuousData, SparseData)


class DataCollection(dict, Equivalence, Generic[V]):
    """A collection of a single type of data with unique labels.

    This collection functions as a dictionary in most part. When initializing, a data type has to be passed. EITData,
    ContinuousData or SparseData is expected as the data type. Other types are allowed, but not supported. The objects
    added to the collection need to have a `label` attribute and a `concatenate()` method.

    When adding an item to the collection, the type of the value has to match the data type of the collection.
    Furthermore, the key has to match the attribute 'label' attached to the value.

    The convenience method `add()` adds an item by setting the key to `value.label`.

    Args:
        data_type: the type of data stored in this collection. Expected to be one of EITData, ContinuousData or
        SparseData.
    """

    data_type: type

    def __init__(self, data_type: type[V], *args, **kwargs):
        if not any(issubclass(data_type, cls) for cls in V.__constraints__):
            msg = f"Type {data_type} not expected to be stored in a DataCollection."
            raise ValueError(msg)
        self.data_type = data_type
        super().__init__(*args, **kwargs)

    def __setitem__(self, __key: str, __value: V) -> None:
        self._check_item(__value, key=__key)
        return super().__setitem__(__key, __value)

    def add(self, *item: V, overwrite: bool = False) -> None:
        """Add one or multiple item(s) to the collection."""
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
            # It is expected that an item in this collection has a key equal to the label of the value.
            msg = f"'{key}' does not match label '{item.label}'."
            raise KeyError(msg)

        if not overwrite and key in self:
            # Generally it is not expected one would want to overwrite existing data with different/derived data. One
            # should probably change the label instead over overwriting existing data.
            msg = f"Item with label {key} already exists. Use `overwrite=True` to overwrite."
            raise KeyError(msg)

    def get_loaded_data(self) -> dict[str, V]:
        """Return all data that was directly loaded from disk."""
        return {k: v for k, v in self.items() if v.loaded}

    def get_data_derived_from(self, obj: V) -> dict[str, V]:
        """Return all data that was derived from a specific source."""
        return {k: v for k, v in self.items() if obj in v.derived_from}

    def get_derived_data(self) -> dict[str, V]:
        """Return all data that was derived from any source."""
        return {k: v for k, v in self.items() if v.derived_from}

    def concatenate(self: Self[V], other: Self[V]) -> Self[V]:
        """Concatenate this collection with an equivalent collection.

        Each item of self of concatenated with the item of other with the same key.
        """
        self.isequivalent(other, raise_=True)

        concatenated = self.__class__(self.data_type)
        for key in self:
            concatenated[key] = self[key].concatenate(other[key])

        return concatenated
