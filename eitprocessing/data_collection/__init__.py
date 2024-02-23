from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

from eitprocessing.mixins.equality import Equivalence

if TYPE_CHECKING:
    from typing_extensions import Self

V = TypeVar("V")


class DataCollection(dict, Equivalence, Generic[V]):
    data_type: type

    def __init__(self, data_type: type[V], *args, **kwargs):
        self.data_type = data_type
        super().__init__(*args, **kwargs)

    def __setitem__(self, __key: str, __value: V) -> None:
        self._check_item(__value, key=__key)
        return super().__setitem__(__key, __value)

    def add(self, *item: V, overwrite: bool = False) -> None:
        for item_ in item:
            self._check_item(item_, overwrite=overwrite)
            super().__setitem__(item_.label, item_)

    def _check_item(
        self,
        item: V,
        key: str | None = None,
        overwrite: bool = False,
    ) -> None:
        if not isinstance(item, self.data_type):
            msg = f"Type of `data` is {type(item)}, not '{self.data_type}'"
            raise TypeError(msg)

        if key and key != item.label:
            msg = f"'{key}' does not match label '{item.label}'."
            raise KeyError(msg)

        if not overwrite and key in self:
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
        return {k: v for k, v in self.items() if len(v.derived_from) >= 1}

    def concatenate(self, other: Self[V]) -> Self[V]:
        self.isequivalent(other, raise_=True)

        concatenated = self.__class__(self.data_type)
        for key in self.keys():
            concatenated[key] = self[key].concatenate(other[key])

        return concatenated
