from typing import Any

from typing_extensions import Self

from eitprocessing.continuous_data import ContinuousData
from eitprocessing.mixins.equality import Equivalence, EquivalenceError


class ContinuousDataCollection(dict, Equivalence):
    def __setitem__(self, __key: Any, __value: Any) -> None:
        self._check_data(__value, key=__key)
        return super().__setitem__(__key, __value)

    def add(self, data: ContinuousData, overwrite: bool = False) -> None:
        self._check_data(data, overwrite=overwrite)
        return super().__setitem__(data.name, data)

    def _check_data(
        self, data: ContinuousData, key=None, overwrite: bool = False
    ) -> None:
        if not isinstance(data, ContinuousData):
            raise TypeError(f"type of `data` is {type(data)}, not 'ContinuousData'")

        if key and key != data.name:
            raise KeyError(f"'{key}' does not match variant name '{data.name}'.")

        if not overwrite and key in self:
            raise KeyError(
                f"Variant with name {key} already exists. Use `overwrite=True` to overwrite."
            )
