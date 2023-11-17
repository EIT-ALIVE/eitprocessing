from typing import Any
from typing_extensions import Self
from ..helper import NotEquivalent
from . import ContinuousData


class ContinuousDataCollection(dict):
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
            raise DuplicateContinuousDataName(
                f"Variant with name {key} already exists. Use `overwrite=True` to overwrite."
            )

    @classmethod
    def concatenate(cls, a: Self, b: Self) -> Self:
        try:
            cls.check_equivalence(a, b, raise_=True)
        except NotEquivalent as e:
            raise ValueError("VariantCollections could not be concatenated") from e

        obj = ContinuousDataCollection()
        for key in a.keys() & b.keys():
            obj.add(ContinuousData.concatenate(a[key], b[key]))

        return obj

    @classmethod
    def check_equivalence(cls, a: Self, b: Self, raise_=False) -> bool:
        try:
            if set(a.keys()) != set(b.keys()):
                raise NotEquivalent(
                    f"VariantCollections do not contain the same variants: {a.keys()=}, {b.keys()=}"
                )

            for key in a.keys():
                ContinuousData.check_equivalence(a[key], b[key], raise_=True)

        except NotEquivalent:
            # re-raises the exceptions if raise_ is True, or returns False
            if raise_:
                raise
            return False

        return True


class DuplicateContinuousDataName(Exception):
    """Raised when a variant with the same name already exists in the collection."""
