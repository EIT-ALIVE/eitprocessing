from typing import Generic
from typing import TypeVar
from typing_extensions import Self
from ..helper import NotEquivalent
from . import Variant


V = TypeVar("V", bound="Variant")


class VariantCollection(dict, Generic[V]):
    variant_type: type[V]

    def __init__(self, variant_type: type[V], *args, **kwargs):
        self.variant_type = variant_type
        super().__init__(*args, **kwargs)

    def __setitem__(self, __key: str, __value: V) -> None:
        self._check_variant(__value, key=__key)
        return super().__setitem__(__key, __value)

    def add(self, variant: V, overwrite: bool = False) -> None:
        self._check_variant(variant, overwrite=overwrite)
        return super().__setitem__(variant.name, variant)

    def _check_variant(self, variant: V, key=None, overwrite: bool = False) -> None:
        if self.variant_type and not isinstance(variant, self.variant_type):
            raise InvalidVariantType(
                f"'{type(variant)}' does not match '{self.variant_type}'."
            )

        if key and key != variant.name:
            raise KeyError(f"'{key}' does not match variant name '{variant.name}'.")

        if not overwrite and key in self:
            raise DuplicateVariantName(
                f"Variant with name {key} already exists. Use `overwrite=True` to overwrite."
            )

    @classmethod
    def concatenate(cls, a: Self, b: Self) -> Self:
        try:
            cls.check_equivalence(a, b, raise_=True)
        except NotEquivalent as e:
            raise ValueError("VariantCollections could not be concatenated") from e

        obj = VariantCollection(a.variant_type)
        for key in a.keys():
            obj.add(a.variant_type.concatenate(a[key], b[key]))

        return obj

    @classmethod
    def check_equivalence(cls, a: Self, b: Self, raise_=False) -> bool:
        try:
            if a.variant_type != b.variant_type:
                raise NotEquivalent(
                    f"Variant types do not match: {a.variant_type}, {b.variant_type}"
                )

            if set(a.keys()) != set(b.keys()):
                raise NotEquivalent(
                    f"VariantCollections do not contain the same variants: {a.keys()=}, {b.keys()=}"
                )

            for key in a.keys():
                Variant.check_equivalence(a[key], b[key], raise_=True)

        except NotEquivalent:
            # re-raises the exceptions if raise_ is True, or returns False
            if raise_:
                raise
            return False

        return True


class InvalidVariantType(Exception):
    """Raised when a variant that does not match the variant type is added."""


class DuplicateVariantName(Exception):
    """Raised when a variant with the same name already exists in the collection."""
