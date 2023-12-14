from typing import Generic
from typing import TypeVar
from typing_extensions import Self
from eitprocessing.mixins.equality import Equivalence
from eitprocessing.mixins.equality import EquivalenceError
from eitprocessing.variants import Variant


V = TypeVar("V", bound="Variant")


class VariantCollection(dict, Equivalence, Generic[V]):
    """A collection of variants of a single type

    A VariantCollection is a dictionary with some added features.

    A VariantCollection can only contain variants (of a certain type). When
    initializing VariantCollection a subclass of Variant (or Variant itself to
    allow diverse types) must be passed as the first argument (`variant_type`),
    limiting the type of variant that is allowed to be added to the collection.
    During initialization, other arguments can be passed as if initializing a
    normal dictionary.

    When adding a variant, the key of the item in the dictionary must equal the
    label of the variant. The method `add()` can be used instead, which
    automatically sets the variant label as the key.

    Thirdly, when setting a variant with a label that already exists, the
    default behaviour is to raise an exception. This prevents overwriting
    existing variants. This behaviour can be overridden using `add(variant,
    overwrite=True)`.

    NB: due to how items are set, a variant with the name 'variant_type' can't
    be

    Args:
    - variant_type (type[Variant]): the type of variant this collection will
    contain
    - *args: will be passed to dict() unchanged
    - **kwargs: will be passed to dict() unchanged

    Examples:
    ```
    >>> variant_a = EITDataVariant(label="raw", ...)
    >>> vc = VariantCollection(EITDataVariant, raw=variant_a)
    >>> variant_b = EITDataVariant(label="filtered", ...)
    >>> vc.add(variant_b)  # equals vc["filtered"] = variant
    ```

    ```
    >>> vc = VariantCollection(EITDataVariant)
    >>> variant_c = SomeOtherVariant(label="offset", ...)
    >>> vc.add(variant_c)  # raises TypeError() exception
    ```

    """

    variant_type: type[V]

    def __eq__(self, other):
        return self.variant_type == other.variant_type and super().__eq__(other)

    def __init__(self, variant_type: type[V], *args, **kwargs):
        if not issubclass(variant_type, Variant):
            raise TypeError(f"{variant_type} is not a subclass of Variant")

        self.variant_type = variant_type
        super().__init__(*args, **kwargs)

    def __setitem__(self, __key: str, __value: V) -> None:
        self._check_variant(__value, key=__key)
        return super().__setitem__(__key, __value)

    def add(self, *variant: V, overwrite: bool = False) -> None:
        """Add one or multiple variants to the dictionary

        This method automatically sets the key of the item to the label of the
        variant. By default, overwriting variants with the same label is
        prevented. Trying to do so will result in a KeyError being raised. Set
        `overwrite` to `True` to allow overwriting.

        Args:
        - variant (Variant): the variant to be added. Multiple variants can be
          added at once.

        Raises:
        - KeyError if one attempts to add a variant with a label
          that already exists as key.
        """
        for variant_ in variant:
            self._check_variant(variant_, overwrite=overwrite)
            super().__setitem__(variant_.name, variant_)

    def _check_variant(self, variant: V, key=None, overwrite: bool = False) -> None:
        if not isinstance(variant, self.variant_type):
            raise TypeError(f"'{type(variant)}' does not match '{self.variant_type}'.")

        if key and key != variant.name:
            raise KeyError(f"'{key}' does not match variant name '{variant.name}'.")

        if not overwrite and variant.name in self:
            raise KeyError(
                f"Variant with name '{variant.name}' already exists. Use `overwrite=True` to overwrite."
            )

    def concatenate(self: Self, other: Self) -> Self:
        try:
            self.isequivalent(other, raise_=True)
        except EquivalenceError as e:
            raise ValueError("VariantCollections could not be concatenated") from e

        obj = VariantCollection(self.variant_type)
        for key in self.keys():
            obj.add(self[key].concatenate(other[key]))

        return obj

    def isequivalent(
        self,
        other: Self,
        raise_=False,
    ) -> bool:
        try:
            # fmt: off
            checks = {
                f"Variant types don't match: {self.variant_type}, {other.variant_type if hasattr(other, 'variant_type') else ''}": hasattr(self, "variant_type") and hasattr(other, "variant_type") and self.variant_type == other.variant_type,
                f"VariantCollections do not contain the same variants: {self.keys()=}, {other.keys() if hasattr(other, 'keys') else ''=}": hasattr(self, "keys") and hasattr(other, "keys") and set(self.keys()) == set(other.keys()),
            }
            for key in set(self.keys()) & set(other.keys()):
                checks[f"Variant data ({key}) is not equivalent: {self[key]}, {other[key]}"] = \
                    Variant.isequivalent(self[key], other[key], raise_)
            # fmt: on
        except EquivalenceError:
            if raise_:
                raise
            return False
        except Exception:
            raise EquivalenceError("Unknown error")
        return super().isequivalent(other, raise_, checks)

    def __ne__(self: Self, other: Self) -> bool:
        return not self.__eq__(other)
