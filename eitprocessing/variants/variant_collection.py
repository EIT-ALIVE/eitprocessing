import contextlib
from typing import Generic
from typing import TypeVar
from typing_extensions import Self
from ..helper import NotEquivalent
from . import Variant


V = TypeVar("V", bound="Variant")


class VariantCollection(dict, Generic[V]):
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

    def __init__(self, variant_type: type[V], *args, **kwargs):
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
            return super().__setitem__(variant_.label, variant_)

    def _check_variant(self, variant: V, key=None, overwrite: bool = False) -> None:
        if not isinstance(variant, self.variant_type):
            raise TypeError(f"'{type(variant)}' does not match '{self.variant_type}'.")

        if key and key != variant.label:
            raise KeyError(f"'{key}' does not match variant name '{variant.label}'.")

        if not overwrite and key in self:
            raise KeyError(
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
        cm = contextlib.nullcontext() if raise_ else contextlib.suppress(NotEquivalent)
        with cm:
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

            return True

        return False
