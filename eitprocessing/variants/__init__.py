import contextlib
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import TypeVar
from typing_extensions import Self
from ..helper import NotEquivalent


T = TypeVar("T", bound="Variant")


@dataclass
class Variant(ABC):
    """Contains a single variant of a dataset.

    A variant of a dataset is defined as either the raw data, or an edited
    version of that raw data. For example, EIT data can contain a "raw"
    versions and a "filtered" version. Both variants share the same time axis.

    The actual data of a variant is contained in a variable that must be set by
    a subclass inheriting from this class.

    Attributes:
    - label (str): a short descriptor for the variant, that is used to access
      the variant
    - description (str): a longer description of the variant
    - params (dict): contains information on how to reproduce the variant, e.g.
      which filters and filters settigs were used
    """

    label: str
    description: str
    params: dict = field(default_factory=dict)

    def check_equivalence(self: T, other: T, raise_=False) -> bool:
        """Check the equivalence of two variants

        For two variants to be equivalent, they need to have the same class,
        the same label, the same description and the same parameters. Only the
        actual data can differ between variants.

        Args:
        - a (Variant)
        - b (Variant)

        Raises:
        - NotEquivalent (only if raise_ is `True`) when a and b are not
          equivalent on one of the attributes
        """
        cm = contextlib.nullcontext() if raise_ else contextlib.suppress(NotEquivalent)
        with cm:
            if not isinstance(self, other.__class__):
                raise NotEquivalent(
                    f"Variant classes don't match: {self.__class__}, {other.__class__}"
                )

            if (a_ := self.label) != (b_ := other.label):
                raise NotEquivalent(f"EITDataVariant names don't match: {a_}, {b_}")

            if (a_ := self.description) != (b_ := other.description):
                raise NotEquivalent(
                    f"EITDataVariant descriptions don't match: {a_}, {b_}"
                )

            if (a_ := self.params) != (b_ := other.params):
                raise NotEquivalent(f"EITDataVariant params don't match: {a_}, {b_}")

            return True

        return False

    @abstractmethod
    def concatenate(self: Self, other: Self) -> Self:
        """Concatenates two variants

        Concatenating two variants results in a single variant with the
        combined length of both variants.

        To merge more than two variants, use
        `functools.reduce(Variant.concatenate, list_of_variants)`.

        Args:
        - self (Variant)
        - other (Variant)

        Raises:
        - NotEquivalent if a and b are not equivalent and can't be merged
        """
