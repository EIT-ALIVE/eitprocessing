from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import get_type_hints
from typing_extensions import Self
from eitprocessing.mixins.equality import Equivalence


@dataclass(eq=False)
class Variant(Equivalence, ABC):
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

    name: str
    label: str
    description: str
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        for attr, type_ in get_type_hints(type(self)).items():
            if not isinstance(getattr(self, attr), type_):
                raise TypeError(
                    f"Invalid type for `{attr}`. "
                    f"Should be {type_}, not {type(getattr(self, attr))}."
                )

    def isequivalent(
        self,
        other: Self,
        raise_=False,
    ) -> bool:
        EDV = "EITDataVariant"
        # fmt: off
        checks = {
            f"{EDV} labels don't match: {self.label}, {other.label}": self.label == other.label,
            f"{EDV} descriptions don't match: {self.description}, {other.description}": self.description == other.description,
            f"{EDV} params don't match: {self.params}, {other.params}": self.params == other.params,
        }
        # fmt: on
        return super().isequivalent(other, raise_, checks)

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
        - EquivalenceError if a and b are not equivalent and can't be merged
        """
