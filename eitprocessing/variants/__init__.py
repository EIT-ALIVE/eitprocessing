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
    name: str
    description: str
    params: dict = field(default_factory=dict)

    @staticmethod
    def check_equivalence(a: T, b: T, raise_=False) -> bool:
        try:
            if not isinstance(a, b.__class__):
                raise NotEquivalent(
                    f"Variant classes don't match: {a.__class__}, {b.__class__}"
                )

            if (a_ := a.name) != (b_ := b.name):
                raise NotEquivalent(f"EITDataVariant names don't match: {a_}, {b_}")

            if (a_ := a.description) != (b_ := b.description):
                raise NotEquivalent(
                    f"EITDataVariant descriptions don't match: {a_}, {b_}"
                )

            if (a_ := a.params) != (b_ := b.params):
                raise NotEquivalent(f"EITDataVariant params don't match: {a_}, {b_}")
        except NotEquivalent:
            if raise_:
                raise
            return False

        return True

    @classmethod
    @abstractmethod
    def concatenate(cls, a: Self, b: Self) -> Self:
        ...
