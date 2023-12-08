from abc import ABC
from abc import abstractmethod
from typing_extensions import Self
from eitprocessing.mixins.equality import Equivalence


class Addition(Equivalence, ABC):
    def __add__(
        self,
        other: Self,
    ) -> Self:
        return self.concatenate(other)

    def concatenate(
        self,
        other: Self,
        label: str | None = None,
    ) -> Self:
        _ = self.isequivalent(other, raise_=True)

        obj = self.__class__()

        return obj
