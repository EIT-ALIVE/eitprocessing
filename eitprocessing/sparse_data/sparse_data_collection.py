from typing_extensions import Self


class SparseDataCollection(dict):
    @classmethod
    def concatenate(cls, a, b) -> Self:
        ...
