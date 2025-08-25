from typing import Generic, TypeVar

T = TypeVar("T", int, str)


class A(Generic[T]):
    """test."""

    test: T = 3
