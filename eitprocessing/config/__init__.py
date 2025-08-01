from dataclasses import MISSING, Field, dataclass, fields, replace
from typing import TypeVar, get_type_hints

from frozendict import frozendict
from typing_extensions import Self

T = TypeVar("T")


@dataclass(frozen=True, kw_only=True)
class Config:
    """Base class for configuration."""

    def __post_init__(self):
        for field_ in fields(self):
            if _get_field_type(field_, self.__class__) in (dict, frozendict):
                # Convert dict fields to frozendict for immutability
                default_factory = field_.default_factory
                default_value = default_factory() if default_factory is not MISSING else {}
                merged = default_value | (getattr(self, field_.name) or {})
                object.__setattr__(self, field_.name, frozendict(merged))

    def __replace__(self, /, **changes) -> Self:
        """Return a copy of the of the Config instance replacing attributes.

        Similar to dataclass.replace(), but with special handling of `colorbar_kwargs`. `colorbar_kwargs` is updated
        with the provided dictionary, rather than replaced.

        Args:
            **changes: New values for attributes to replace.

        Returns:
            Self: A new instance with the replaced attributes.
        """
        for field_ in fields(self):
            if _get_field_type(field_, self.__class__) in (dict, frozendict) and field_.name in changes:
                # Instead of replacing the existing with the new dict, merge the changes
                changes[field_.name] = getattr(self, field_.name) | changes[field_.name]

        return replace(self, **changes)

    update = __replace__


def _get_field_type(field_: Field[T], cls: type) -> type[T]:
    # If using future annotations, resolve string to real type
    type_hints = get_type_hints(cls)
    return type_hints[field_.name]
