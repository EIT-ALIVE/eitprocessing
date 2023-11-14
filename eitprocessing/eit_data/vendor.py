from enum import auto
from strenum import LowercaseStrEnum


class Vendor(LowercaseStrEnum):
    """Enum indicating the vendor (manufacturer) of the source EIT device."""

    DRAEGER = auto()
    TIMPEL = auto()
    SENTEC = auto()
    DRAGER = DRAEGER
    DRÄGER = DRAEGER  # pylint: disable = non-ascii-name
