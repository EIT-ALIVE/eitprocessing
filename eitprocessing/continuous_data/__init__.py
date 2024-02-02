from dataclasses import dataclass
from dataclasses import field
import numpy as np
from typing_extensions import Any
from typing_extensions import Self


@dataclass
class ContinuousData:
    label: str
    name: str
    unit: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    loaded: bool = False
    derived_from: Any | list[Any] = field(default_factory=list)
    values: np.ndarray = field(kw_only=True)

    def __post_init__(self):
        if not self.loaded and not self.derived_from:
            raise ValueError("Data must be loaded or calculated form another dataset.")
