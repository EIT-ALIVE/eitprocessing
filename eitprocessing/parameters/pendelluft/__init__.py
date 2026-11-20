from dataclasses import dataclass

from eitprocessing.datahandling.pixelmap import PendelluftMap


@dataclass(frozen=True, kw_only=True)
class PendelluftResult:
    mean_map: PendelluftMap
    mean_value: float
    breath_maps: tuple[PendelluftMap, ...]
    breath_values: tuple[float, ...]
