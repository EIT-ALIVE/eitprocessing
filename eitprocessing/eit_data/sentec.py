from pathlib import Path
from typing_extensions import Self
from eitprocessing.continuous_data.continuous_data_collection import (
    ContinuousDataCollection,
)
from eitprocessing.sparse_data.sparse_data_collection import SparseDataCollection
from . import EITData_


class SentecEITData(EITData_):
    @classmethod
    def _from_path(  # pylint: disable=too-many-arguments
        cls,
        path: Path,
        label: str | None = None,
        framerate: float | None = 50.2,
        first_frame: int = 0,
        max_frames: int | None = None,
        return_non_eit_data: bool = False,
    ) -> Self | tuple[Self, ContinuousDataCollection, SparseDataCollection]:
        ...
