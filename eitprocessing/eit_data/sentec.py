from pathlib import Path
from . import EITData


class SentecEITData(EITData):
    @classmethod
    def _from_path(  # pylint: disable=too-many-arguments
        cls, path: Path, label: str, framerate: float, first_frame: int, max_frames: int
    ) -> "SentecEITData":
        ...
