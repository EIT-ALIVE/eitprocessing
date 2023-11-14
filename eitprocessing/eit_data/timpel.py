import warnings
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from eitprocessing.eit_data.eit_data_variant import EITDataVariant
from eitprocessing.eit_data.phases import MaxValue
from eitprocessing.eit_data.phases import MinValue
from eitprocessing.eit_data.phases import QRSMark
from eitprocessing.eit_data.vendor import Vendor
from . import EITData


class TimpelEITData(EITData):
    vendor = Vendor.TIMPEL
    framerate = 50

    @classmethod
    def _from_path(
        cls, path: Path, label: str, framerate: float, first_frame: int, max_frames: int
    ) -> "TimpelEITData":
        ...

    def _load_data(self, first_frame: int):
        """Load data for TIMPEL files."""

        COLUMN_WIDTH = 1030

        try:
            data = np.loadtxt(
                str(self.path),
                dtype=float,
                delimiter=",",
                skiprows=first_frame,
                max_rows=self.nframes,
            )
        except UnicodeDecodeError as e:
            raise OSError(
                f"File {self.path} could not be read as Timpel data.\n"
                "Make sure this is a valid and uncorrupted Timpel data file.\n"
                f"Original error message: {e}"
            ) from e

        data: NDArray
        if data.shape[1] != COLUMN_WIDTH:
            raise OSError(
                f"Input does not have a width of {COLUMN_WIDTH} columns.\n"
                "Make sure this is a valid and uncorrupted Timpel data file."
            )
        if data.shape[0] == 0:
            raise ValueError(
                f"Invalid input: `first_frame` {first_frame} is larger than the "
                f"total number of frames in the file."
            )
        if data.shape[0] != self.nframes:
            if self.nframes:
                warnings.warn(
                    f"The number of frames requested ({self.nframes}) is larger "
                    f"than the available number ({data.shape[0]}) of frames after "
                    f"the first frame selected ({first_frame}).\n"
                    f"{data.shape[0]} frames have been loaded."
                )
            self.nframes = data.shape[0]

        # TODO (#80): QUESTION: check whether below issue was only a Drager problem or also
        # applicable to Timpel.
        # The implemented method seems convoluted: it's easier to create an array
        # with nframes and add a time_offset. However, this results in floating
        # point errors, creating issues with comparing times later on.
        self.time = np.arange(self.nframes + first_frame) / self.framerate
        self.time = self.time[first_frame:]

        pixel_data = data[:, :1024]
        pixel_data = np.reshape(pixel_data, newshape=(-1, 32, 32), order="C")
        pixel_data = np.where(pixel_data == -1000, np.nan, pixel_data)

        # extract waveform data
        waveform_data = {
            "airway_pressure": data[:, 1024],
            "flow": data[:, 1025],
            "volume": data[:, 1026],
        }

        # extract breath start, breath end and QRS marks
        for index in np.flatnonzero(data[:, 1027] == 1):
            self.phases.append(MinValue(index, self.time[index]))

        for index in np.flatnonzero(data[:, 1028] == 1):
            self.phases.append(MaxValue(index, self.time[index]))

        for index in np.flatnonzero(data[:, 1029] == 1):
            self.phases.append(QRSMark(index, self.time[index]))

        self.phases.sort(key=lambda x: x.index)

        self.framesets["raw"] = EITDataVariant(
            name="raw",
            description="raw timpel data",
            params={"framerate": self.framerate},
            pixel_impedance=pixel_data,
            waveform_data=waveform_data,
        )
