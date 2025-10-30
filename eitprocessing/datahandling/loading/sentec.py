from __future__ import annotations

import mmap
import os
import warnings
from enum import IntEnum
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.datacollection import DataCollection
from eitprocessing.datahandling.eitdata import EITData, Vendor
from eitprocessing.datahandling.intervaldata import IntervalData
from eitprocessing.datahandling.loading import load_eit_data
from eitprocessing.datahandling.loading.binreader import BinReader
from eitprocessing.datahandling.sparsedata import SparseData

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

SENTEC_SAMPLE_FREQUENCY = 50.2

load_sentec_data = partial(load_eit_data, vendor=Vendor.SENTEC)


def load_from_single_path(
    path: Path,
    sample_frequency: float | None = None,
    first_frame: int = 0,
    max_frames: int | None = None,
) -> dict[str, DataCollection]:
    """Load Sentec EIT data from path."""
    time: list[float] = []
    index = 0
    with path.open("br") as fo, mmap.mmap(fo.fileno(), length=0, access=mmap.ACCESS_READ) as fh:
        file_length = os.fstat(fo.fileno()).st_size
        reader = BinReader(fh, endian="little")
        version = reader.uint8()

        max_n_images = int(file_length / 32 / 32 / 4)
        if max_frames is not None:
            max_n_images = min(max_n_images, max_frames)

        image = np.full(shape=(max_n_images, 32, 32), fill_value=np.nan)

        # while there are still data to be read and the number of read data points is higher
        # than the maximum specified, keep reading
        while fh.tell() < file_length and (len(time) < max_n_images):
            _ = reader.uint64()  # skip timestamp reading
            domain_id = reader.uint8()
            number_data_fields = reader.uint8()

            for _ in range(number_data_fields):
                index, sample_frequency = _read_data_field(
                    reader, domain_id, index, first_frame, fh, time, version, image, sample_frequency
                )

    if first_frame >= index:
        msg = f"`first_frame` ({first_frame}) is larger than or equal to the number of frames in the file ({index})."
        raise ValueError(msg)

    image = image[first_frame:index, :, :]
    n_frames = len(image)

    if max_frames and n_frames != max_frames:
        msg = (
            f"The number of frames requested ({max_frames}) is larger "
            f"than the available number ({n_frames}) of frames after "
            f"the first frame selected ({first_frame}, total frames: "
            f"{index}).\n {n_frames} frames were loaded."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    if sample_frequency is None:
        sample_frequency = SENTEC_SAMPLE_FREQUENCY

    time_array = np.unwrap(np.array(time), period=np.iinfo(np.uint32).max) / 1_000_000

    eitdata_collection = DataCollection(EITData)
    eitdata_collection.add(
        EITData(
            vendor=Vendor.SENTEC,
            path=path,
            sample_frequency=sample_frequency,
            nframes=n_frames,
            time=time_array,
            label="raw",
            pixel_impedance=image,
        ),
    )

    return {
        "eitdata_collection": eitdata_collection,
        "continuousdata_collection": DataCollection(ContinuousData),
        "sparsedata_collection": DataCollection(SparseData),
        "intervaldata_collection": DataCollection(IntervalData),
    }


def _read_data_field(
    reader: BinReader,
    domain_id: int,
    index: int,
    first_frame: int,
    fh: mmap.mmap,
    time: list[float],
    version: int,
    image: np.ndarray,
    sample_frequency: float | None,
) -> tuple[int, float | None]:
    """Reads the specified data field from file, and returns an updated index and sample frequency."""
    data_id = reader.uint8()
    payload_size = reader.uint16()

    if payload_size == 0:
        return index, sample_frequency

    match domain_id, data_id:
        case Domain.MEASUREMENT, MeasurementDataID.TIMESTAMP:
            if index < first_frame:
                fh.seek(payload_size, os.SEEK_CUR)
            else:
                time.append(reader.uint32())

        case Domain.MEASUREMENT, MeasurementDataID.ZERO_REF_IMAGE:
            if index < first_frame:
                fh.seek(payload_size, os.SEEK_CUR)
            else:
                ref = _read_frame(version=version, payload_size=payload_size, reader=reader)
                image[index, :, :] = ref

            index += 1

        case Domain.CONFIGURATION, ConfigurationDataID.SAMPLE_FREQUENCY:
            # read the sample frequency from the file, if present
            # (domain 64 = configuration, data 5 = sample frequency)

            loaded_sample_frequency = np.round(reader.float32(), 4)
            if sample_frequency and not np.isclose(loaded_sample_frequency, sample_frequency):
                msg = (
                    f"Sample frequency provided ({sample_frequency:.2f} Hz) "
                    f"differs from value found in file "
                    f"({loaded_sample_frequency:.2f} Hz). "
                    f"The sample frequency value found in the file will be used."
                )
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
            sample_frequency = loaded_sample_frequency

        case _, _:
            fh.seek(payload_size, os.SEEK_CUR)

    return index, sample_frequency


def _read_frame(
    version: int,
    payload_size: int,
    reader: BinReader,
) -> NDArray | None:
    """Read a single frame in the file.

    The current position of the file has to be already set to the point where the image should be read (data_id 5).

    Args:
                fh: opened file object
                version: version of the Sentec file
                index: current number of read frames
                payload_size: size of the payload of the data to be read.
                reader: bites reader object
                first_frame: index of first time point of sequence.

    Returns: A 32 x 32 matrix, containing the pixels values.

    """
    if version > 1:
        # read quality index. We don't use it, so we skip the bytes
        _ = reader.uint8()
        n_pixels = (payload_size - 3) // 4
    else:
        n_pixels = (payload_size - 2) // 4

    image_width = reader.uint8()
    image_height = reader.uint8()

    if image_width * image_height != n_pixels:
        msg = (
            f"The length of image array is {n_pixels} which is not equal to "
            f"the product of the width ({image_width}) and height "
            f"({image_height}) of the frame."
        )
        raise OSError(msg)

    # the sign of the zero_ref values has to be inverted
    zero_ref = -reader.npfloat32(n_pixels)
    return np.reshape(zero_ref, (image_width, image_height), order="C")


class Domain(IntEnum):
    """Domain loaded data falls in."""

    MEASUREMENT = 16
    CONFIGURATION = 64


class MeasurementDataID(IntEnum):
    """ID of measured data."""

    TIMESTAMP = 0
    ZERO_REF_IMAGE = 5


class ConfigurationDataID(IntEnum):
    """ID of configuration data."""

    SAMPLE_FREQUENCY = 1
