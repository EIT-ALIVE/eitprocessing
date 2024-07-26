from __future__ import annotations

import mmap
import os
import warnings
from enum import IntEnum
from functools import partial
from typing import TYPE_CHECKING, BinaryIO

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


def load_from_single_path(  # noqa: C901, PLR0912
    path: Path,
    sample_frequency: float | None = 50.2,
    first_frame: int = 0,
    max_frames: int | None = None,
) -> dict[str, DataCollection]:
    """Load Sentec EIT data from path."""
    with path.open("br") as fo, mmap.mmap(fo.fileno(), length=0, access=mmap.ACCESS_READ) as fh:
        file_length = os.fstat(fo.fileno()).st_size
        reader = BinReader(fh, endian="little")
        version = reader.uint8()

        time: list[float] = []
        max_n_images = int(file_length / 32 / 32 / 4)
        image = np.full(shape=(max_n_images, 32, 32), fill_value=np.nan)
        index = 0
        n_images_added = 0

        # while there are still data to be read and the number of read data points is higher
        # than the maximum specified, keep reading
        while fh.tell() < file_length and (max_frames is None or len(time) < max_frames):
            _ = reader.uint64()  # skip timestamp reading
            domain_id = reader.uint8()
            number_data_fields = reader.uint8()

            for _ in range(number_data_fields):
                data_id = reader.uint8()
                payload_size = reader.uint16()

                if payload_size == 0:
                    continue

                if domain_id == Domain.MEASUREMENT:
                    if data_id == MeasurementDataID.TIMESTAMP:
                        time_of_caption = reader.uint32()
                        time.append(time_of_caption)

                    elif data_id == MeasurementDataID.ZERO_REF_IMAGE:
                        index += 1

                        ref = _read_frame(
                            fh,
                            version,
                            index,
                            payload_size,
                            reader,
                            first_frame,
                        )

                        if ref is not None:
                            image[n_images_added, :, :] = ref
                            n_images_added += 1
                    else:
                        fh.seek(payload_size, os.SEEK_CUR)

                # read the sample frequency from the file, if present
                # (domain 64 = configuration, data 5 = sample frequency)
                elif domain_id == Domain.CONFIGURATION and data_id == ConfigurationDataID.SAMPLE_FREQUENCY:
                    sample_frequency = np.round(reader.float32(), 4)
                    msg = (
                        "Sample frequency value found in file. "
                        f"The sample frequency value will be set to {sample_frequency:.2f}"
                    )
                    warnings.warn(msg)

                else:
                    fh.seek(payload_size, os.SEEK_CUR)
    image = image[:n_images_added, :, :]
    n_frames = len(image) if image is not None else 0

    if (f0 := first_frame) > (fn := index):
        msg = f"Invalid input: `first_frame` ({f0}) is larger than the total number of frames in the file ({fn})."
        raise ValueError(msg)

    if max_frames and n_frames != max_frames:
        msg = (
            f"The number of frames requested ({max_frames}) is larger "
            f"than the available number ({n_frames}) of frames after "
            f"the first frame selected ({first_frame}, total frames: "
            f"{index}).\n {n_frames} frames will be loaded."
        )
        warnings.warn(msg)

    if not sample_frequency:
        sample_frequency = SENTEC_SAMPLE_FREQUENCY

    eitdata_collection = DataCollection(EITData)
    eitdata_collection.add(
        EITData(
            vendor=Vendor.SENTEC,
            path=path,
            sample_frequency=sample_frequency,
            nframes=n_frames,
            time=np.unwrap(np.array(time), period=np.iinfo(np.uint32).max) / 1000000,
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


def _read_frame(
    fh: BinaryIO | mmap.mmap,
    version: int,
    index: int,
    payload_size: int,
    reader: BinReader,
    first_frame: int = 0,
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
    if index < first_frame:
        fh.seek(payload_size, os.SEEK_CUR)
        return None

    if version > 1:
        # read quality index. We don't use it, so we skip the bytes
        _ = reader.uint8()
        n_pixels = (payload_size - 3) // 4
    else:
        n_pixels = (payload_size - 2) // 4

    image_width = reader.uint8()
    image_height = reader.uint8()

    # the sign of the zero_ref values has to be inverted
    zero_ref = -reader.npfloat32(n_pixels)

    if image_width * image_height != n_pixels:
        msg = (
            f"The length of image array is "
            f"{n_pixels} which is not equal to the "
            f"product of the width ({image_width}) and "
            f"height ({image_height}) of the frame."
            f"Image will not be stored."
        )
        raise OSError(msg)

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
