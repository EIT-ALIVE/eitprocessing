from __future__ import annotations

import mmap
import os
import warnings
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, BinaryIO

import numpy as np

from eitprocessing.binreader.reader import Reader
from eitprocessing.continuous_data import ContinuousData
from eitprocessing.data_collection import DataCollection
from eitprocessing.sparse_data import SparseData

from . import EITData_
from .vendor import Vendor

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


@dataclass(eq=False)
class SentecEITData(EITData_):
    """Container for EIT data recorded using the Sentec."""  # TODO: which model?

    vendor: Vendor = field(default=Vendor.SENTEC, init=False)
    framerate: float = 50.2

    @classmethod
    def _from_path(  # noqa: C901
        cls,
        path: Path,
        framerate: float | None = 50.2,
        first_frame: int = 0,
        max_frames: int | None = None,
        return_non_eit_data: bool = False,
    ) -> DataCollection | tuple[DataCollection, DataCollection, DataCollection]:
        with path.open("br") as fo, mmap.mmap(fo.fileno(), length=0, access=mmap.ACCESS_READ) as fh:
            file_length = os.fstat(fo.fileno()).st_size
            reader = Reader(fh, endian="little")
            version = reader.uint8()

            time = []
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

                            ref = cls._read_frame(
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

                    # read the framerate from the file, if present
                    # (domain 64 = configuration, data 5 = framerate)
                    elif domain_id == Domain.CONFIGURATION and data_id == ConfigurationDataID.FRAMERATE:
                        framerate = reader.float32()
                        msg = f"Framerate value found in file. The framerate value will be set to {framerate:.2f}"
                        warnings.warn(msg)

                    else:
                        fh.seek(payload_size, os.SEEK_CUR)
        image = image[:n_images_added, :, :]
        n_frames = len(image) if image is not None else 0

        if first_frame > index:
            msg = (
                f"Invalid input: `first_frame` {first_frame} is larger than the "
                f"total number of frames in the file {index}.",
            )
            raise ValueError(msg)

        if max_frames and n_frames != max_frames:
            msg = (
                f"The number of frames requested ({max_frames}) is larger "
                f"than the available number ({n_frames}) of frames after "
                f"the first frame selected ({first_frame}, total frames: "
                f"{index}).\n {n_frames} frames will be loaded."
            )
            warnings.warn(msg)

        if not framerate:
            framerate = cls.framerate

        eit_data_collection = DataCollection(cls)
        eit_data_collection.add(
            cls(
                path=path,
                framerate=framerate,
                nframes=n_frames,
                time=np.unwrap(np.array(time), period=np.iinfo(np.uint32).max) / 1000000,
                label="raw",
                pixel_impedance=image,
            ),
        )

        if return_non_eit_data:
            return eit_data_collection, DataCollection(ContinuousData), DataCollection(SparseData)
        return eit_data_collection

    @classmethod
    def _read_frame(
        cls,
        fh: BinaryIO,
        version: int,
        index: int,
        payload_size: int,
        reader: Reader,
        first_frame: int = 0,
    ) -> NDArray | None:
        """
        Read a single frame in the file.

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
            )
            raise OSError(msg)

        return np.reshape(zero_ref, (image_width, image_height), order="C")


class Domain(IntEnum):
    MEASUREMENT = 16
    CONFIGURATION = 64


class MeasurementDataID(IntEnum):
    TIMESTAMP = 0
    ZERO_REF_IMAGE = 5


class ConfigurationDataID(IntEnum):
    FRAMERATE = 1
