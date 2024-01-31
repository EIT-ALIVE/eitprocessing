import numpy as np
import warnings
from dataclasses import dataclass, field
from eitprocessing.continuous_data.continuous_data_collection import (
    ContinuousDataCollection,
)
from eitprocessing.sparse_data.sparse_data_collection import SparseDataCollection
from numpy.typing import NDArray
from pathlib import Path
from typing import BinaryIO
from typing_extensions import Self
from . import EITData_
from .eit_data_variant import EITDataVariant
from .vendor import Vendor
from ..binreader.reader import Reader


@dataclass(eq=False)
class SentecEITData(EITData_):
    """Container for EIT data recorded using the Sentec."""  # TODO: which model?

    vendor: Vendor = field(default=Vendor.SENTEC, init=False)
    framerate: float = 50.2

    @classmethod
    def _from_path(  # pylint: disable=too-many-arguments,too-many-locals
        cls,
        path: Path,
        label: str | None = None,
        framerate: float | None = 50.2,
        first_frame: int = 0,
        max_frames: int | None = None,
        return_non_eit_data: bool = False,
    ) -> Self | tuple[Self, ContinuousDataCollection, SparseDataCollection]:
        with open(path, "br") as fh:
            # Find the length of the file
            fh.seek(0, 2)
            file_length = fh.tell()

            # go back to the beginning of the file
            fh.seek(0, 0)

            # instantiate reader
            reader = Reader(fh, endian="little")

            # read the version int8
            version = reader.uint8()

            time = []
            image = []
            index = 0
            first_time = None

            # while there are still data to be read and the number of read data points is higher
            # than the maximum specified, keep reading
            while fh.tell() < file_length and (
                max_frames is None or len(time) < max_frames
            ):
                # Skip timestamp reading
                fh.seek(8, 1)
                # Read DomainId uint8
                domain_id = reader.uint8()
                # read number of data fields uint8
                number_data_fields = reader.uint8()

                for _ in range(number_data_fields):
                    # read data id uint8
                    data_id = reader.uint8()
                    # read payload size ushort
                    payload_size = reader.ushort()

                    if payload_size != 0:
                        # read frame (domain 16 = measurements, data 5 = zero_ref_image)
                        if domain_id == 16:
                            if data_id == 0:
                                time_caption = reader.uint32()

                                # save the first time value and subtract it from each time stamp
                                if not first_time:
                                    first_time = time_caption

                                time_caption -= first_time

                                # convert to seconds and store
                                time.append(time_caption)

                            elif data_id == 5:
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
                                    image.append(ref)
                            else:
                                fh.seek(payload_size, 1)

                        # read the framerate from the file, if present
                        # (domain 64 = configuration, data 5 = framerate)
                        elif domain_id == 64 and data_id == 1:
                            framerate = reader.float32()
                            warnings.warn(
                                f"Framerate value found in file. The framerate value "
                                f"will be set to {framerate}"
                            )

                        else:
                            fh.seek(payload_size, 1)

        n_frames = len(image) if image is not None else 0

        if first_frame > index:
            raise ValueError(
                f"Invalid input: `first_frame` {first_frame} is larger than the "
                f"total number of frames in the file {index}."
            )

        if max_frames and n_frames != max_frames:
            warnings.warn(
                f"The number of frames requested ({max_frames}) is larger "
                f"than the available number ({n_frames}) of frames after "
                f"the first frame selected ({first_frame}, total frames: "
                f"{index}).\n {n_frames} frames will be loaded."
            )

        obj = cls(
            path=path,
            framerate=framerate,
            nframes=n_frames,
            time=np.unwrap(np.array(time), period=np.iinfo(np.uint32).max) / 1000000,
            label=label,
        )
        obj.variants.add(
            EITDataVariant(
                label="raw",
                description="raw impedance data",
                pixel_impedance=np.array(image),
            )
        )

        return obj

    @classmethod
    def _read_frame(  # pylint: disable=too-many-arguments
        cls,
        fh: BinaryIO,
        version: int,
        index: int,
        payload_size: int,
        reader: Reader,
        first_frame: int = 0,
    ) -> NDArray | None:
        """
                Read a single frame in the file. The current position of the file has to be already
                set to the point where the image should be read (data_id 5).
                Args:
                    fh: opened file object
                    version: version of the Sentec file
                    index: current number of read frames
                    payload_size: size of the payload of the data to be read.
                    reader: bites reader object
                    first_frame: index of first time point of sequence

        Returns: A 32 x 32 matrix, containing the pixels values.

        """
        if index < first_frame:
            fh.seek(payload_size, 1)

            return None

        if version > 1:
            # read quality index. We don't use it, so we skip the bytes
            fh.seek(1, 1)
            zero_ref_payload = (payload_size - 3) // 4
        else:
            zero_ref_payload = (payload_size - 2) // 4

        mes_width = reader.uint8()
        mes_height = reader.uint8()
        zero_ref = reader.npfloat32(zero_ref_payload)

        if mes_width * mes_height != len(zero_ref):
            warnings.warn(
                f"The length of image array is "
                f"{len(zero_ref)} which is not equal to the "
                f"product of the width ({mes_width}) and "
                f"height ({mes_height}) of the frame."
                f"Image will not be stored"
            )

            return None

        # the sign of the zero_ref values has to be inverted
        # and the array has to be reshaped into the matrix
        ref_reshape = -np.reshape(zero_ref, (mes_width, mes_height)).T

        return ref_reshape
