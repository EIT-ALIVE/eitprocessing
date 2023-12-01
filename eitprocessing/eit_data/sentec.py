import numpy as np
import struct
import warnings
from dataclasses import dataclass, field
from eitprocessing.continuous_data.continuous_data_collection import ContinuousDataCollection
from eitprocessing.sparse_data.sparse_data_collection import SparseDataCollection
from pathlib import Path
from typing_extensions import Self
from . import EITData_
from .eit_data_variant import EITDataVariant
from .vendor import Vendor
from ..binreader.reader import Reader
from ..variants.variant_collection import VariantCollection


@dataclass(eq=False)
class SentecEITData(EITData_):
    """Container for EIT data recorded using the Sentec."""  # TODO: which model?

    vendor: Vendor = field(default=Vendor.SENTEC, init=False)
    framerate: float = 50.2
    variants: VariantCollection = field(
        default_factory=lambda: VariantCollection(EITDataVariant)
    )

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

        with open(path, "br") as fh:
            # Find the length of the file
            fh.seek(0, 2)
            file_length = fh.tell()

            # go back to the beginning of the file
            fh.seek(0, 0)

            # instantiate reader
            reader = Reader(fh)

            # read the version int8
            version = reader.unsigned_char('little')

            if version < 2:
                warnings.warn(f'File version {version}. Version 2 or higher expected.')

            time = []
            image = None
            index = 0

            while fh.tell() < file_length:

                # if the number of read data points is higher than the maximum specified, end
                if max_frames and len(time) >= max_frames:
                    break

                # Read time stamp uint64
                timestamp = reader.unsigned_long_long('little')
                # Read DomainId uint8
                domain_id = reader.unsigned_char('little')
                # read number of data fields uint8
                number_data_fields = reader.unsigned_char('little')

                for data_field in range(number_data_fields):
                    # read data id uint8
                    data_id = reader.unsigned_char('little')
                    # read payload size ushort
                    payload_size = reader.unsigned_short('little')

                    if payload_size != 0:
                        # read measurements data
                        if domain_id == 16:
                            if data_id == 5:
                                index += 1

                                if index < first_frame:
                                    fh.seek(payload_size, 1)
                                else:
                                    # read quality index. We don't use it, so we skip the bytes
                                    fh.seek(1, 1)

                                    mes_width = reader.unsigned_char('little')
                                    mes_height = reader.unsigned_char('little')
                                    zero_ref = reader.npfloat32((payload_size - 3) // 4, 'little')

                                    if mes_width * mes_height != len(zero_ref):
                                        warnings.warn(f'The length of image array is '
                                                      f'{len(zero_ref)} which is not equal to the '
                                                      f'product of the width ({mes_width}) and '
                                                      f'height ({mes_height}) of the frame')
                                        return
                                    else:
                                        # the sign of the zero_ref values has to be inverted
                                        # and the array has to be reshaped into the matrix
                                        ref_reshape = -np.reshape(zero_ref, (mes_width, mes_height))
                                        if image is not None:
                                            image = np.concatenate(
                                                [image, ref_reshape[np.newaxis, :, :]], axis=-0)
                                        else:
                                            image = ref_reshape[np.newaxis, :, :]

                                    time.append(timestamp)

                            else:
                                fh.seek(payload_size, 1)
                        # read configuration data
                        elif domain_id == 64:
                            if data_id == 1:
                                # read the framerate from the file, if present
                                framerate = reader.float32('little')
                            else:
                                fh.seek(payload_size, 1)
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
            time=np.array(time),
            label=label
        )
        obj.variants.add(
            EITDataVariant(
                label="raw",
                description="raw impedance data",
                pixel_impedance=image,
            )
        )

        return obj
