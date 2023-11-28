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
            # read the version int8
            version = int.from_bytes(fh.read(1), "little")

            if version < 2:
                warnings.warn("Old file version")

            time = []
            image = None
            index = -1

            while fh.tell() < file_length:

                # if the number of read data points is higher than the maximum specified, end
                if max_frames and len(time) >= max_frames:
                    break

                # Read time stamp uint64
                timestamp = struct.unpack('<Q', fh.read(8))[0]
                # Read DomainId uint8
                domain_id = int.from_bytes(fh.read(1), "little")
                # read number of data fields uint8
                number_data_fields = int.from_bytes(fh.read(1), "little")

                for data_field in range(number_data_fields):
                    # read data id uint8
                    data_id = int.from_bytes(fh.read(1), "little")
                    # read payload size ushort
                    payload_size_bytes = fh.read(2)
                    payload_size = struct.unpack('<H', payload_size_bytes)[0]

                    if payload_size != 0:
                        # read measurements data
                        if domain_id == 16:
                            if data_id == 5:
                                index += 1

                                if index < first_frame:
                                    break

                                # read quality index. We don't use it, so we skip the bytes
                                fh.seek(1, 1)

                                mes_width = struct.unpack('B', fh.read(1))[0]
                                mes_height = struct.unpack('B', fh.read(1))[0]
                                zero_ref = struct.unpack(f'{(payload_size - 3) // 4}f', fh.read(
                                    payload_size - 3))

                                if mes_width * mes_height != len(zero_ref):
                                    print('wrong size, not storing zeroRef, please check file')
                                    return
                                else:
                                    # the sign of the zero_ref values has to be inverted
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
                                framerate = struct.unpack('<f', fh.read(payload_size))[0]
                            else:
                                fh.seek(payload_size, 1)
                        else:
                            fh.seek(payload_size, 1)

        n_frames = len(image)

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
