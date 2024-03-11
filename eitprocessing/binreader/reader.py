import io
import struct
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T")
N = TypeVar("N", bound=np.number)


@dataclass
class Reader:
    file_handle: io.BufferedReader
    endian: str | None = None

    def read_single(self, type_code: str, cast: type[T]) -> T:
        data = self._read_full_type_code(type_code)
        return cast(data[0])

    def read_list(self, type_code: str, cast: type[T], length: int) -> list[T]:
        full_type_code = f"{length}{type_code}"
        data = self._read_full_type_code(full_type_code)
        return [cast(d) for d in data]

    def read_array(
        self,
        type_code: str,
        cast: type[N],
        length: int,
    ) -> NDArray[N]:
        full_type_code = f"{length}{type_code}"
        data = self._read_full_type_code(full_type_code)
        return np.array(data, dtype=cast)

    def read_string(self, length: int = 1) -> str:
        full_type_code = f"{length}s"
        data = self._read_full_type_code(full_type_code)
        return data[0].decode().rstrip()

    def _read_full_type_code(self, full_type_code: str) -> tuple[Any, ...]:
        if self.endian:
            if self.endian not in ["little", "big"]:
                msg = f"Endian type '{self.endian}' not recognized. Allowed values are 'little' and 'big'."
                raise ValueError(msg)

            prefix = "<" if self.endian == "little" else ">"
            full_type_code = prefix + full_type_code

        data_size = struct.calcsize(full_type_code)
        packed_data = self.file_handle.read(data_size)
        return struct.unpack(full_type_code, packed_data)

    def float32(self) -> float:
        return self.read_single(type_code="f", cast=float)

    def float64(self) -> float:
        return self.read_single(type_code="d", cast=float)

    def npfloat32(self, length: int = 1) -> NDArray[np.float32]:
        return self.read_array(type_code="f", cast=np.float32, length=length)

    def npfloat64(self, length: int = 1) -> NDArray[np.float64]:
        return self.read_array(type_code="d", cast=np.float64, length=length)

    def int32(self) -> int:
        return self.read_single(type_code="i", cast=int)

    def npint32(self, length: int = 1) -> NDArray[np.int32]:
        return self.read_array(type_code="i", cast=np.int32, length=length)

    def string(self, length: int = 1) -> str:
        return self.read_string(length=length)

    def uint8(self) -> int:
        return self.read_single(type_code="B", cast=int)

    def uint16(self) -> int:
        return self.read_single(type_code="H", cast=int)

    def uint32(self) -> int:
        return self.read_single(type_code="I", cast=int)

    def uint64(self) -> int:
        return self.read_single(type_code="Q", cast=int)
