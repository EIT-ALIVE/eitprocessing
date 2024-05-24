import io
import struct
from dataclasses import dataclass
from mmap import mmap
from typing import Any, Literal, TypeVar

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T")
N = TypeVar("N", bound=np.number)


@dataclass
class BinReader:
    """Helper class for reading binary files from disk.

    Args:
        file_handle: a buffered reader handle, e.g. the result of the `open()` function.
        endian: the endianness of the binary data. Either 'little' or 'big', or None.
    """

    file_handle: io.BufferedReader | mmap
    endian: Literal["little", "big"] | None = None

    def read_single(self, type_code: str, cast: type[T]) -> T:
        """Read and return a single unit of the given type code.

        The type of data to be read should be provided as a single typ code. See
        https://docs.python.org/3.10/library/struct.html#byte-order-size-and-alignment for a list of available type
        codes.

        A unit returns a single value, and can be one or more bytes of data. E.g. requesting a signed 32-bit integer
        ('q') will result in reading 8 bytes of data.

        `cast` should be a type, e.g. `int` or `float` used to cast the value to the proper type.

        Args:
            type_code: singular type code.
            cast: the associated type.
        """
        data = self._read_full_type_code(type_code)
        return cast(data[0])

    def read_list(self, type_code: str, cast: type[T], length: int) -> list[T]:
        """Read multiple values of the same type and return as list.

        See `read_single()`.

        Args:
            type_code: singular type code.
            cast: the associated type.
            length: number of values to be read.
        """
        full_type_code = f"{length}{type_code}"
        data = self._read_full_type_code(full_type_code)
        return [cast(d) for d in data]

    def read_array(
        self,
        type_code: str,
        cast: type[N],
        length: int,
    ) -> NDArray[N]:
        """Read multiple values of the same type and return as NumPy array.

        See `read_list()`.
        """
        full_type_code = f"{length}{type_code}"
        data = self._read_full_type_code(full_type_code)
        return np.array(data, dtype=cast)

    def read_string(self, length: int = 1) -> str:
        """Read and return a string with a given length.

        Reads `length` characters of type code 's' and returns as a string. When length is not provided, a single
        character is returned.

        Args:
            length: number of characters.
        """
        full_type_code = f"{length}s"
        data = self._read_full_type_code(full_type_code)
        return data[0].decode().rstrip()

    string = read_string

    def _read_full_type_code(self, full_type_code: str) -> tuple[Any, ...]:
        """Read the data associated with the type code."""
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
        """Read and return a single signed 32-bit floating point value."""
        return self.read_single(type_code="f", cast=float)

    def float64(self) -> float:
        """Read and return a single signed 64-bit floating point value."""
        return self.read_single(type_code="d", cast=float)

    def npfloat32(self, length: int = 1) -> NDArray[np.float32]:
        """Read and return an array of signed 32-bit floating point values."""
        return self.read_array(type_code="f", cast=np.float32, length=length)

    def npfloat64(self, length: int = 1) -> NDArray[np.float64]:
        """Read and return an array of signed 64-bit floating point values."""
        return self.read_array(type_code="d", cast=np.float64, length=length)

    def int32(self) -> int:
        """Read and return a single signed 32-bit integer value."""
        return self.read_single(type_code="i", cast=int)

    def npint32(self, length: int = 1) -> NDArray[np.int32]:
        """Read and return an array of signed 32-bit integer values."""
        return self.read_array(type_code="i", cast=np.int32, length=length)

    def uint8(self) -> int:
        """Read and return a single unsigned 8-bit integer value."""
        return self.read_single(type_code="B", cast=int)

    def uint16(self) -> int:
        """Read and return a single unsigned 16-bit integer value."""
        return self.read_single(type_code="H", cast=int)

    def uint32(self) -> int:
        """Read and return a single unsigned 32-bit integer value."""
        return self.read_single(type_code="I", cast=int)

    def uint64(self) -> int:
        """Read and return a single unsigned 64-bit integer value."""
        return self.read_single(type_code="Q", cast=int)
