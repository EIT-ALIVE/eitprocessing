# OLD FILE. TESTS NOT YET FUNCTIONAL
# TODO: remove line below to activate linting
# ruff: noqa

import bisect
import os
from dataclasses import dataclass, is_dataclass
from pprint import pprint
from pathlib import Path
import numpy as np
import pytest
from typing_extensions import Self

from eitprocessing.datahandling.sequence import Sequence
from eitprocessing.datahandling.loading import load_eit_data
from eitprocessing.datahandling.eitdata import EITData, Vendor
from eitprocessing.datahandling.mixins.slicing import SelectByIndex

environment = os.environ.get(
    "EIT_PROCESSING_TEST_DATA",
    Path.resolve(Path(__file__).parent.parent),
)
data_directory = Path(environment) / "tests" / "test_data"
draeger_file1 = Path(data_directory) / "Draeger_Test3.bin"
draeger_file2 = Path(data_directory) / "Draeger_Test.bin"
timpel_file = Path(data_directory) / "Timpel_Test.txt"
dummy_file = Path(data_directory) / "not_a_file.dummy"


def test_eq():
    data = load_eit_data(draeger_file1, vendor="draeger")
    data2 = load_eit_data(draeger_file1, vendor="draeger")

    data.isequivalent(data2)
