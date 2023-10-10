import warnings
import numpy as np
import pytest
from eitprocessing.roi_selection.gridselection import GridSelection
from eitprocessing.roi_selection.gridselection import InvalidDivision
from eitprocessing.roi_selection.gridselection import InvalidHorizontalDivision
from eitprocessing.roi_selection.gridselection import InvalidVerticalDivision
from eitprocessing.roi_selection.gridselection import MoreHorizontalGroupsThanColumns
from eitprocessing.roi_selection.gridselection import MoreVerticalGroupsThanRows
from eitprocessing.roi_selection.gridselection import UnevenHorizontalDivision
from eitprocessing.roi_selection.gridselection import UnevenVerticalDivision


def matrices_from_string(string: str) -> list[np.ndarray]:
    """Generates a list of matrices from a string containing a representation of matrices.

    A represtation of matrices contains one character per cell that describes the value of that cell.
    Rows are delimited by commas. Matrices are delimited by semi-colons.

    The following characters are transformed to these corresponding values:
    - T / 1 -> 1
    - F / 0 -> 0
    - R -> np.random.int(2, 100)
    - N / any other character -> np.nan

    Examples:
    >>> matrices_from_string("1T1,FNR")
    [array([[ 1.,  1.,  1.],
            [ 0., nan, 40.]])]
    >>> matrices_from_string("1T1,FNR")
    [array([[ 1.,  1.,  1.],
            [ 0., nan,  37]])]
    >>> matrices_from_string("RR,RR;RRR;1R")
    [array([[21., 80.],
            [43., 10.]]),
    array([[26., 43., 62.]]),
    array([[ 1., 89.]])]
    """

    matrices = []
    for part in string.split(";"):
        str_matrix = np.array([tuple(row) for row in part.split(",")], dtype="object")
        matrix = np.full(str_matrix.shape, np.nan, dtype=np.floating)
        matrix[np.nonzero(str_matrix == "1")] = 1
        matrix[np.nonzero(str_matrix == "T")] = 1
        matrix[np.nonzero(str_matrix == "0")] = 0
        matrix[np.nonzero(str_matrix == "F")] = 0
        matrix = np.where(
            str_matrix == "R",
            np.random.default_rng().integers(2, 100, matrix.shape),
            matrix,
        )
        matrices.append(matrix)

    return matrices


@pytest.mark.parametrize(
    "v_split,h_split,split_pixels,exception_type",
    [
        (1, 1, False, None),
        (0, 1, False, InvalidVerticalDivision),
        (-1, 1, False, InvalidVerticalDivision),
        (1.1, 1, False, TypeError),
        (1, 0, False, InvalidHorizontalDivision),
        (1, -1, False, InvalidHorizontalDivision),
        (1, 1.1, False, TypeError),
        (2, 2, "not a boolean", TypeError),
        (2, 2, 1, TypeError),
        (2, 2, 0, TypeError),
    ],
)
def test_initialisation(v_split, h_split, split_pixels, exception_type):
    if exception_type is None:
        GridSelection(
            v_split, h_split, split_columns=split_pixels, split_rows=split_pixels
        )

    else:
        with pytest.raises(exception_type):
            GridSelection(
                v_split, h_split, split_columns=split_pixels, split_rows=split_pixels
            )


@pytest.mark.parametrize(
    "shape,split_vh,result_string",
    [
        ((2, 1), (2, 1), "T,F;F,T"),
        ((3, 1), (3, 1), "T,F,F;F,T,F;F,F,T"),
        ((1, 3), (1, 3), "TFF;FTF;FFT"),
        ((1, 3), (1, 2), "TTF;FFT"),
        (
            (4, 4),
            (2, 2),
            "TTFF,TTFF,FFFF,FFFF;"
            "FFTT,FFTT,FFFF,FFFF;"
            "FFFF,FFFF,TTFF,TTFF;"
            "FFFF,FFFF,FFTT,FFTT",
        ),
        (
            (5, 5),
            (2, 2),
            "TTTFF,TTTFF,TTTFF,FFFFF,FFFFF;"
            "FFFTT,FFFTT,FFFTT,FFFFF,FFFFF;"
            "FFFFF,FFFFF,FFFFF,TTTFF,TTTFF;"
            "FFFFF,FFFFF,FFFFF,FFFTT,FFFTT",
        ),
        ((5, 2), (1, 2), "TF,TF,TF,TF,TF;FT,FT,FT,FT,FT"),
        ((1, 9), (1, 6), "TTFFFFFFF;FFTFFFFFF;FFFTTFFFF;FFFFFTFFF;FFFFFFTTF;FFFFFFFFT"),
        ((2, 2), (1, 1), "TT,TT"),
    ],
)
def test_no_split_pixels_no_nans(shape, split_vh, result_string):
    data = np.random.default_rng().integers(1, 100, shape)
    result_matrices = matrices_from_string(result_string)

    gs = GridSelection(*split_vh)
    matrices = gs.find_grid(data)

    num_appearances = np.sum(np.stack(matrices, axis=-1), axis=-1)

    assert len(matrices) == np.prod(split_vh)
    assert np.array_equal(num_appearances, (~np.isnan(data) * 1))
    assert np.array_equal(matrices, result_matrices)


@pytest.mark.parametrize(
    "shape,split_vh,result",
    [
        (
            (2, 3),
            (1, 2),
            [[[1.0, 0.5, 0], [1.0, 0.5, 0]], [[0, 0.5, 1.0], [0, 0.5, 1.0]]],
        ),
        (
            (3, 2),
            (2, 1),
            [[[1.0, 1.0], [0.5, 0.5], [0, 0]], [[0, 0], [0.5, 0.5], [1, 1]]],
        ),
        (
            (1, 4),
            (1, 3),
            [
                [[1.0, 1 / 3, 0.0, 0.0]],
                [[0.0, 2 / 3, 2 / 3, 0.0]],
                [[0.0, 0.0, 1 / 3, 1.0]],
            ],
        ),
    ],
)
def test_split_pixels_no_nans(shape, split_vh, result):
    data = np.random.default_rng().integers(1, 100, shape)
    expected_result = [np.array(r) for r in result]

    gs = GridSelection(*split_vh, split_rows=True, split_columns=True)
    actual_result = gs.find_grid(data)

    num_appearances = np.sum(np.stack(actual_result, axis=-1), axis=-1)

    assert len(actual_result) == np.prod(split_vh)
    assert np.array_equal(num_appearances, (~np.isnan(data) * 1))

    # Ideally, we'd use np.array_equal() here, but due to floating point arithmetic, they values
    # are off by an insignificant amount.
    assert np.allclose(actual_result, expected_result)


@pytest.mark.parametrize(
    "data_string,split_vh,result_string",
    [
        ("NNN,NRR,NRR", (1, 1), "NNN,NTT,NTT"),
        (
            "NNN,RRR,RRR,RRR,RRR",
            (2, 2),
            "NNN,TTF,TTF,FFF,FFF;"
            "NNN,FFT,FFT,FFF,FFF;"
            "NNN,FFF,FFF,TTF,TTF;"
            "NNN,FFF,FFF,FFT,FFT",
        ),
        (
            "NNNNNN,NNNNNN,NRRRRR,RNRRRR,NNNNNN",
            (2, 2),
            "NNNNNN,NNNNNN,NTTFFF,FNFFFF,NNNNNN;"
            "NNNNNN,NNNNNN,NFFTTT,FNFFFF,NNNNNN;"
            "NNNNNN,NNNNNN,NFFFFF,TNTFFF,NNNNNN;"
            "NNNNNN,NNNNNN,NFFFFF,FNFTTT,NNNNNN",
        ),
    ],
)
def test_no_split_pixels_nans(data_string, split_vh, result_string):
    data = matrices_from_string(data_string)[0]
    numeric_values = np.ones(data.shape)
    numeric_values[np.isnan(data)] = np.nan
    result = matrices_from_string(result_string)

    v_split, h_split = split_vh
    gs = GridSelection(v_split, h_split, split_rows=False, split_columns=False)

    matrices = gs.find_grid(data)
    num_appearances = np.sum(np.stack(matrices, axis=-1), axis=-1)

    assert len(matrices) == h_split * v_split
    assert np.array_equal(num_appearances, numeric_values, equal_nan=True)
    assert np.array_equal(matrices, result, equal_nan=True)


@pytest.mark.parametrize(
    "data_string,split_vh,split_rows,split_columns,warning_type",
    [
        ("RR,RR", (2, 2), False, False, None),
        ("RRR,RRR", (2, 2), False, False, UnevenHorizontalDivision),
        ("RRR,RRR", (1, 3), False, False, None),
        ("RRRR,RRRR", (1, 3), False, False, UnevenHorizontalDivision),
        ("RR,RR,RR", (2, 1), False, False, UnevenVerticalDivision),
        ("RR,RR,RR", (3, 1), False, False, None),
        ("NN,RR,RR", (2, 1), False, False, None),
        ("R", (2, 1), True, True, MoreVerticalGroupsThanRows),
        ("R", (1, 2), True, True, MoreHorizontalGroupsThanColumns),
        ("RRR,RRR,RRR", (4, 3), True, True, MoreVerticalGroupsThanRows),
    ],
)
def test_warnings(data_string, split_vh, split_rows, split_columns, warning_type):
    data = matrices_from_string(data_string)[0]
    gs = GridSelection(*split_vh, split_rows=split_rows, split_columns=split_columns)

    if warning_type is None:
        # catch all warnings and raises them
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            gs.find_grid(data)
    else:
        with pytest.warns(warning_type):
            gs.find_grid(data)


@pytest.mark.parametrize(
    "data_string,split_vh,exception_type",
    [
        ("RR,RR", (2, 2), None),
        ("RR,RR", (3, 1), InvalidVerticalDivision),
        ("RR,RR", (1, 3), InvalidHorizontalDivision),
        ("RR,RR", (3, 1), InvalidDivision),
        ("RR,RR", (1, 3), InvalidDivision),
    ],
)
def test_exceptions(data_string, split_vh, exception_type):
    data = matrices_from_string(data_string)[0]
    gs = GridSelection(*split_vh)

    if exception_type is None:
        gs.find_grid(data)

    else:
        with pytest.raises(exception_type):
            gs.find_grid(data)


@pytest.mark.parametrize(
    "split_vh,result",
    [
        ((1, 1), [[0]]),
        ((1, 2), [[0, 1]]),
        ((2, 1), [[0], [1]]),
        ((2, 2), [[0, 1], [2, 3]]),
        ((3, 4), [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]),
    ],
)
def test_matrix_layout(split_vh, result):
    gs = GridSelection(*split_vh)
    layout = gs.matrix_layout()

    assert np.array_equal(layout, np.array(result))
