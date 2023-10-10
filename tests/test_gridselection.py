import warnings
import numpy as np
import pytest
from eitprocessing.roi_selection.gridselection import GridSelection
from eitprocessing.roi_selection.gridselection import InvalidDivision
from eitprocessing.roi_selection.gridselection import InvalidHorizontalDivision
from eitprocessing.roi_selection.gridselection import InvalidVerticalDivision


def matrices_from_string(string: str, boolean: bool = False) -> list[np.ndarray]:
    """Generates a list of matrices from a string containing a matrix representation.

    A matrix represtation contains one character per cell that describes the value of that cell.
    Rows are delimited by commas. Matrices are delimited by semi-colons.

    The returned matrices by default have `np.floating` as dtype. When `boolean` is set to True, the
    dtype is `bool`. That means that the actual values in the matrices depend on the value of `boolean`.

    The following characters are transformed to these corresponding values in either `np.floating` or
    `bool` mode:
    - T -> 1. or True
    - F -> 0. or False
    - 1 -> 1. or True
    - N -> np.nan or False
    - R -> np.random.int(1, 100) or True

    Examples:
    >>> matrices_from_string("1T1,FNR")
    [array([[ 1.,  1.,  1.],
            [ 0., nan, 40.]])]
    >>> matrices_from_string("1T1,FNR", boolean=True)
    [array([[ True,  True,  True],
            [False, False,  True]])]
    >>> matrices_from_string("RR,RR;RRR;1R")
    [array([[21., 80.],
            [43., 10.]]),
    array([[26., 43., 62.]]),
    array([[ 1., 89.]])]
    """

    matrices = []
    for part in string.split(";"):
        str_matrix = np.array([tuple(row) for row in part.split(",")], dtype="object")
        if boolean:
            matrix = np.full(str_matrix.shape, False, dtype=bool)
            matrix[np.nonzero(str_matrix == "N")] = False
            matrix[np.nonzero(str_matrix == "1")] = True
            matrix[np.nonzero(str_matrix == "R")] = True

        else:
            matrix = np.full(str_matrix.shape, np.nan, dtype=np.floating)
            matrix[np.nonzero(str_matrix == "N")] = np.nan
            matrix[np.nonzero(str_matrix == "1")] = 1
            matrix = np.where(
                str_matrix == "R",
                np.random.default_rng().integers(1, 100, matrix.shape),
                matrix,
            )

        matrix[np.nonzero(str_matrix == "T")] = True
        matrix[np.nonzero(str_matrix == "F")] = False

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
        GridSelection(v_split, h_split, split_pixels)

    else:
        with pytest.raises(exception_type):
            GridSelection(v_split, h_split, split_pixels)


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

    gs = GridSelection(*split_vh, split_pixels=True)
    actual_result = gs.find_grid(data)

    num_appearances = np.sum(np.stack(actual_result, axis=-1), axis=-1)

    assert len(actual_result) == np.prod(split_vh)
    assert np.array_equal(num_appearances, (~np.isnan(data) * 1))
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
    result = matrices_from_string(result_string, boolean=False)

    v_split, h_split = split_vh
    gs = GridSelection(v_split, h_split, split_pixels=False)

    matrices = gs.find_grid(data)
    num_appearances = np.sum(np.stack(matrices, axis=-1), axis=-1)

    assert len(matrices) == h_split * v_split
    assert np.array_equal(num_appearances, numeric_values, equal_nan=True)
    assert np.array_equal(matrices, result, equal_nan=True)


@pytest.mark.parametrize(
    "data_string,split_vh,warning_type",
    [
        ("RR,RR", (2, 2), None),
        ("RRR,RRR", (2, 2), RuntimeWarning),
        ("RRR,RRR", (1, 3), None),
        ("RRRR,RRRR", (1, 3), RuntimeWarning),
        ("RR,RR,RR", (2, 1), RuntimeWarning),
        ("RR,RR,RR", (3, 1), None),
        ("NN,RR,RR", (2, 1), None),
    ],
)
def test_warnings(data_string, split_vh, warning_type):
    data = matrices_from_string(data_string)[0]
    gs = GridSelection(*split_vh)

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
