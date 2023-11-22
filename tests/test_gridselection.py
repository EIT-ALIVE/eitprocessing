import warnings
from typing import Final
import numpy as np
import pytest
from numpy.typing import NDArray
from eitprocessing.roi_selection.gridselection import GridSelection
from eitprocessing.roi_selection.gridselection import InvalidDivision
from eitprocessing.roi_selection.gridselection import InvalidHorizontalDivision
from eitprocessing.roi_selection.gridselection import InvalidVerticalDivision
from eitprocessing.roi_selection.gridselection import MoreHorizontalGroupsThanColumns
from eitprocessing.roi_selection.gridselection import MoreVerticalGroupsThanRows
from eitprocessing.roi_selection.gridselection import UnevenHorizontalDivision
from eitprocessing.roi_selection.gridselection import UnevenVerticalDivision


N: Final = np.nan  # shorthand for readabililty


def matrix_from_string(string: str) -> NDArray:
    """
    Generate a of matrix from a string containing a representation of that
    matrix.

    A representation of a matrix contains one character per cell that describes
    the value of that cell. Rows are delimited by commas. Matrices are
    delimited by semi-colons.

    The following characters are transformed to these corresponding values:
    - T / 1 -> 1
    - F / 0 -> 0
    - R -> np.random.int(2, 100)
    - N / any other character -> np.nan

    Examples:
    >>> matrix_from_string("1T1,FNR")
    array([[ 1.,  1.,  1.],
           [ 0., nan, 93.]])
    >>> matrix_from_string("RRR,RNR")
    array([[10., 68., 20.],
           [46., nan, 25.]])
    """

    str_matrix = np.array([tuple(row) for row in string.split(",")], dtype="object")
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
    return matrix


def matrices_from_string(string: str) -> list[NDArray]:
    """
    Generate a list of matrices from a string representation containing
    multiple matrices.

    The input string should contain the representation of one or multiple
    matrices to be used in `matrix_from_string()`, deliminated by `;`.

    Examples:
    >>>matrices_from_string("RR,RR;RRR;1R")
    [array([[64., 82.],
            [40., 65.]]),
    array([[56., 40., 88.]]),
    array([[ 1., 76.]])]
    """

    return [matrix_from_string(part) for part in string.split(";")]


@pytest.mark.parametrize(
    "split_vh,split_columns,split_rows,ign_nan_cols,ign_nan_rows,exception_type",
    [
        ((1, 1), False, False, True, True, None),
        ((1, 1), False, True, True, True, None),
        ((1, 1), True, False, True, True, None),
        ((1, 1), True, True, False, True, None),
        ((1, 1), True, True, True, False, None),
        # Vertical divider invalid
        ((0, 1), False, False, True, True, InvalidVerticalDivision),
        ((-1, 1), False, False, True, True, InvalidVerticalDivision),
        ((1.1, 1), False, False, True, True, TypeError),
        # ( Horiz)ontal divider invalid
        ((1, 0), False, False, True, True, InvalidHorizontalDivision),
        ((1, -1), False, False, True, True, InvalidHorizontalDivision),
        ((1, 1.1), False, False, True, True, TypeError),
        # split_rows invalid
        ((2, 2), "not a boolean", False, True, True, TypeError),
        ((2, 2), 1, False, True, True, TypeError),
        ((2, 2), 0, False, True, True, TypeError),
        # split_columns invalid
        ((2, 2), False, "not a boolean", True, True, TypeError),
        ((2, 2), False, 1, True, True, TypeError),
        ((2, 2), False, 0, True, True, TypeError),
        # ignore_nan_rows invalid
        ((1, 1), False, False, "not a boolean", True, TypeError),
        ((1, 1), False, False, 1, True, TypeError),
        # ignore_nan_columns invalid
        ((1, 1), False, False, True, "not a boolean", TypeError),
        ((1, 1), False, False, True, 1, TypeError),
    ],
)
def test_initialisation(  # pylint: disable=too-many-arguments
    split_vh: tuple[int, int],
    split_columns: bool,
    split_rows: bool,
    ign_nan_cols: bool,
    ign_nan_rows: bool,
    exception_type: type[Exception] | None,
):
    """
    Test the initialisation of GridSelection and corresponding expected errors.

    Args:
        split_columns (bool): whether to allow splitting columns.
        split_rows (bool): whether to allow splitting rows.
        ign_nan_cols (bool): whether to ignore NaN columns.
        ign_nan_rows (bool): whether to ignonre NaN rows.
        exception_type (type[Exception] | None): type of exception expected to
            be raised.
    """

    if exception_type is None:
        GridSelection(
            *split_vh,
            split_columns=split_columns,
            split_rows=split_rows,
            ignore_nan_columns=ign_nan_cols,
            ignore_nan_rows=ign_nan_rows,
        )

    else:
        with pytest.raises(exception_type):
            GridSelection(
                *split_vh,
                split_columns=split_columns,
                split_rows=split_rows,
                ignore_nan_columns=ign_nan_cols,
                ignore_nan_rows=ign_nan_rows,
            )


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
def test_warnings(
    data_string: str,
    split_vh: tuple[int, int],
    split_rows: bool,
    split_columns: bool,
    warning_type: type[Warning] | None,
):
    """
    Test for warnings generated when `find_grid()` is called.

    Args:
        data_string (str): represents the input data, to be converted using
            `matrices_from_string()`
        split_vh (tuple[int, int]): `v_split` and `h_split`.
        split_rows (bool): whether to allow splitting rows.
        split_columns (bool): whether to allow splitting columns.
        warning_type (type[Warning] | None): type of warning to be expected.
    """

    data = matrix_from_string(data_string)
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
    "data_string,split_vh,split_rows,split_columns,exception_type",
    [
        ("RR,RR", (2, 2), False, False, None),
        ("RR,RR", (3, 1), False, False, InvalidVerticalDivision),
        ("RR,RR", (1, 3), False, False, InvalidHorizontalDivision),
        ("RR,RR", (3, 1), False, False, InvalidDivision),
        ("RR,RR", (1, 3), False, False, InvalidDivision),
        ("RR,RR", (3, 2), True, False, None),
        ("RR,RR", (3, 2), False, False, InvalidVerticalDivision),
        ("RR,RR", (2, 3), False, True, None),
        ("RR,RR", (2, 3), False, False, InvalidHorizontalDivision),
    ],
)
def test_exceptions(
    data_string: str,
    split_vh: tuple[int, int],
    split_rows: bool,
    split_columns: bool,
    exception_type: type[Exception] | None,
):
    """
    Test for exceptions raised when `find_grid()` is called.

    Args:
        data_string (str): represents the input data, to be converted using `matrices_from_string()`.
        split_vh (tuple[int, int]): `v_split` and `h_split`.
        split_rows (bool): whether to allow splitting rows.
        split_columns (bool): whether to allow splitting columns.
        exception_type (type[Exception] | None): type of exception expected to be raised.
    """

    data = matrix_from_string(data_string)
    gs = GridSelection(*split_vh, split_columns=split_columns, split_rows=split_rows)

    if exception_type is None:
        gs.find_grid(data)

    else:
        with pytest.raises(exception_type):
            gs.find_grid(data)


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
def test_no_split_pixels_no_nans(
    shape: tuple[int, int], split_vh: tuple[int, int], result_string: str
):
    """
    Test `find_grid()` without split rows/columns and no NaN values.

    Args:
        shape (tuple[int, int]): shape of the input data to be generated.
        split_vh (tuple[int, int]):  `v_split` and `h_split`.
        result_string (str): represents the expected result, to be converted
            using `matrices_from_string()`.
    """

    data = np.random.default_rng().integers(1, 100, shape)
    result_matrices = matrices_from_string(result_string)

    gs = GridSelection(*split_vh)
    matrices = gs.find_grid(data)

    num_appearances = np.sum(np.stack(matrices, axis=-1), axis=-1)

    assert len(matrices) == np.prod(split_vh)
    assert np.array_equal(num_appearances, (~np.isnan(data) * 1))
    assert np.array_equal(matrices, result_matrices)


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
def test_no_split_pixels_nans(
    data_string: str, split_vh: tuple[int, int], result_string: str
):
    """
    Test `find_grid()` without row/column splitting, with NaN values.

    Args:
        data_string (str): represents the input data, to be converted using `matrices_from_string()`.
        split_vh (tuple[int, int]): `v_split` and `h_split`.
        result_string (str): represents the expected result, to be converted
            using `matrices_from_string()`.
    """

    data = matrix_from_string(data_string)
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
        (
            (3, 3),
            (2, 2),
            [
                [[1, 0.5, 0], [0.5, 0.25, 0], [0, 0, 0]],
                [[0, 0.5, 1], [0, 0.25, 0.5], [0, 0, 0]],
                [[0, 0, 0], [0.5, 0.25, 0], [1, 0.5, 0]],
                [[0, 0, 0], [0, 0.25, 0.5], [0, 0.5, 1]],
            ],
        ),
    ],
)
def test_split_pixels_no_nans(
    shape: tuple[int, int], split_vh: tuple[int, int], result: list[list[list[float]]]
):
    """
    Test `find_grid()` with split rows/columns and no NaN values.

    Args:
        shape (tuple[int, int]): shape of the input data to be generated.
        split_vh (tuple[int, int]):  `v_split` and `h_split`.
        result (str): list of lists to be converted to matrices, representing
            the expected result.
    """
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
    "data_string,split_vh,result",
    (
        (
            "NRRR,NRRR",
            (2, 2),
            [
                [[N, 1, 0.5, 0], [N, 0, 0, 0]],
                [[N, 0, 0.5, 1], [N, 0, 0, 0]],
                [[N, 0, 0, 0], [N, 1, 0.5, 0]],
                [[N, 0, 0, 0], [N, 0, 0.5, 1]],
            ],
        ),
        (
            "RNRR,RNRR,RNRR,NNNN",
            (2, 2),
            [
                [[1, N, 0, 0], [0.5, N, 0, 0], [0, N, 0, 0], [N, N, N, N]],
                [[0, N, 1, 1], [0, N, 0.5, 0.5], [0, N, 0, 0], [N, N, N, N]],
                [[0, N, 0, 0], [0.5, N, 0, 0], [1, N, 0, 0], [N, N, N, N]],
                [[0, N, 0, 0], [0, N, 0.5, 0.5], [0, N, 1, 1], [N, N, N, N]],
            ],
        ),
    ),
)
def test_split_pixels_nans(data_string, split_vh, result):
    """
    Test `find_grid()` with row/column splitting, with NaN values.

    Args:
        data_string (str): represents the input data, to be converted using `matrices_from_string()`.
        split_vh (tuple[int, int]): `v_split` and `h_split`.
        result (str): list of list representation of matrices, representing
            the expected result.
    """

    data = matrix_from_string(data_string)
    expected_result = [np.array(r) for r in result]
    numeric_values = np.ones(data.shape)
    numeric_values[np.isnan(data)] = np.nan

    gs = GridSelection(*split_vh, split_rows=True, split_columns=True)
    actual_result = gs.find_grid(data)

    num_appearances = np.sum(np.stack(actual_result, axis=-1), axis=-1)

    assert len(actual_result) == np.prod(split_vh)
    assert len(actual_result) == len(expected_result)
    assert np.array_equal(num_appearances, numeric_values, equal_nan=True)


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
def test_matrix_layout(split_vh: tuple[int, int], result: list[list[int]]):
    """
    Test `matrix_layout` method.

    Args:
        split_vh (tuple[int, int]): `v_split` and `h_split`.
        result (list[list[int]]): list representation of a matrix, representing
            the expected result.
    """

    gs = GridSelection(*split_vh)

    assert np.array_equal(gs.matrix_layout, np.array(result))
