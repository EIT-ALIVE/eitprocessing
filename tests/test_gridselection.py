import numpy as np
import pytest
from eitprocessing.roi_selection.gridselection import GridSelection


@pytest.mark.parametrize(
    "n_rows,n_columns,v_split,h_split,result",
    [
        (2, 1, 2, 1, "T,F;F,T"),
        (3, 1, 3, 1, "T,F,F;F,T,F;F,F,T"),
        (1, 3, 1, 3, "TFF;FTF;FFT"),
        (1, 3, 1, 2, "TTF;FFT"),
        (
            4,
            4,
            2,
            2,
            "TTFF,TTFF,FFFF,FFFF;"
            "FFTT,FFTT,FFFF,FFFF;"
            "FFFF,FFFF,TTFF,TTFF;"
            "FFFF,FFFF,FFTT,FFTT",
        ),
        (
            5,
            5,
            2,
            2,
            "TTTFF,TTTFF,TTTFF,FFFFF,FFFFF;"
            "FFFTT,FFFTT,FFFTT,FFFFF,FFFFF;"
            "FFFFF,FFFFF,FFFFF,TTTFF,TTTFF;"
            "FFFFF,FFFFF,FFFFF,FFFTT,FFFTT",
        ),
        (5, 2, 1, 2, "TF,TF,TF,TF,TF;FT,FT,FT,FT,FT"),
    ],
)
def test_no_split_pixels_no_nans(n_rows, n_columns, v_split, h_split, result):
    data = np.ones((n_rows, n_columns))

    gs = GridSelection(v_split, h_split)
    matrices = gs.find_grid(data)

    assert len(matrices) == h_split * v_split
    num_appearances = np.sum(np.stack(matrices, axis=-1), axis=-1)
    assert np.array_equal(num_appearances, np.ones(data.shape))

    result = [
        np.array([tuple(row) for row in matrix.split(",")]) == "T"
        for matrix in result.split(";")
    ]
    assert np.array_equal(matrices, result)
