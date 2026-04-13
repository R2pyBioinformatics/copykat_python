"""Tests for the main copykat pipeline (R source: copykat.R)."""

import numpy as np
import pandas as pd
import pytest

from copykat.copykat import copykat, CopykatResult
from copykat._data import load_example_data


@pytest.fixture
def example_data():
    """Load the example dataset."""
    return load_example_data()


class TestCopykat:
    @pytest.mark.slow
    def test_end_to_end_small(self, example_data):
        """Test the full pipeline on a small subset."""
        subset = example_data.iloc[:, :50]
        result = copykat(
            rawmat=subset,
            id_type="S",
            cell_line="no",
            ngene_chr=5,
            win_size=25,
            KS_cut=0.2,
            sam_name="test",
            distance="euclidean",
            n_cores=1,
        )
        assert isinstance(result, CopykatResult)
        assert result.prediction.shape[0] > 0
        assert "cell.names" in result.prediction.columns
        assert "copykat.pred" in result.prediction.columns
        assert result.CNAmat.shape[0] > 0
        assert "chrom" in result.CNAmat.columns

    def test_prediction_categories(self, example_data):
        """Test that predictions contain expected categories."""
        subset = example_data.iloc[:, :50]
        result = copykat(
            rawmat=subset,
            id_type="S",
            cell_line="no",
            ngene_chr=5,
            win_size=25,
            KS_cut=0.2,
            distance="euclidean",
            n_cores=1,
        )
        valid_preds = {"diploid", "aneuploid", "not.defined",
                       "c1:diploid:low.conf", "c2:aneuploid:low.conf"}
        for pred in result.prediction["copykat.pred"].unique():
            assert pred in valid_preds, f"Unexpected prediction: {pred}"

    def test_all_cells_in_prediction(self, example_data):
        """Test that all input cells appear in predictions."""
        subset = example_data.iloc[:, :50]
        result = copykat(
            rawmat=subset,
            id_type="S",
            cell_line="no",
            ngene_chr=5,
            win_size=25,
            KS_cut=0.2,
            distance="euclidean",
            n_cores=1,
        )
        pred_cells = set(result.prediction["cell.names"])
        input_cells = set(subset.columns)
        assert input_cells.issubset(pred_cells)
