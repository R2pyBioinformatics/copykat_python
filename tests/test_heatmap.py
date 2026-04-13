"""Tests for heatmap visualization (R source: heatmap.3.R)."""

import numpy as np
import pytest

from copykat.heatmap import heatmap3


class TestHeatmap3:
    def test_basic_call(self):
        rng = np.random.RandomState(42)
        data = rng.randn(50, 30)
        result = heatmap3(data, row_cluster=True, col_cluster=False, show=False)
        assert "row_order" in result
        assert len(result["row_order"]) == 50

    def test_no_clustering(self):
        data = np.random.RandomState(42).randn(20, 10)
        result = heatmap3(data, row_cluster=False, col_cluster=False, show=False)
        np.testing.assert_array_equal(result["row_order"], np.arange(20))

    def test_save(self, tmp_path):
        data = np.random.RandomState(42).randn(20, 10)
        path = str(tmp_path / "test.png")
        result = heatmap3(data, save_path=path, show=False)
        import os
        assert os.path.exists(path)
