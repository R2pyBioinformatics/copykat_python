"""Tests for baseline detection (R source: baseline.norm.cl.R, baseline.GMM.R, baseline.synthetic.R)."""

import numpy as np
import pytest

from copykat.baseline import baseline_gmm, baseline_norm_cl, baseline_synthetic, BaselineResult


@pytest.fixture
def mock_smooth_mat():
    """Create a mock smoothed matrix with two populations."""
    rng = np.random.RandomState(42)
    n_genes = 200
    n_diploid = 30
    n_aneuploid = 20
    # Diploid cells: near zero
    diploid = rng.normal(0, 0.02, (n_genes, n_diploid))
    # Aneuploid cells: some regions shifted
    aneuploid = rng.normal(0, 0.02, (n_genes, n_aneuploid))
    aneuploid[50:100, :] += 0.3  # amplification
    aneuploid[150:180, :] -= 0.3  # deletion

    mat = np.hstack([diploid, aneuploid])
    names = [f"diploid_{i}" for i in range(n_diploid)] + [f"aneuploid_{i}" for i in range(n_aneuploid)]
    return mat, names


class TestBaselineNormCl:
    def test_returns_baseline_result(self, mock_smooth_mat):
        mat, names = mock_smooth_mat
        result = baseline_norm_cl(mat, names, min_cells=5, n_cores=1)
        assert isinstance(result, BaselineResult)
        assert len(result.basel) == mat.shape[0]
        assert isinstance(result.cl, np.ndarray)
        assert len(result.cl) == mat.shape[1]


class TestBaselineGmm:
    def test_returns_baseline_result(self, mock_smooth_mat):
        mat, names = mock_smooth_mat
        result = baseline_gmm(mat, names, max_normal=5, n_cores=1)
        assert isinstance(result, BaselineResult)
        assert len(result.basel) == mat.shape[0]


class TestBaselineSynthetic:
    def test_returns_tuple(self, mock_smooth_mat):
        mat, names = mock_smooth_mat
        expr_relat, syn_normal, cl = baseline_synthetic(mat, names, min_cells=5, n_cores=1)
        assert expr_relat.shape[1] == mat.shape[0]  # (n_cells, n_genes)
        assert len(cl) == mat.shape[1]
