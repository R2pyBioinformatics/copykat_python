"""Tests for MCMC segmentation (R source: CNA.MCMC.R)."""

import numpy as np
import pytest

from copykat.segmentation import cna_mcmc, SegmentationResult, _mc_poisson_gamma


class TestMcPoissonGamma:
    def test_output_length(self):
        data = np.array([1.0, 2.0, 3.0])
        samples = _mc_poisson_gamma(data, alpha=2.0, beta=1.0, mc=1000)
        assert len(samples) == 1000

    def test_positive_samples(self):
        data = np.array([1.0, 2.0, 3.0])
        samples = _mc_poisson_gamma(data, alpha=2.0, beta=1.0, mc=1000)
        assert np.all(samples > 0)


class TestCnaMcmc:
    def test_basic(self):
        rng = np.random.RandomState(42)
        n_genes = 200
        n_cells = 20
        fttmat = rng.normal(0, 0.1, (n_genes, n_cells))
        clu = np.array([1] * 10 + [2] * 10)

        result = cna_mcmc(clu, fttmat, bins=25, cut_cor=0.2, n_cores=1)
        assert isinstance(result, SegmentationResult)
        assert result.log_cna.shape == (n_genes, n_cells)
        assert len(result.breaks) >= 2  # At least start and end

    def test_output_shape_matches_cells(self):
        rng = np.random.RandomState(42)
        n_genes = 100
        n_cells = 15
        fttmat = rng.normal(0, 0.1, (n_genes, n_cells))
        clu = np.array([1] * 8 + [2] * 7)

        result = cna_mcmc(clu, fttmat, bins=20, cut_cor=0.2, n_cores=1)
        assert result.log_cna.shape[1] == n_cells
