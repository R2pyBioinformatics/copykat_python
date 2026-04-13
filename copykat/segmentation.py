"""MCMC-based copy number segmentation (R source: CNA.MCMC.R)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import ks_2samp

__all__ = ["cna_mcmc", "SegmentationResult"]


@dataclass
class SegmentationResult:
    """Result from CNA MCMC segmentation.

    Parameters
    ----------
    log_cna : np.ndarray
        Segmented copy number matrix in log space (n_genes, n_cells).
    breaks : np.ndarray
        Array of breakpoint positions (1-indexed, matching R convention).
    """

    log_cna: np.ndarray
    breaks: np.ndarray


def _mc_poisson_gamma(
    data: np.ndarray,
    alpha: float,
    beta: float,
    mc: int = 1000,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Sample from the Poisson-Gamma conjugate posterior.

    Matches R's MCMCpack::MCpoissongamma(y, alpha, beta, mc).
    Posterior: Gamma(alpha + sum(y), beta + n).

    Parameters
    ----------
    data : np.ndarray
        Observed count-like data.
    alpha : float
        Prior shape parameter.
    beta : float
        Prior rate parameter.
    mc : int
        Number of posterior samples.
    rng : np.random.RandomState | None
        Random state for reproducibility.

    Returns
    -------
    np.ndarray
        Posterior samples of the Poisson rate parameter.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    post_shape = alpha + np.sum(data)
    post_rate = beta + len(data)
    samples = rng.gamma(post_shape, 1.0 / post_rate, size=mc)
    return samples


def cna_mcmc(
    clu: np.ndarray,
    fttmat: np.ndarray,
    bins: int,
    cut_cor: float,
    n_cores: int = 1,
) -> SegmentationResult:
    """MCMC segmentation for copy number analysis.

    Breakpoints are identified using cluster medians, then segmentation
    is applied to all individual cells.

    Parameters
    ----------
    clu : np.ndarray
        Cluster assignments for cells (1-indexed).
    fttmat : np.ndarray
        Log-scale normalized expression matrix (n_genes, n_cells).
    bins : int
        Window size (number of genes per segment).
    cut_cor : float
        KS test cutoff for breakpoint detection (0-1).
    n_cores : int
        Number of parallel cores.

    Returns
    -------
    SegmentationResult
        Segmented CNA matrix (n_genes, n_cells) and breakpoints.
    """
    n = fttmat.shape[0]
    n_cells = fttmat.shape[1]

    # Step 1: Compute median profiles per cluster (for breakpoint detection)
    unique_clusters = np.unique(clu)
    con_parts = []
    for i in unique_clusters:
        mask = clu == i
        data_c = np.median(fttmat[:, mask], axis=1)
        con_parts.append(data_c)

    cluster_medians = np.exp(np.column_stack(con_parts))

    # Step 2: Find breakpoints using KS test on Poisson-Gamma posteriors
    # R uses 1-based indexing: breks <- c(seq(1, as.integer(n/bins-1)*bins, bins), n)
    # Python equivalent with 1-based positions:
    br_all = set()

    for c in range(cluster_medians.shape[1]):
        # R: seq(1, as.integer(n/bins-1)*bins, bins)
        breks = list(range(1, int(n / bins - 1) * bins + 1, bins)) + [n]

        bre = []
        for i in range(len(breks) - 2):
            # R indexing: breks[i]:breks[i+1] (1-based inclusive)
            # Python: breks[i]-1 to breks[i+1] (0-based, exclusive end)
            start1 = breks[i] - 1
            end1 = breks[i + 1]
            seg1 = cluster_medians[start1:end1, c]

            start2 = breks[i + 1]  # (breks[i+1]+1) in R = breks[i+1] in 0-based
            end2 = breks[i + 2]
            seg2 = cluster_medians[start2:end2, c]

            if len(seg1) == 0 or len(seg2) == 0:
                continue

            a1 = max(np.mean(seg1), 0.001)
            rng1 = np.random.RandomState(42)
            posterior1 = _mc_poisson_gamma(seg1, a1, 1, mc=1000, rng=rng1)

            a2 = max(np.mean(seg2), 0.001)
            rng2 = np.random.RandomState(42)
            posterior2 = _mc_poisson_gamma(seg2, a2, 1, mc=1000, rng=rng2)

            ks_stat = ks_2samp(posterior1, posterior2).statistic
            if ks_stat > cut_cor:
                bre.append(breks[i + 1])

        br_all.update(bre)

    # R: BR <- sort(unique(c(BR, breks))) — includes 1 and n
    br = sorted(set([1] + list(br_all) + [n]))

    # Step 3: Segment each INDIVIDUAL cell (not cluster medians)
    # R line 59: norm.mat.sm <- exp(fttmat)  — reassigns to full cell matrix
    cell_data = np.exp(fttmat)

    def _segment_cell(z: int) -> np.ndarray:
        x = np.zeros(n)
        rng_cell = np.random.RandomState(42)
        for i in range(len(br) - 1):
            # R: BR[i]:BR[i+1] (1-based inclusive)
            start = br[i] - 1
            end = br[i + 1]
            seg = cell_data[start:end, z]
            if len(seg) == 0:
                continue
            a = max(np.mean(seg), 0.001)
            posterior = _mc_poisson_gamma(seg, a, 1, mc=1000, rng=rng_cell)
            x[start:end] = np.mean(posterior)
        x = np.log(np.maximum(x, 1e-10))
        return x

    if n_cores > 1:
        results = Parallel(n_jobs=n_cores)(
            delayed(_segment_cell)(z) for z in range(n_cells)
        )
    else:
        results = [_segment_cell(z) for z in range(n_cells)]

    log_cna = np.column_stack(results)
    br_array = np.array(br)

    return SegmentationResult(log_cna=log_cna, breaks=br_array)
