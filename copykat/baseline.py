"""Baseline (diploid) cell detection (R source: baseline.norm.cl.R, baseline.GMM.R, baseline.synthetic.R)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import f as f_dist
from sklearn.metrics import silhouette_samples
from sklearn.mixture import GaussianMixture

__all__ = ["baseline_norm_cl", "baseline_gmm", "baseline_synthetic", "BaselineResult"]


@dataclass
class BaselineResult:
    """Result from baseline detection.

    Parameters
    ----------
    basel : np.ndarray
        Baseline copy number profile (per-gene medians of diploid cells).
    wns : str
        Warning string. ``""`` means confident, ``"unclassified.prediction"``
        means low confidence.
    pre_n : list[str]
        Names of predicted normal (diploid) cells.
    cl : np.ndarray
        Cluster assignments for all cells.
    """

    basel: np.ndarray
    wns: str
    pre_n: list[str]
    cl: np.ndarray


def _fit_gmm_3comp(
    data: np.ndarray,
    sigma: float,
    maxit: int = 5000,
) -> GaussianMixture:
    """Fit a 3-component GMM matching mixtools::normalmixEM settings.

    Parameters
    ----------
    data : np.ndarray
        1-D array of values.
    sigma : float
        Initial standard deviation for all components.
    maxit : int
        Maximum EM iterations.

    Returns
    -------
    GaussianMixture
        Fitted model.
    """
    gm = GaussianMixture(
        n_components=3,
        covariance_type="tied",  # arbvar=FALSE → equal variances
        means_init=np.array([[-0.2], [0.0], [0.2]]),
        weights_init=np.array([1 / 3, 1 / 3, 1 / 3]),
        max_iter=maxit,
        n_init=1,
        tol=1e-6,
        random_state=0,
    )
    gm.fit(data.reshape(-1, 1))
    return gm


def baseline_norm_cl(
    norm_mat_smooth: np.ndarray,
    cell_names: list[str],
    min_cells: int = 5,
    n_cores: int = 1,
) -> BaselineResult:
    """Find cluster of diploid cells using integrative clustering.

    Parameters
    ----------
    norm_mat_smooth : np.ndarray
        Smoothed normalized expression matrix (n_genes, n_cells).
    cell_names : list[str]
        Cell names corresponding to columns.
    min_cells : int
        Minimum cells per cluster.
    n_cores : int
        Number of parallel cores (used for distance computation).

    Returns
    -------
    BaselineResult
        Baseline profile, warning string, predicted normal cells, and clusters.
    """
    n_cells = norm_mat_smooth.shape[1]

    # Hierarchical clustering (ward.D2)
    dist_mat = pdist(norm_mat_smooth.T, metric="euclidean")
    Z = linkage(dist_mat, method="ward")

    km = 6
    ct = fcluster(Z, t=km, criterion="maxclust")

    # Reduce k until all clusters have >= min_cells
    while not all(np.bincount(ct)[1:] >= min_cells):
        km -= 1
        ct = fcluster(Z, t=km, criterion="maxclust")
        if km == 2:
            break

    # GMM validation for each cluster
    sdm = []
    for i in range(1, ct.max() + 1):
        mask = ct == i
        if mask.sum() == 0:
            continue
        data_c = np.median(norm_mat_smooth[:, mask], axis=1)
        sx = max(0.05, 0.5 * np.std(data_c))

        gm = _fit_gmm_3comp(data_c, sigma=sx, maxit=5000)
        # Extract tied sigma
        sigma = np.sqrt(gm.covariances_.flatten()[0]) if gm.covariances_.ndim > 1 else np.sqrt(float(gm.covariances_))
        sdm.append(sigma)

    # Quality control
    sq_dist = squareform(dist_mat)
    ct2 = fcluster(Z, t=2, criterion="maxclust")
    sil_vals = silhouette_samples(sq_dist, ct2, metric="precomputed")
    wn = np.mean(sil_vals)

    # F-test for variance homogeneity
    n_genes = norm_mat_smooth.shape[0]
    if len(sdm) >= 2 and min(sdm) > 0:
        f_stat = max(sdm) ** 2 / min(sdm) ** 2
        pdt = 1.0 - f_dist.cdf(f_stat, n_genes, n_genes)
    else:
        pdt = 1.0

    counts = np.bincount(ct)[1:]
    all_above_min = all(c >= min_cells for c in counts if c > 0)

    if wn <= 0.15 or not all_above_min or pdt > 0.05:
        wns = "unclassified.prediction"
    else:
        wns = ""

    # Select cluster with minimum sigma as diploid baseline
    min_sigma_cluster = int(np.argmin(sdm)) + 1
    mask = ct == min_sigma_cluster
    basel = np.median(norm_mat_smooth[:, mask], axis=1)
    pre_n = [cell_names[j] for j in range(n_cells) if mask[j]]

    return BaselineResult(basel=basel, wns=wns, pre_n=pre_n, cl=ct)


def baseline_gmm(
    cna_mat: np.ndarray,
    cell_names: list[str],
    max_normal: int = 5,
    mu_cut: float = 0.05,
    nfraq_cut: float = 0.99,
    re_before: BaselineResult | None = None,
    n_cores: int = 1,
) -> BaselineResult:
    """Pre-define normal (diploid) cells using Gaussian Mixture Model.

    Parameters
    ----------
    cna_mat : np.ndarray
        CNA matrix (n_genes, n_cells).
    cell_names : list[str]
        Cell names.
    max_normal : int
        Stop after finding this many diploid cells.
    mu_cut : float
        Threshold for neutral component mean.
    nfraq_cut : float
        Fraction threshold for dominant neutral component.
    re_before : BaselineResult | None
        Previous result to fall back to.
    n_cores : int
        Number of parallel cores.

    Returns
    -------
    BaselineResult
        Baseline profile, warning, normal cells, clusters.
    """
    n_normal = []
    n_normal_names = []

    for m in range(cna_mat.shape[1]):
        sam = cna_mat[:, m]
        sg = max(0.05, 0.5 * np.std(sam))

        gm = _fit_gmm_3comp(sam, sigma=sg, maxit=500)

        mus = gm.means_.flatten()
        weights = gm.weights_

        # Count components with |mu| <= mu_cut
        neutral_mask = np.abs(mus) <= mu_cut

        if neutral_mask.sum() >= 1:
            frq = weights[neutral_mask].sum()
            pred = "diploid" if frq > nfraq_cut else "aneuploid"
        else:
            pred = "aneuploid"

        n_normal.append(pred)
        n_normal_names.append(cell_names[m])

        if sum(1 for p in n_normal if p == "diploid") >= max_normal:
            break

    pre_n = [n_normal_names[i] for i, p in enumerate(n_normal) if p == "diploid"]

    # Cluster all cells
    dist_mat = pdist(cna_mat.T, metric="euclidean")
    Z = linkage(dist_mat, method="ward")
    km = 6
    ct = fcluster(Z, t=km, criterion="maxclust")

    if len(pre_n) > 2:
        pre_n_idx = [cell_names.index(name) for name in pre_n]
        basel = np.mean(cna_mat[:, pre_n_idx], axis=1)
        return BaselineResult(basel=basel, wns="", pre_n=pre_n, cl=ct)
    elif re_before is not None:
        return re_before
    else:
        # Fallback: use overall median
        return BaselineResult(
            basel=np.median(cna_mat, axis=1),
            wns="unclassified.prediction",
            pre_n=[],
            cl=ct,
        )


def baseline_synthetic(
    norm_mat: np.ndarray,
    cell_names: list[str],
    min_cells: int = 10,
    n_cores: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate baseline using synthetic normal cells (cell line mode).

    Parameters
    ----------
    norm_mat : np.ndarray
        Normalized expression matrix (n_genes, n_cells).
    cell_names : list[str]
        Cell names.
    min_cells : int
        Minimum cells per cluster.
    n_cores : int
        Number of cores.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - expr_relat: Relative expression matrix (n_cells, n_genes)
        - syn_normal: Synthetic normal profiles (n_genes, n_clusters)
        - cl: Cluster assignments
    """
    dist_mat = pdist(norm_mat.T, metric="euclidean")
    Z = linkage(dist_mat, method="ward")

    km = 6
    ct = fcluster(Z, t=km, criterion="maxclust")

    while not all(np.bincount(ct)[1:] >= min_cells):
        km -= 1
        ct = fcluster(Z, t=km, criterion="maxclust")
        if km == 2:
            break

    rng = np.random.RandomState(123)  # Match R's set.seed(123)
    expr_relat_parts = []
    syn_parts = []

    for i in range(1, ct.max() + 1):
        mask = ct == i
        if mask.sum() == 0:
            continue
        data_c1 = norm_mat[:, mask]
        sd1 = np.std(data_c1, axis=1, ddof=1)  # R's sd uses n-1

        # For each gene, sample one value from N(0, sd_i)
        syn_norm = rng.normal(0, sd1)

        # Relative to synthetic baseline
        relat1 = data_c1 - syn_norm[:, np.newaxis]
        expr_relat_parts.append(relat1.T)  # (n_cells_in_cluster, n_genes)
        syn_parts.append(syn_norm)

    expr_relat = np.vstack(expr_relat_parts)
    syn_normal = np.column_stack(syn_parts)

    return expr_relat, syn_normal, ct
