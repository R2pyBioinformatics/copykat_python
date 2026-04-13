"""Main CopyKAT pipeline (R source: copykat.R)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from ._data import load_cycle_genes, load_dna_bins, load_gene_annotations
from .annotate import annotate_genes
from .baseline import baseline_gmm, baseline_norm_cl, baseline_synthetic
from .bins import convert_to_bins
from .heatmap import heatmap3
from .segmentation import cna_mcmc
from .smoothing import dlm_smooth

__all__ = ["copykat", "CopykatResult"]


@dataclass
class CopykatResult:
    """Result from the CopyKAT analysis pipeline.

    Parameters
    ----------
    prediction : pd.DataFrame
        Cell-level predictions with columns ``cell.names`` and ``copykat.pred``.
    CNAmat : pd.DataFrame
        Copy number matrix at 220KB bin resolution (hg20) or gene resolution (mm10).
        First 3 columns are genomic coordinates.
    hclustering : Any
        Hierarchical clustering linkage matrix.
    """

    prediction: pd.DataFrame
    CNAmat: pd.DataFrame
    hclustering: Any


def _filter_cells_by_chr(
    anno_mat: pd.DataFrame,
    ngene_chr: int,
    cell_cols: list[str],
) -> list[str]:
    """Filter out cells with insufficient genes per chromosome.

    Parameters
    ----------
    anno_mat : pd.DataFrame
        Annotated expression matrix.
    ngene_chr : int
        Minimum genes per chromosome.
    cell_cols : list[str]
        Cell column names.

    Returns
    -------
    list[str]
        Cell names to remove.
    """
    to_remove = []
    chrom_col = anno_mat["chromosome_name"].values

    for cell in cell_cols:
        vals = anno_mat[cell].values
        nonzero_mask = vals != 0
        if nonzero_mask.sum() < 5:
            to_remove.append(cell)
            continue

        # Check chromosome coverage
        chroms_nonzero = chrom_col[nonzero_mask]
        unique_chroms = np.unique(chrom_col)

        # Run-length encoding equivalent: count contiguous runs of same chromosome
        # among nonzero genes
        rle_lengths = []
        if len(chroms_nonzero) > 0:
            current = chroms_nonzero[0]
            count = 1
            for j in range(1, len(chroms_nonzero)):
                if chroms_nonzero[j] == current:
                    count += 1
                else:
                    rle_lengths.append(count)
                    current = chroms_nonzero[j]
                    count = 1
            rle_lengths.append(count)

        if len(rle_lengths) < len(unique_chroms) or min(rle_lengths) < ngene_chr:
            to_remove.append(cell)

    return to_remove


def copykat(
    rawmat: pd.DataFrame,
    id_type: str = "S",
    cell_line: str = "no",
    ngene_chr: int = 5,
    min_gene_per_cell: int = 200,
    LOW_DR: float = 0.05,
    UP_DR: float = 0.1,
    win_size: int = 25,
    norm_cell_names: list[str] | None = None,
    KS_cut: float = 0.1,
    sam_name: str = "",
    distance: str = "euclidean",
    output_seg: bool = False,
    plot_genes: bool = True,
    genome: str = "hg20",
    n_cores: int = 1,
) -> CopykatResult:
    """Run the CopyKAT copy number inference pipeline.

    Parameters
    ----------
    rawmat : pd.DataFrame
        Raw gene expression matrix with genes as rows and cells as columns.
        Row index should be gene identifiers.
    id_type : str
        Gene ID type: ``"S"`` for gene symbols, ``"E"`` for Ensembl IDs.
    cell_line : str
        ``"yes"`` for pure cell line data, ``"no"`` for tissue samples.
    ngene_chr : int
        Minimum number of genes per chromosome for cell filtering.
    min_gene_per_cell : int
        Minimum genes detected per cell.
    LOW_DR : float
        Minimum detection rate for gene filtering (smoothing step).
    UP_DR : float
        Minimum detection rate for gene filtering (segmentation step).
    win_size : int
        Window size for segmentation (genes per segment).
    norm_cell_names : list[str] | None
        Known normal cell names. If provided, used as baseline.
    KS_cut : float
        KS test cutoff for breakpoint detection (0-1, higher = looser).
    sam_name : str
        Sample name prefix for output files.
    distance : str
        Distance metric: ``"euclidean"``, ``"pearson"``, or ``"spearman"``.
    output_seg : bool
        Whether to output IGV .seg files.
    plot_genes : bool
        Whether to include gene names in heatmap.
    genome : str
        Genome build: ``"hg20"`` or ``"mm10"``.
    n_cores : int
        Number of parallel cores.

    Returns
    -------
    CopykatResult
        Object containing prediction, CNA matrix, and clustering results.
    """
    np.random.seed(1234)
    sample_name = f"{sam_name}_copykat_"

    print("running copykat v1.1.0")

    # Step 1: Read and filter data
    print("step1: read and filter data ...")
    print(f"{rawmat.shape[0]} genes, {rawmat.shape[1]} cells in raw data")

    genes_raw = (rawmat > 0).sum(axis=0)

    if (genes_raw > min_gene_per_cell).sum() == 0:
        raise ValueError("No cells have more than min_gene_per_cell genes")

    if (genes_raw < min_gene_per_cell).sum() > 1:
        keep_cells = genes_raw[genes_raw >= min_gene_per_cell].index
        n_removed = rawmat.shape[1] - len(keep_cells)
        rawmat = rawmat[keep_cells]
        print(f"filtered out {n_removed} cells with less than {min_gene_per_cell} genes; remaining {rawmat.shape[1]} cells")

    # Gene detection rate filtering
    der = (rawmat > 0).sum(axis=1) / rawmat.shape[1]
    if (der > LOW_DR).sum() >= 1:
        rawmat = rawmat.loc[der > LOW_DR]
        print(f"{rawmat.shape[0]} genes past LOW.DR filtering")

    wns1 = "data quality is ok"
    if rawmat.shape[0] < 7000:
        wns1 = "low data quality"
        UP_DR = LOW_DR
        print("WARNING: low data quality; assigned LOW.DR to UP.DR...")

    # Step 2: Gene annotation
    print("step 2: annotations gene coordinates ...")
    anno_mat = annotate_genes(rawmat, id_type=id_type, genome=genome)
    anno_mat = anno_mat.sort_values("abspos").reset_index(drop=True)

    # Remove cell cycle genes and HLA genes (hg20 only)
    if genome == "hg20":
        cycle_genes = load_cycle_genes()
        symbol_col = "hgnc_symbol"
        hla_mask = anno_mat[symbol_col].str.startswith("HLA-", na=False)
        cycle_mask = anno_mat[symbol_col].isin(cycle_genes)
        remove_mask = hla_mask | cycle_mask
        if remove_mask.sum() > 0:
            anno_mat = anno_mat[~remove_mask].reset_index(drop=True)
    else:
        symbol_col = "mgi_symbol"

    # Identify annotation and cell columns
    anno_cols = ["abspos", "chromosome_name", "start_position", "end_position",
                 "ensembl_gene_id", symbol_col, "band"]
    cell_cols = [c for c in anno_mat.columns if c not in anno_cols]

    # Secondary cell filtering
    to_remove = _filter_cells_by_chr(anno_mat, ngene_chr, cell_cols)

    if len(to_remove) == len(cell_cols):
        raise ValueError("All cells are filtered out")

    if to_remove:
        anno_mat = anno_mat.drop(columns=to_remove)
        cell_cols = [c for c in cell_cols if c not in to_remove]

    # Normalize
    rawmat3 = anno_mat[cell_cols].values.astype(np.float64)
    norm_mat = np.log(np.sqrt(rawmat3) + np.sqrt(rawmat3 + 1))
    norm_mat = norm_mat - norm_mat.mean(axis=0, keepdims=True)

    # Step 3: DLM smoothing
    print("step 3: smoothing data with dlm ...")
    norm_mat_smooth = dlm_smooth(norm_mat, dV=0.16, dW=0.001, n_cores=n_cores)

    # Step 4: Measure baseline
    print("step 4: measuring baselines ...")

    if cell_line == "yes":
        print("running pure cell line mode")
        relt = baseline_synthetic(norm_mat_smooth, cell_cols, min_cells=10, n_cores=n_cores)
        expr_relat, syn_normal, cl = relt
        # norm_mat_relat is transposed back: (n_genes, n_cells)
        norm_mat_relat = expr_relat.T
        wns = "run with cell line mode"
        pre_n: list[str] = []

    elif norm_cell_names is not None and len(norm_cell_names) > 1:
        # Known normal cells
        norm_idx = [i for i, c in enumerate(cell_cols) if c in norm_cell_names]
        n_found = len(norm_idx)
        print(f"{n_found} known normal cells found in dataset")
        if n_found == 0:
            raise ValueError("Known normal cells provided but none found in dataset")
        print("run with known normal...")

        basel = np.median(norm_mat_smooth[:, norm_idx], axis=1)

        # Cluster all cells
        dist_mat = pdist(norm_mat_smooth.T, metric="euclidean")
        Z = linkage(dist_mat, method="ward")
        km = 6
        cl = fcluster(Z, t=km, criterion="maxclust")
        while not all(np.bincount(cl)[1:] >= 5):
            km -= 1
            cl = fcluster(Z, t=km, criterion="maxclust")
            if km == 2:
                break

        wns = "run with known normal"
        pre_n = list(norm_cell_names)
        norm_mat_relat = norm_mat_smooth - basel[:, np.newaxis]

    else:
        # Auto-detect baseline
        basa = baseline_norm_cl(norm_mat_smooth, cell_cols, min_cells=5, n_cores=n_cores)
        basel = basa.basel
        wns = basa.wns
        pre_n = basa.pre_n
        cl = basa.cl

        if wns == "unclassified.prediction":
            basa = baseline_gmm(norm_mat_smooth, cell_cols, max_normal=5,
                                mu_cut=0.05, nfraq_cut=0.99, re_before=basa,
                                n_cores=n_cores)
            basel = basa.basel
            wns = basa.wns
            pre_n = basa.pre_n

        norm_mat_relat = norm_mat_smooth - basel[:, np.newaxis]

    # Use smaller gene set for segmentation (UP.DR filter)
    dr2 = (rawmat3 > 0).sum(axis=1) / rawmat3.shape[1]
    up_dr_mask = dr2 >= UP_DR
    norm_mat_relat = norm_mat_relat[up_dr_mask, :]
    anno_mat2 = anno_mat.iloc[up_dr_mask].reset_index(drop=True)

    # Filter cells again after UP.DR filtering
    cell_cols2 = [c for c in anno_mat2.columns if c not in anno_cols]
    to_remove3 = _filter_cells_by_chr(anno_mat2, ngene_chr, cell_cols2)

    if len(to_remove3) == norm_mat_relat.shape[1]:
        raise ValueError("All cells filtered after UP.DR filtering")

    if to_remove3:
        keep_idx = [i for i, c in enumerate(cell_cols2) if c not in to_remove3]
        norm_mat_relat = norm_mat_relat[:, keep_idx]
        cell_cols2 = [cell_cols2[i] for i in keep_idx]

    # Align cluster labels
    cl_dict = {cell_cols[i]: cl[i] for i in range(len(cell_cols)) if i < len(cl)}
    cl_aligned = np.array([cl_dict.get(c, 1) for c in cell_cols2])

    # Step 5: Segmentation
    print("step 5: segmentation...")
    results = cna_mcmc(cl_aligned, norm_mat_relat, win_size, KS_cut, n_cores=n_cores)

    if len(results.breaks) < 25:
        print("too few breakpoints detected; decreased KS.cut to 50%")
        results = cna_mcmc(cl_aligned, norm_mat_relat, win_size, 0.5 * KS_cut, n_cores=n_cores)

    if len(results.breaks) < 25:
        print("too few breakpoints detected; decreased KS.cut to 75%")
        results = cna_mcmc(cl_aligned, norm_mat_relat, win_size, 0.25 * KS_cut, n_cores=n_cores)

    if len(results.breaks) < 25:
        raise ValueError("Too few segments; try decreasing KS.cut or improving data quality")

    results_com = results.log_cna - results.log_cna.mean(axis=0, keepdims=True)

    if genome == "hg20":
        # Step 6: Convert to genomic bins
        print("step 6: convert to genomic bins...")
        dna_bins = load_dna_bins()

        cell_data = pd.DataFrame(results_com, columns=cell_cols2)
        rna_copycat = pd.concat([anno_mat2[anno_cols].reset_index(drop=True), cell_data], axis=1)

        aj = convert_to_bins(dna_bins, rna_copycat, n_cores=n_cores)
        bin_cell_cols = [c for c in aj.rna_adj.columns if c not in ["chrom", "chrompos", "abspos"]]
        uber_mat_adj = aj.rna_adj[bin_cell_cols].values

        if cell_line == "yes":
            # Cell line mode: simpler path
            mat_adj = uber_mat_adj
            hcc_linkage = _hierarchical_cluster(mat_adj, distance, n_cores)
            cna_mat = aj.rna_adj.copy()
            return CopykatResult(
                prediction=pd.DataFrame(),
                CNAmat=cna_mat,
                hclustering=hcc_linkage,
            )

        # Step 7: Adjust baseline
        print("step 7: adjust baseline ...")
        mat_adj, com_pre_n, hcc_linkage = _adjust_and_predict(
            uber_mat_adj, cell_cols2, pre_n, wns, distance, n_cores,
        )

        # Build prediction DataFrame
        prediction = _build_prediction(com_pre_n, rawmat.columns.tolist(), wns)

        # Build CNA result
        cell_adj_df = pd.DataFrame(mat_adj, columns=cell_cols2)
        cna_df = pd.concat([aj.rna_adj[["chrom", "chrompos", "abspos"]].reset_index(drop=True), cell_adj_df], axis=1)

        return CopykatResult(
            prediction=prediction,
            CNAmat=cna_df,
            hclustering=hcc_linkage,
        )

    elif genome == "mm10":
        # mm10: work in gene space (no bin conversion)
        uber_mat_adj = results_com

        mat_adj, com_pre_n, hcc_linkage = _adjust_and_predict(
            uber_mat_adj, cell_cols2, pre_n, wns, distance, n_cores,
        )

        prediction = _build_prediction(com_pre_n, rawmat.columns.tolist(), wns)

        # CNA in gene space
        cna_df = anno_mat2[["abspos", "chromosome_name", "start_position"]].copy()
        cna_df.columns = ["chrom", "chrompos", "abspos"]
        for i, cell in enumerate(cell_cols2):
            cna_df[cell] = mat_adj[:, i]

        return CopykatResult(
            prediction=prediction,
            CNAmat=cna_df,
            hclustering=hcc_linkage,
        )
    else:
        raise ValueError(f"Unsupported genome: {genome}")


def _hierarchical_cluster(
    mat: np.ndarray,
    distance: str,
    n_cores: int,
) -> np.ndarray:
    """Perform hierarchical clustering on cell matrix.

    Parameters
    ----------
    mat : np.ndarray
        Matrix (n_bins, n_cells).
    distance : str
        Distance metric.
    n_cores : int
        Number of cores.

    Returns
    -------
    np.ndarray
        Linkage matrix.
    """
    if distance == "euclidean":
        dist_mat = pdist(mat.T, metric="euclidean")
    else:
        corr = np.corrcoef(mat.T)
        dist_mat = 1 - corr
        # Convert to condensed form
        n = mat.shape[1]
        dist_condensed = []
        for i in range(n):
            for j in range(i + 1, n):
                dist_condensed.append(dist_mat[i, j])
        dist_mat = np.array(dist_condensed)

    return linkage(dist_mat, method="ward")


def _adjust_and_predict(
    uber_mat_adj: np.ndarray,
    cell_names: list[str],
    pre_n: list[str],
    wns: str,
    distance: str,
    n_cores: int,
) -> tuple[np.ndarray, dict[str, str], np.ndarray]:
    """Adjust baseline and predict diploid/aneuploid.

    Parameters
    ----------
    uber_mat_adj : np.ndarray
        Unadjusted CNA matrix (n_bins/genes, n_cells).
    cell_names : list[str]
        Cell names.
    pre_n : list[str]
        Predicted normal cell names.
    wns : str
        Warning string.
    distance : str
        Distance metric.
    n_cores : int
        Number of cores.

    Returns
    -------
    tuple[np.ndarray, dict[str, str], np.ndarray]
        Adjusted matrix, predictions dict, linkage matrix.
    """
    # Initial clustering to split diploid/aneuploid
    Z = _hierarchical_cluster(uber_mat_adj, distance, n_cores)
    hc_labels = fcluster(Z, t=2, criterion="maxclust")

    # Determine which cluster is diploid based on overlap with pre_n
    cl_id = []
    for i in range(1, hc_labels.max() + 1):
        cli = [cell_names[j] for j in range(len(cell_names)) if hc_labels[j] == i]
        pid = len(set(cli) & set(pre_n)) / max(len(cli), 1)
        cl_id.append(pid)

    diploid_cluster = int(np.argmax(cl_id)) + 1
    aneuploid_cluster = int(np.argmin(cl_id)) + 1

    com_pred = {}
    for j, name in enumerate(cell_names):
        if hc_labels[j] == diploid_cluster:
            com_pred[name] = "diploid"
        else:
            com_pred[name] = "aneuploid"

    # Baseline adjustment
    diploid_idx = [j for j, name in enumerate(cell_names) if com_pred[name] == "diploid"]
    if len(diploid_idx) == 0:
        diploid_idx = list(range(len(cell_names)))

    diploid_mean = np.mean(uber_mat_adj[:, diploid_idx], axis=1)
    results_com_rat = uber_mat_adj - diploid_mean[:, np.newaxis]
    results_com_rat = results_com_rat - results_com_rat.mean(axis=0, keepdims=True)

    # Noise estimation from diploid cells
    norm_rat = results_com_rat[:, diploid_idx]
    cf_h = np.std(norm_rat, axis=1, ddof=1)
    base = np.mean(norm_rat, axis=1)

    # Adjust: set values close to baseline to mean
    adj_results = np.zeros_like(results_com_rat)
    for j in range(results_com_rat.shape[1]):
        a = results_com_rat[:, j].copy()
        close_mask = np.abs(a - base) <= 0.25 * cf_h
        a[close_mask] = np.mean(a)
        adj_results[:, j] = a

    # Center columns
    mat_adj = adj_results - adj_results.mean(axis=0, keepdims=True)

    # Final clustering and prediction
    print("step 8: final prediction ...")
    Z_final = _hierarchical_cluster(mat_adj, distance, n_cores)
    hc_final = fcluster(Z_final, t=2, criterion="maxclust")

    cl_id_final = []
    for i in range(1, hc_final.max() + 1):
        cli = [cell_names[j] for j in range(len(cell_names)) if hc_final[j] == i]
        pid = len(set(cli) & set(pre_n)) / max(len(cli), 1)
        cl_id_final.append(pid)

    diploid_cluster_final = int(np.argmax(cl_id_final)) + 1
    aneuploid_cluster_final = int(np.argmin(cl_id_final)) + 1

    com_pre_n = {}
    for j, name in enumerate(cell_names):
        if hc_final[j] == diploid_cluster_final:
            com_pre_n[name] = "diploid"
        else:
            com_pre_n[name] = "aneuploid"

    # Low confidence labels
    if wns == "unclassified.prediction":
        for name in com_pre_n:
            if com_pre_n[name] == "diploid":
                com_pre_n[name] = "c1:diploid:low.conf"
            elif com_pre_n[name] == "aneuploid":
                com_pre_n[name] = "c2:aneuploid:low.conf"

    return mat_adj, com_pre_n, Z_final


def _build_prediction(
    com_pre_n: dict[str, str],
    all_cells: list[str],
    wns: str,
) -> pd.DataFrame:
    """Build prediction DataFrame including filtered cells.

    Parameters
    ----------
    com_pre_n : dict[str, str]
        Cell name → prediction mapping.
    all_cells : list[str]
        All original cell names.
    wns : str
        Warning string.

    Returns
    -------
    pd.DataFrame
        Prediction DataFrame with columns ``cell.names`` and ``copykat.pred``.
    """
    names = []
    preds = []

    for name, pred in com_pre_n.items():
        names.append(name)
        preds.append(pred)

    # Add back filtered cells as "not.defined"
    for cell in all_cells:
        if cell not in com_pre_n:
            names.append(cell)
            preds.append("not.defined")

    return pd.DataFrame({"cell.names": names, "copykat.pred": preds})
