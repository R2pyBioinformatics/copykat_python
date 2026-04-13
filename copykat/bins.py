"""Gene-to-genomic-bin conversion (R source: convert.all.bins.hg20.R)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ._data import load_dna_bins, load_gene_annotations

__all__ = ["convert_to_bins", "BinConversionResult"]


@dataclass
class BinConversionResult:
    """Result from gene-to-bin conversion.

    Parameters
    ----------
    dna_adj : pd.DataFrame
        Adjusted DNA bin coordinate DataFrame (excluding MT chromosome).
    rna_adj : pd.DataFrame
        Adjusted RNA expression at bin resolution with coordinate columns.
    """

    dna_adj: pd.DataFrame
    rna_adj: pd.DataFrame


def convert_to_bins(
    dna_mat: pd.DataFrame,
    rna_mat: pd.DataFrame,
    n_cores: int = 1,
) -> BinConversionResult:
    """Convert gene-by-cell matrix to genomic bins-by-cell matrix.

    Maps gene expression data to 220KB fixed-size genomic bins using
    median aggregation, with nearest-neighbor imputation for empty bins.

    Parameters
    ----------
    dna_mat : pd.DataFrame
        Genomic bin coordinates (chrom, chrompos, abspos).
    rna_mat : pd.DataFrame
        Annotated expression matrix with annotation columns prepended
        (annotation cols + cell expression columns). Must include
        ``hgnc_symbol`` column.
    n_cores : int
        Number of parallel cores.

    Returns
    -------
    BinConversionResult
        DNA bin coordinates and RNA expression aggregated to bin resolution.
    """
    full_anno = load_gene_annotations("hg20")

    # Remove chromosome 24 (MT) if present
    dna = dna_mat[dna_mat["chrom"] != 24].copy().reset_index(drop=True)

    end = dna["chrompos"].values
    start = np.concatenate([[0], end[:-1]])

    # Identify expression columns (everything after annotation cols)
    # rna_mat has annotation columns first, then cell columns
    anno_cols = ["abspos", "chromosome_name", "start_position", "end_position",
                 "ensembl_gene_id", "hgnc_symbol", "band"]
    cell_cols = [c for c in rna_mat.columns if c not in anno_cols]
    rna_values = rna_mat[cell_cols].values
    gene_symbols = rna_mat["hgnc_symbol"].values if "hgnc_symbol" in rna_mat.columns else None

    # Map genes to bins
    ls_all: list[list[str]] = []
    for i in range(len(dna)):
        chrom = dna["chrom"].iloc[i]
        sub_anno = full_anno[full_anno["chromosome_name"] == chrom]
        cent_gene = 0.5 * (sub_anno["start_position"].values + sub_anno["end_position"].values)
        mask = (cent_gene <= end[i]) & (cent_gene >= start[i])
        genes_in_bin = sub_anno["hgnc_symbol"].values[mask].tolist()
        ls_all.append(genes_in_bin if genes_in_bin else [])

    # Convert: take median expression of genes in each bin
    def _aggregate_bin(i: int) -> np.ndarray | None:
        shared = [g for g in ls_all[i] if gene_symbols is not None and g in gene_symbols]
        if shared:
            idx = [j for j, g in enumerate(gene_symbols) if g in shared]
            return np.median(rna_values[idx, :], axis=0)
        return None

    if n_cores > 1:
        results = Parallel(n_jobs=n_cores)(
            delayed(_aggregate_bin)(i) for i in range(len(dna))
        )
    else:
        results = [_aggregate_bin(i) for i in range(len(dna))]

    # Build result matrix, tracking which bins have data
    rna_aj = np.zeros((len(dna), len(cell_cols)))
    valid_mask = np.zeros(len(dna), dtype=bool)

    for i, res in enumerate(results):
        if res is not None:
            rna_aj[i, :] = res
            valid_mask[i] = True

    # Fill missing bins with nearest valid neighbor
    if not all(valid_mask):
        valid_indices = np.where(valid_mask)[0]
        missing_indices = np.where(~valid_mask)[0]

        for mi in missing_indices:
            distances = np.abs(valid_indices - mi)
            nearest = valid_indices[np.argmin(distances)]
            rna_aj[mi, :] = rna_aj[nearest, :]

    # Build output DataFrame
    rna_adj = pd.DataFrame(rna_aj, columns=cell_cols)
    rna_adj.insert(0, "abspos", dna["abspos"].values)
    rna_adj.insert(0, "chrompos", dna["chrompos"].values)
    rna_adj.insert(0, "chrom", dna["chrom"].values)

    return BinConversionResult(dna_adj=dna, rna_adj=rna_adj)
