"""Smoke test for copykat._biobabel.

Exercises the real copykat() -> heatmap3() path end-to-end on a tiny,
offline, synthetic UMI matrix (built from real hg20 gene symbols so
annotate_genes finds matches). No network access, no GUI, no persistent
files -- the heatmap is rendered but never saved or displayed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import copykat


def _tiny_rawmat(n_cells: int = 20, genes_per_chrom: int = 1000) -> pd.DataFrame:
    # >= ~1600 genes needed: cna_mcmc's candidate-breakpoint count is capped at
    # ~(n_genes/win_size), and copykat() requires >= 25 breakpoints or raises.
    anno = copykat.load_gene_annotations("hg20")
    genes: list[str] = []
    for chrom in (1, 2):
        chrom_genes = (
            anno.loc[anno["chromosome_name"] == chrom, "hgnc_symbol"]
            .dropna()
            .unique()
            .tolist()[:genes_per_chrom]
        )
        genes.extend(chrom_genes)

    rng = np.random.RandomState(0)
    counts = rng.poisson(lam=5, size=(len(genes), n_cells)) + 1  # dense, all > 0
    return pd.DataFrame(counts, index=genes, columns=[f"cell{i}" for i in range(n_cells)])


def main() -> None:
    rawmat = _tiny_rawmat()

    result = copykat.copykat(
        rawmat=rawmat,
        id_type="S",
        cell_line="no",
        ngene_chr=5,
        min_gene_per_cell=50,
        win_size=25,
        KS_cut=0.1,
        distance="euclidean",
        genome="hg20",
        n_cores=1,
    )
    assert "copykat.pred" in result.prediction.columns
    assert "chrom" in result.CNAmat.columns

    cell_cols = [c for c in result.CNAmat.columns if c not in ("chrom", "chrompos", "abspos")]
    mat = result.CNAmat[cell_cols].values.T
    heatmap_result = copykat.heatmap3(mat, row_cluster=True, col_cluster=False, show=False)
    assert "row_order" in heatmap_result

    print("copykat smoke test passed:", result.prediction.shape[0], "cells,",
          result.CNAmat.shape[0], "bins")


if __name__ == "__main__":
    main()
