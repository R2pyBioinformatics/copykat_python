"""End-to-end CopyKAT CNA inference + heatmap, on tiny synthetic data.

Maps to workflow ``copykat.cna_inference_and_heatmap``:

1. (workflow step ``copykat.load_example_data``) -- replaced here by a tiny
   synthetic raw UMI matrix built from real hg20 gene symbols, so the script
   runs offline and fast. Substitute ``copykat.load_example_data()`` (or your
   own raw count matrix) for real analyses.
2. ``copykat.copykat`` -- run the pipeline to get per-cell predictions and a
   CNA matrix.
3. ``copykat.heatmap3`` -- render the CNA heatmap, colored by chromosome
   (columns) and by prediction (rows).

Uses a handful of real gene symbols and ~40 synthetic cells so it runs in
seconds without network access, a GUI, or any file left behind (the heatmap
is saved to a temporary directory and not displayed).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import copykat


def _make_synthetic_rawmat(n_cells: int = 40, genes_per_chrom: int = 1200) -> pd.DataFrame:
    """Build a small, dense raw UMI matrix using real hg20 gene symbols.

    Restricted to two chromosomes with a dense (all-nonzero) count matrix so
    copykat's per-chromosome contiguous-coverage cell filter (``ngene_chr``)
    is trivially satisfied without needing many more genes/cells. Needs
    >= ~1600 genes total: cna_mcmc's candidate-breakpoint count is capped at
    ~(n_genes/win_size), and copykat() requires >= 25 breakpoints or raises.
    """
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
    cell_names = [f"cell{i}" for i in range(n_cells)]
    return pd.DataFrame(counts, index=genes, columns=cell_names)


def main() -> dict:
    rawmat = _make_synthetic_rawmat()

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

    cell_cols = [c for c in result.CNAmat.columns if c not in ("chrom", "chrompos", "abspos")]
    mat = result.CNAmat[cell_cols].values.T  # cells x bins

    chr_vals = result.CNAmat["chrom"].values.astype(int) % 2
    chr_colors = np.array(["black" if c == 0 else "grey" for c in chr_vals])

    pred_dict = dict(zip(result.prediction["cell.names"], result.prediction["copykat.pred"]))
    pred_colors = np.array([
        "#D95F02" if "aneuploid" in pred_dict.get(c, "") else
        "#1B9E77" if "diploid" in pred_dict.get(c, "") else "grey"
        for c in cell_cols
    ])

    with tempfile.TemporaryDirectory() as tmp:
        heatmap_result = copykat.heatmap3(
            mat,
            row_cluster=True,
            col_cluster=False,
            dist_func="euclidean",
            link_method="ward.D2",
            col_side_colors=chr_colors,
            row_side_colors=pred_colors,
            cmap="RdBu_r",
            dendrogram_="row",
            figsize=(6, 4),
            save_path=str(Path(tmp) / "cna_heatmap.png"),
            show=False,
        )

    return {
        "prediction": result.prediction,
        "cna_matrix": result.CNAmat,
        "heatmap_row_order": heatmap_result["row_order"],
    }


if __name__ == "__main__":
    out = main()
    print(out["prediction"]["copykat.pred"].value_counts())
    print(f"CNA matrix: {out['cna_matrix'].shape[0]} bins x {out['cna_matrix'].shape[1]} columns")
