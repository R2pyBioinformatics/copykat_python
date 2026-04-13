"""Gene annotation with genomic coordinates (R source: annotateGenes.hg20.R, annotateGenes.mm10.R)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._data import load_gene_annotations

__all__ = ["annotate_genes"]


def annotate_genes(
    mat: pd.DataFrame,
    id_type: str = "S",
    genome: str = "hg20",
) -> pd.DataFrame:
    """Annotate genes in an expression matrix with genomic coordinates.

    Parameters
    ----------
    mat : pd.DataFrame
        Expression matrix with gene IDs as row index and cells as columns.
    id_type : str
        Gene ID type: ``"S"`` for gene symbols (HGNC/MGI),
        ``"E"`` for Ensembl IDs.
    genome : str
        Genome build: ``"hg20"`` for human hg38, ``"mm10"`` for mouse.

    Returns
    -------
    pd.DataFrame
        Annotated matrix with annotation columns prepended
        (abspos, chromosome_name, start_position, end_position,
        ensembl_gene_id, hgnc_symbol/mgi_symbol, band) followed
        by expression columns.
    """
    full_anno = load_gene_annotations(genome)

    if genome == "mm10":
        symbol_col = "mgi_symbol"
    else:
        symbol_col = "hgnc_symbol"

    if id_type.upper().startswith("E"):
        id_col = "ensembl_gene_id"
    else:
        id_col = symbol_col

    # Find shared gene IDs
    gene_ids = mat.index.astype(str)
    shared = gene_ids.intersection(full_anno[id_col].astype(str))
    mat_sub = mat.loc[shared].copy()

    # Get matching annotations
    anno = full_anno[full_anno[id_col].isin(shared)].copy()
    anno = anno.drop_duplicates(subset=[symbol_col])

    # Reorder annotations to match expression matrix order
    anno_lookup = anno.set_index(id_col)
    anno_ordered = anno_lookup.loc[mat_sub.index].reset_index()

    # Ensure symbol column is named correctly
    if anno_ordered.columns[0] != id_col:
        anno_ordered = anno_ordered.rename(columns={anno_ordered.columns[0]: id_col})

    # Combine annotation with expression data
    mat_sub_reset = mat_sub.reset_index(drop=True)
    anno_reset = anno_ordered.reset_index(drop=True)

    result = pd.concat([anno_reset, mat_sub_reset], axis=1)
    return result
