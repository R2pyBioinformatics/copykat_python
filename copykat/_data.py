"""Data loaders for bundled annotation resources (R source: sysdata.rda)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = ["load_gene_annotations", "load_dna_bins", "load_cycle_genes", "load_example_data"]

_RESOURCES = Path(__file__).resolve().parent / "resources"


def load_gene_annotations(genome: str = "hg20") -> pd.DataFrame:
    """Load gene annotation table for the specified genome.

    Parameters
    ----------
    genome : str
        Genome build identifier. ``"hg20"`` for human hg38 annotations,
        ``"mm10"`` for mouse mm10 annotations.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: abspos, chromosome_name, start_position,
        end_position, ensembl_gene_id, hgnc_symbol (or mgi_symbol), band.
    """
    if genome in ("hg20", "hg38"):
        path = _RESOURCES / "full_anno_hg38.parquet"
    elif genome == "mm10":
        path = _RESOURCES / "full_anno_mm10.parquet"
    else:
        raise ValueError(f"Unsupported genome: {genome!r}. Use 'hg20' or 'mm10'.")
    return pd.read_parquet(path)


def load_dna_bins() -> pd.DataFrame:
    """Load 220KB genomic bin coordinates for hg38.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: chrom, chrompos, abspos.
        12205 rows covering all autosomes and X chromosome.
    """
    return pd.read_parquet(_RESOURCES / "DNA_hg20.parquet")


def load_cycle_genes() -> list[str]:
    """Load the list of cell cycle gene symbols to exclude from analysis.

    Returns
    -------
    list[str]
        List of 1316 cell cycle gene symbols.
    """
    df = pd.read_parquet(_RESOURCES / "cyclegenes.parquet")
    return df["x"].tolist()


def load_example_data() -> pd.DataFrame:
    """Load the example breast tumor UMI count matrix.

    Uses three-tier resolution: local staging → cache → registry download.

    Returns
    -------
    pd.DataFrame
        UMI count matrix with genes as rows and cells as columns.
        Index is gene symbols.
    """
    from ._download import resolve_data_path

    path = resolve_data_path("exp_rawdata.tsv.gz")
    df = pd.read_csv(path, sep="\t", index_col=0)
    return df
