"""Tests for gene annotation (R source: annotateGenes.hg20.R, annotateGenes.mm10.R)."""

import numpy as np
import pandas as pd
import pytest

from copykat.annotate import annotate_genes
from copykat._data import load_example_data


@pytest.fixture
def small_mat():
    """Small expression matrix with known gene symbols."""
    genes = ["TP53", "BRCA1", "EGFR", "MYC", "NOTREAL123"]
    cells = ["cell1", "cell2", "cell3"]
    data = np.random.RandomState(42).randint(0, 10, (len(genes), len(cells)))
    return pd.DataFrame(data, index=genes, columns=cells)


class TestAnnotateGenes:
    def test_basic_hg20(self, small_mat):
        result = annotate_genes(small_mat, id_type="S", genome="hg20")
        assert "hgnc_symbol" in result.columns
        assert "abspos" in result.columns
        # NOTREAL123 should be dropped
        assert result.shape[0] <= small_mat.shape[0]
        assert result.shape[0] >= 1

    def test_annotation_columns_present(self, small_mat):
        result = annotate_genes(small_mat, id_type="S", genome="hg20")
        for col in ["abspos", "chromosome_name", "start_position", "end_position",
                     "ensembl_gene_id", "band"]:
            assert col in result.columns

    def test_cell_columns_preserved(self, small_mat):
        result = annotate_genes(small_mat, id_type="S", genome="hg20")
        for cell in ["cell1", "cell2", "cell3"]:
            assert cell in result.columns

    def test_larger_dataset(self):
        rawdata = load_example_data()
        subset = rawdata.iloc[:1000, :5]
        result = annotate_genes(subset, id_type="S", genome="hg20")
        assert result.shape[0] > 0
        assert "hgnc_symbol" in result.columns
