"""Tests for data loading modules (R source: sysdata.rda, exp.rawdata.rda)."""

import numpy as np
import pandas as pd
import pytest

from copykat._data import load_cycle_genes, load_dna_bins, load_example_data, load_gene_annotations


class TestLoadGeneAnnotations:
    def test_hg20_shape(self):
        df = load_gene_annotations("hg20")
        assert df.shape == (56051, 7)

    def test_hg20_columns(self):
        df = load_gene_annotations("hg20")
        expected = ["abspos", "chromosome_name", "start_position", "end_position",
                    "ensembl_gene_id", "hgnc_symbol", "band"]
        assert list(df.columns) == expected

    def test_mm10_shape(self):
        df = load_gene_annotations("mm10")
        assert df.shape == (137030, 7)

    def test_mm10_has_mgi_symbol(self):
        df = load_gene_annotations("mm10")
        assert "mgi_symbol" in df.columns

    def test_invalid_genome(self):
        with pytest.raises(ValueError, match="Unsupported genome"):
            load_gene_annotations("hg19")


class TestLoadDnaBins:
    def test_shape(self):
        df = load_dna_bins()
        assert df.shape == (12205, 3)

    def test_columns(self):
        df = load_dna_bins()
        assert list(df.columns) == ["chrom", "chrompos", "abspos"]


class TestLoadCycleGenes:
    def test_length(self):
        genes = load_cycle_genes()
        assert len(genes) == 1316

    def test_type(self):
        genes = load_cycle_genes()
        assert isinstance(genes, list)
        assert isinstance(genes[0], str)


class TestLoadExampleData:
    def test_shape(self):
        df = load_example_data()
        assert df.shape == (33694, 302)

    def test_index_is_genes(self):
        df = load_example_data()
        assert df.index.name == "gene"
