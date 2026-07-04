---
name: use-copykat
description: Use copykat to infer copy number aberrations (CNA) and classify
  tumor (aneuploid) vs normal (diploid) cells from raw single-cell RNA-seq
  UMI count data.
---

# copykat

Python port of the R **copykat** package (Gao et al., *Nature Biotechnology*
2021). Given a raw gene-by-cell UMI count matrix, `copykat.copykat()` is a
**single self-contained call** -- not a multi-step state pipeline you drive
yourself -- that returns a `CopykatResult` with per-cell diploid/aneuploid
predictions and a genome-wide copy number matrix (220KB bins for human hg38,
gene resolution for mouse mm10). Internally it chains gene annotation, DLM
smoothing, diploid-baseline detection, MCMC segmentation, and (for hg20)
gene-to-bin conversion; those internal building blocks (`annotate_genes`,
`dlm_smooth`, `baseline_norm_cl`/`baseline_gmm`/`baseline_synthetic`,
`cna_mcmc`, `convert_to_bins`) are also exported publicly for advanced/custom
pipelines but are not meant to be re-chained to reproduce `copykat()` itself.

**Use when**: you have a raw (unnormalized) scRNA-seq UMI matrix and want to
separate tumor cells from stromal/normal cells, quantify aneuploidy, or
visualize a genome-wide CNA heatmap.

**Do not use when**: the input is already normalized/log-transformed, the
genome build is not hg38(hg20)/mm10, or you need copy number from bulk
RNA-seq/WGS/WES (use a DNA-based CNV caller instead).

## Entry points

- `copykat.copykat(rawmat, ...)` -- the main pipeline.
- `copykat.heatmap3(x, ...)` -- render the CNA heatmap (call separately on
  `result.CNAmat`).
- `copykat.load_example_data()` -- bundled example dataset (downloads/caches
  from Zenodo on first use).
- `copykat.load_gene_annotations(genome)` / `copykat.load_dna_bins()` /
  `copykat.load_cycle_genes()` -- bundled reference data, no network needed.

## Quick reference

```python
import copykat

rawdata = copykat.load_example_data()
result = copykat.copykat(
    rawmat=rawdata, id_type="S", cell_line="no",
    ngene_chr=5, win_size=25, KS_cut=0.2,
    distance="euclidean", n_cores=4,
)
result.prediction  # cell.names, copykat.pred (diploid/aneuploid/...)
result.CNAmat      # chrom, chrompos, abspos + one column per cell
```

For more: `biobabel.describe_package(import_name="copykat")`.
