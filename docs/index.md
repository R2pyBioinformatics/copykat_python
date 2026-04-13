# copykat_py

Python port of the R **copykat** package (v1.1.0) for inference of genomic copy number and subclonal structure of human tumors from high-throughput single cell RNA-seq data.

CopyKAT uses integrative Bayesian approaches to identify genome-wide aneuploidy at 5MB resolution in single cells, separating tumor (aneuploid) cells from normal (diploid) cells, and identifying tumor subclones.

**Citation**: Gao, R., et al. (2021). *Nature Biotechnology*.

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import copykat

# Load example breast tumor scRNAseq data (33694 genes x 302 cells)
rawdata = copykat.load_example_data()

# Run CopyKAT
result = copykat.copykat(
    rawmat=rawdata,
    id_type="S",
    cell_line="no",
    ngene_chr=5,
    win_size=25,
    KS_cut=0.2,
    distance="euclidean",
    n_cores=4,
)

# Access results
predictions = result.prediction       # cell.names, copykat.pred (diploid/aneuploid)
cna_matrix = result.CNAmat            # Copy number at 220KB bin resolution
clustering = result.hclustering       # Hierarchical clustering linkage matrix

# View predictions
print(predictions["copykat.pred"].value_counts())
```

## Supported Genomes

- **hg20** (hg38): Human genome with 56,051 gene annotations
- **mm10**: Mouse genome with 137,030 gene annotations

## Key Functions

- `copykat()` — Main analysis pipeline
- `annotate_genes()` — Gene coordinate annotation
- `baseline_gmm()` / `baseline_norm_cl()` — Diploid cell detection
- `cna_mcmc()` — MCMC segmentation
- `convert_to_bins()` — Gene-to-bin conversion
- `heatmap3()` — CNA heatmap visualization
- `load_example_data()` — Example dataset loader
