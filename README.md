# scSHEFT

A Novel Framework for Label Transfer from scRNA-seq to scATAC-seq

## Overview

We propose a new label transfer tool scSHEFT, which simultaneously considers gene expression count data, peak count data, and Gene Activity Scores as inputs to bridge the gap of heterogeneous features. Specifically, we transform scATAC-seq data into Gene Activity Scores based on prior knowledge to harmonize heterogeneous features. As the feature transformation would result in information loss, we introduce the raw ATAC-seq embeddings to preserve the original information. To achieve a balance between inter-omics alignment and intra-omics heterogeneity, scSHEFT employs an anchor-based approach for aligning inter-omics anchor pairs, while using a contrastive-based strategy to preserve the cellular heterogeneity within each omics layer.

## Installation

First, clone this repository:

```bash
git clone https://github.com/CSUBioGroup/scSHEFT.git
cd scSHEFT/
```

Then install the dependencies (must be installed in order, as some packages will install related dependencies):

```bash
pip install seaborn==0.13.2
pip install scanpy==1.10.3
pip install numpy==1.26.3
pip install torch==1.13.0
pip install annoy==1.17.3
```

Or install all dependencies at once:

```bash
pip install -r requirements.txt
```

## Hardware Requirements

All experiments were run on a workstation with:
- **CPU**: Intel Core i9-10980XE
- **GPU**: NVIDIA GeForce RTX 3090 (24GB VRAM)
- **Runtime**: Approximately 100-1000 seconds depending on dataset size
- **Memory Usage**: 4000-40000 MB depending on dataset complexity

## Datasets

The following datasets were used for evaluation:

- **SNARE-seq**: [GSE126074](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE126074)
- **SHARE-seq**: [GSE140203](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140203)
- **CITE-ASAP**: [GSE156478](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156478)
- **10x Multiome**: [10x B Lymphoma - Multi-Omic Integration](https://www.10xgenomics.com/datasets/multiomic-integration-neuroscience-application-note-single-cell-multiome-rna-atac-alzheimers-disease-mouse-model-brain-coronal-sections-from-one-hemisphere-over-a-time-course-1-standard)
- **T-cell bone marrow & The CD34 bone marrow**: [GSE200046](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE200046)
- **PBMC**: [Zenodo](https://zenodo.org/records/7555405)

## Usage

### Quick Start
- **demo_data**: [T-cell bone marrow](https://drive.google.com/drive/folders/1S4T-W9jkaLj4lehSYRCNmTV080g4d_Ar?usp=drive_link) \
Make sure data and code is available for running.

#### Data Preparation

Before running the demo, you need to prepare the following data files:

1. **Raw ATAC-seq data**: Raw peak count matrix (`.h5ad` format)
2. **Processed ATAC-seq data**: Gene Activity Scores (`.h5ad` format) 
3. **RNA-seq data**: Gene Expression Matrix (`.h5ad` format)
4. **LSI embedding**: Latent Semantic Indexing embedding for ATAC-seq (`.npy` format)
5. **Structural PCA embedding**: PCA embedding for structural information (`.npy` format)

#### File Structure Example

```
data/
├── bm_multiome_atac.h5ad    # Raw ATAC-peak data
├── bm-GAM.h5ad              # Processed ATAC-gene data (Gene Activity Scores)
├── bm-GEM.h5ad              # RNA-seq data (Gene Expression Matrix)
├── bm_lsi.npy               # LSI embedding for ATAC-seq
└── bm_struct_pca.npy        # Structural PCA embedding
```

#### Running the Demo

1. **Edit the demo script**: Open `demo.py` and replace the placeholder strings with your actual file paths:

```python
# Example path configuration
exp_id = "bm"  # Your experiment ID
adata_raw_target = sc.read_h5ad("./data/bm_multiome_atac.h5ad")
adata_target = sc.read_h5ad("./data/bm-GAM.h5ad")
adata_source = sc.read_h5ad("./data/bm-GEM.h5ad")
target_raw_emb = np.load("./data/bm_lsi.npy")
struct_target_emb = np.load("./data/bm_struct_pca.npy")
```

2. **Run the demo**:

```bash
python demo.py
```

**Important Note**: Setting `hvg_num` may cause unstable results. It's recommended to use the default settings for optimal performance.

### Default Parameters

#### Hyperparameter Sensitivity Analysis

We performed comprehensive hyperparameter sensitivity analysis to guide users in selecting optimal weights for different loss components. The key hyperparameters correspond to:

- **α₁ (anchor_w = 0.5)**: Anchor inter-alignment weight for anchor loss
- **α₂ (center_w = 5)**: Center inter-alignment weight for center loss  
- **β₁ (ce_w = 1)**: Classification-guided intra-alignment weight for cross-entropy loss
- **β₂ (cont_w = 0.1)**: Contrast intra-alignment weight for InfoNCE loss




## File Structure

```
scSHEFT/
├── README.md             # This file
├── scSHEFT.py            # Main model implementation
├── models.py             # Neural network architectures
├── losses.py             # Loss functions
├── utils.py              # Utility functions
├── dataset.py            # Data loading and processing
├── demo.py               # Demo script
└── requirements.txt      # Python dependencies
```