# scSHEFT
A Novel Framework for Label Transfer from scRNA-seq to scATAC-seq
# Overview

We propose a new label transfer tool scSHEFT, which simultaneously considers gene expression count data, peak count data, and Gene Activity Scores as inputs to bridge the gap of heterogeneous features. Specifically, we transform scATAC-seq data into Gene Activity Scores based on prior knowledge to harmonize heterogeneous features. As the feature transformation would result in information loss, we introduce the raw ATAC-seq embeddings to preserve the original information. To achieve a balance between inter-omics alignment and intra-omics heterogeneity, scSHEFT employs an anchor-based approach for aligning inter-omics anchor pairs, while using a contrastive-based strategy to preserve the cellular heterogeneity within each omics layer. 
# Installation
First, clone this repository.
```
git clone https://github.com/CSUBioGroup/scSHEFT.git
cd scSHEFT/
```
Then install the dependencies (must in order, some packages will install related package):
```
seaborn == 0.13.2
scanpy == 1.10.3
numpy == 1.26.3
torch == 1.13.0
annoy == 1.17.3
```
```
pip install -r requirements.txt
```

# Datasets
- **SNARE_seq**: [GSE126074](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE126074)
- **Ma-2020**: [GSE140203](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140203)
- **CITE-ASAP**: [GSE156478](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156478)
- **10x B Lymphoma - Multi-Omic Integration**: [10x B Lymphoma - Multi-Omic Integration](https://www.10xgenomics.com/datasets/multiomic-integration-neuroscience-application-note-single-cell-multiome-rna-atac-alzheimers-disease-mouse-model-brain-coronal-sections-from-one-hemisphere-over-a-time-course-1-standard)
- **T-cell bone marrow & The CD34 bone marrow**: [GSE200046](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE200046)
- **PBMC_vaccine**: [Zenodo](https://zenodo.org/records/7555405)



# Demo
- **demo_data**: [T-cell bone marrow](https://drive.google.com/drive/quota) \
Make sure data and code is availavle for running.  \
The **default parameters** are set within the script demo_bm.py, and can be executed with the command:
```
python demo_bm.py
```
**Note that:** set *hvg_num* will cause unstable results.

