# Gene Editing Knowledge Base — Build Report

**Version**: `GEA_v2026_Q1`  
**Date**: 2026-02-26 22:17:44  
**Builder**: `scripts/build_literature_db.py`

## Summary
| Metric | Value |
|--------|-------|
| Initial DB size | 1,543 |
| Records fetched this run | 135,317 |
| New records added | 85,056 |
| **Final DB size** | **86,599** |
| Gold Standard coverage | 86.0% (43/50) |

## Source Breakdown
| Source | Records Fetched |
|--------|----------------|
| ClinicalTrials | 556 |
| EuropePMC | 77,472 |
| PubMed | 57,289 |

## Tier Breakdown
| Tier | Fetched | Added |
|------|---------|-------|
| T1_Core_HighCite | 15,000 | 14,912 |
| T2_Broad_PubMed | 9,848 | 9,074 |
| T3_NovelCas | 8,000 | 6,970 |
| T4_BaseEditing | 9,947 | 6,520 |
| T5_PrimeEditing | 5,979 | 3,856 |
| T6_RNAEditing | 9,831 | 7,925 |
| T7_Delivery | 15,008 | 7,855 |
| T8_Safety | 11,925 | 2,488 |
| T9_Clinical | 5,154 | 849 |
| T9_CT_gov | 332 | 93 |
| T10_Epigenome | 9,959 | 6,551 |
| T11_Diagnostics | 3,000 | 1,554 |
| T12_GeneDrive_Agri | 3,000 | 2,282 |
| T13_Screens | 9,946 | 5,652 |
| T14_Computational | 4,824 | 1,945 |
| T15_Transposon_Retron | 2,000 | 1,726 |
| T16_Preprints | 4,472 | 2,614 |
| T17_Disease_sickle_cell_disease | 765 | 398 |
| T17_Disease_beta-thalassemia | 481 | 76 |
| T17_Disease_Duchenne_muscular_dy | 486 | 221 |
| T17_Disease_cystic_fibrosis | 990 | 219 |
| T17_Disease_hereditary_angioedem | 23 | 0 |
| T17_Disease_TTR_amyloidosis | 113 | 38 |
| T17_Disease_hypercholesterolemia | 344 | 125 |
| T17_Disease_retinal_dystrophy | 94 | 24 |
| T17_Disease_hemophilia | 219 | 66 |
| T17_Disease_HIV | 989 | 386 |
| T17_Disease_cancer_immunotherapy | 994 | 145 |
| T17_Disease_alpha-1_antitrypsin | 987 | 196 |
| T17_Disease_Huntington | 607 | 296 |

## Missing Gold Standard Papers
- `10.1126/science.aax9249`
- `10.1038/s41586-024-07998-8`
- `10.1038/s41586-020-2008-9`
- `10.1126/science.aah5297`
- `10.1038/s41586-019-1048-8`
- `10.1038/s41592-019-0549-5`
- `10.1038/s41576-019-0128-9`

## Quality Notes
- Deduplication: DOI (normalized) + Title hash
- Sources: PubMed NCBI, Europe PMC, ClinicalTrials.gov
- Enrichment: Semantic Scholar citation counts, PubMed metadata