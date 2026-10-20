# Dataset Description and Analysis Handoff

**Project:** DFG — Uncertainty-Aware Deep Learning for Super-Resolution of Brain Microstructure Pathology in Diffusion MRI  
**Scope:** Stage 1 dataset inventory, acquisition characterization, quality control, and domain-shift analysis  
**Prepared for:** Human review and continuation with ChatGPT, Claude, or another analysis agent  
**Handoff date:** 2026-10-05  
**Analysis snapshot:** Results generated primarily on 2026-08-31 and 2026-09-01

## 1. Executive summary

This analysis characterizes paired HCP 3T/7T diffusion MRI as the healthy internal baseline and IDEAS as the external epilepsy/pathology cohort. HCP contains 170 subjects with matched 3T and 7T acquisitions. IDEAS contains 542 subject directories, but only 183 subjects contain both diffusion-weighted imaging (DWI) and anatomical data and are therefore eligible for the multimodal analysis.

One of those 183 subjects, `sub-4056`, does contain both DWI and anatomical data, but its checksum-valid DWI NIfTI file is gzip-malformed/truncated and cannot be decoded. It is retained in the dataset register and explicitly excluded from voxel-level analysis. The resulting QC-valid multimodal IDEAS cohort contains 182 subjects.

IDEAS must be treated as two distinct acquisition domains:

- **IDEAS-1:** raw b=1200 s/mm² only; 59 QC-valid subjects.
- **IDEAS-2:** raw b=300, 700, and 2500 s/mm²; 123 QC-valid subjects. `sub-4056` belongs to this protocol, giving 124 selected subjects before its QC exclusion.

The central finding is that meaningful acquisition-related shifts already exist between HCP 3T and HCP 7T. Against this internal baseline, both IDEAS protocols exhibit additional protocol, q-space, signal-to-noise, and normalized-signal shifts. IDEAS-1 generally has higher robust b0 SNR and normalized b=1000 attenuation closer to HCP 3T than IDEAS-2. IDEAS-1 and IDEAS-2 should therefore not be pooled as if they were a single acquisition protocol in primary analyses.

## 2. Dataset roles and cohort inventory

| Dataset/domain | Role | Subjects considered | QC-valid voxel analysis | DWI | Anatomy | Notes |
|---|---|---:|---:|---|---|---|
| HCP 3T | Healthy baseline/training domain | 170 | 170 | Yes | Yes | Paired subject identities with HCP 7T |
| HCP 7T | Healthy internal comparison | 170 | 170 | Yes | Yes | Same 170 subjects as HCP 3T |
| IDEAS full repository | External epilepsy/pathology dataset | 542 | Not used as one voxel cohort | 189 subjects | 536 subjects | Includes anatomy-only and DWI-only subjects |
| IDEAS multimodal selection | External multimodal cohort | 183 | 182 | Yes | Yes | `sub-4056` excluded after image-decode QC |
| IDEAS-1 | Raw b=1200-only IDEAS protocol | 59 | 59 | Yes | Yes | Harmonized to b=1000 only for the normalized-signal comparison |
| IDEAS-2 | Raw b=300/700/2500 IDEAS protocol | 124 | 123 | Yes | Yes | Includes excluded `sub-4056` before QC |

IDEAS subject availability across all 542 directories:

- 183 subjects have both DWI and anatomical data.
- 6 subjects have DWI but no anatomical data.
- 353 subjects have anatomical data but no DWI.
- 0 subjects have neither modality.

The six DWI-only subjects are outside the restricted 183-subject multimodal cohort. The current recommended external cohort for multimodal modeling is therefore the 182 QC-valid subjects, while preserving the 183-subject selection and the reason for the single exclusion in all provenance records.

## 3. Required naming convention

Use these labels consistently in reports, tables, filenames, and model metadata:

- **IDEAS-1:** the IDEAS protocol acquired at raw b=1200 s/mm² only.
- **IDEAS-2:** the IDEAS protocol acquired at raw b=300, 700, and 2500 s/mm².

Do not describe IDEAS-1 as a raw b=1000 protocol. Its signal is mapped to b=1000 only in the harmonized normalized-signal calculation. A pooled “IDEAS” result may be shown as a secondary summary, but protocol-stratified IDEAS-1 and IDEAS-2 results should be primary.

## 4. Data locations and retrieval status

### Windows source locations

- HCP 3T: `Z:\lohrd\MRI_data\HCP\Diffusion\3T`
- HCP 7T: `Z:\lohrd\MRI_data\HCP\Diffusion\7T`
- IDEAS: `Z:\lohrd\MRI_data\OpenNeuro\ds007401`

### Calcium source locations

- HCP 3T: `/home/david/mnt_anarchyNAS/lohrd/MRI_data/HCP/Diffusion/3T`
- HCP 7T: `/home/david/mnt_anarchyNAS/lohrd/MRI_data/HCP/Diffusion/7T`
- IDEAS: `/home/david/mnt_anarchyNAS/lohrd/MRI_data/OpenNeuro/ds007401`

### Results locations

- Local working results: `C:\Users\admin\Documents\ChatGPT\DFG_ Super-resolution in diffusion MRI\stage1_dataset_analysis`
- Canonical NAS results: `/home/david/mnt_anarchyNAS/lohrd/analysis_artifacts/dfg_super_resolution_dmri/stage1_dataset_analysis_20260831`

The IDEAS annex payload has already been retrieved. Retrieval used DataLad 1.4.1 and the existing git-annex 10.20260213 installation. The repository was clean after retrieval. Do not repeat the full download unless an explicit integrity check demonstrates missing content. The Calcium NAS path was verified as backed by `10.0.1.5:/mnt/ICNS/anarchy` at the time of analysis.

## 5. Acquisition characterization

| Domain | Field strength | Voxel dimensions | Principal nonzero shells (s/mm²) | Non-b0 directions | Median angular nearest-neighbor separation |
|---|---:|---|---|---:|---:|
| HCP 3T | 3 T | 1.25 mm isotropic | approximately 1000, 2000, 2995 | 270 | 5.2584° |
| HCP 7T | 7 T | approximately 1.05 mm isotropic | approximately 1000, 1995 | 128 | 8.1854° |
| IDEAS-1 | See register | See register | 1200 | See protocol table | See protocol table |
| IDEAS-2 | See register | See register | 300, 700, 2500 | See protocol table | See protocol table |

For shell classification, corrected b0 volumes are defined as b≤100 s/mm². This is necessary because HCP 7T encodes nominal b0 volumes around 60–65 s/mm²; a stricter threshold would misclassify them as diffusion-weighted volumes.

TE, TR, and phase-encoding information were absent or incomplete in the available harmonized metadata. These fields were not imputed and are not included in the protocol-distance calculation.

## 6. Inclusion and quality-control rules

### HCP

- The HCP 3T/7T intersection contains the same 170 subject identities.
- DWI, b-values, b-vectors, brain masks, and T1-weighted anatomy were present for all selected subjects.
- All 170 subjects were valid for voxel-level analysis in both acquisitions.
- Any training/validation/test split must be made at the subject level and kept identical across 3T and 7T to avoid identity leakage.

### IDEAS

- Multimodal eligibility requires both DWI and anatomical data.
- This filter selects exactly 183 subjects.
- `sub-4056` passes the modality-availability filter: it has both DWI and anatomy.
- Its DWI `.nii.gz` is nevertheless gzip-malformed/truncated and cannot be decoded, despite matching its recorded checksum.
- `sub-4056` is therefore retained in inventory/provenance tables with an explicit exclusion reason, but omitted from voxel-derived statistics.
- Final voxel-valid counts are 59 for IDEAS-1 and 123 for IDEAS-2, or 182 total.

## 7. Metrics and their interpretation

### 7.1 Cohort composition

The intended headline demographic comparison was an age standardized mean difference. It could not be computed because suitable subject-level age data were not available in the analyzed material. Sex was also unavailable. HCP is a healthy cohort, whereas IDEAS represents an epilepsy/pathology cohort, but this distinction is currently available only at the cohort level and cannot be separated from acquisition effects statistically.

### 7.2 Protocol distance

Protocol distance is a Gower distance over:

- magnetic field strength;
- voxel size in x, y, and z;
- voxel volume.

Gower distance scales mixed acquisition features into a common 0–1 comparison and averages their feature-wise differences. A distance of 0 indicates identical values for the included features; larger values indicate stronger acquisition separation. “Protocol similarity” is reported as `1 − distance`.

TE, TR, and phase-encoding values are not imputed. Their absence therefore does not silently create artificial similarity or dissimilarity. Distances depend on the comparison set used to establish feature ranges; the unified HCP/IDEAS report uses common scaling across its four protocol domains.

### 7.3 q-space discrepancy

The q-space discrepancy is the equal-weight mean of three components:

1. **Shell Jaccard discrepancy:** one minus the Jaccard overlap of nonzero shell sets.
2. **Relative direction-count difference:** the absolute difference in non-b0 direction counts, normalized by the larger count.
3. **Relative angular-NN-median difference:** the absolute difference between median angular nearest-neighbor spacings, normalized by the larger value.

The composite lies on an intuitive 0–1 scale: 0 represents agreement on all three summaries and larger values indicate more divergent q-space sampling. “q-space similarity” is `1 − discrepancy`. It is a summary of shell coverage, sampling density, and angular spacing; it is not a full comparison of every gradient direction.

### 7.4 Robust repeated-b0 SNR

For each subject, the repeated-b0 volumes are summarized voxelwise by their median. Within the analysis brain mask, the robust SNR statistic is:

`median repeated-b0 signal / (1.4826 × MAD across repeated-b0 measurements)`

HCP uses the provided brain masks. IDEAS uses an automated Otsu-derived mask because equivalent validated masks were not available. Group uncertainty was estimated using 10,000 percentile-bootstrap replicates with seed `20260831`.

This is a robust measure of repeated-b0 temporal signal stability, not a classical scanner SNR derived from a background-noise region. Raw intensities are not compared between datasets because scanner scaling and preprocessing make such values non-comparable.

### 7.5 Normalized signal shift

Tissue segmentation and RISH features were unavailable. The substitute is rotation-invariant direction-mean normalized attenuation, `S/S0`, where shell signals are averaged across directions and normalized to b0.

- HCP uses its actual b=1000 and b=2000 shells.
- IDEAS is compared at a harmonized b=1000 target using voxelwise mono-exponential fitting constrained through `S/S0=1` at b=0.
- IDEAS-1 uses interpolation from its raw b=1200 shell.
- IDEAS-2 uses limited extrapolation from its raw b=300 and b=700 shells; the b=2500 shell is not used in the ≤1500 fitting subset.

This measure is whole-brain normalized attenuation. It is not tissue-specific RISH, and the IDEAS-2 b=1000 value is more assumption-dependent than the IDEAS-1 interpolation.

## 8. Results

### 8.1 HCP internal baseline

The paired HCP comparison establishes how large a domain shift can arise from acquisition differences even within the same healthy subjects.

| Metric | HCP 3T | HCP 7T | Difference (7T−3T) | 95% bootstrap CI | Standardized/Wasserstein summary |
|---|---:|---:|---:|---|---|
| Protocol distance/similarity | — | — | 0.335695 / 0.664305 | — | — |
| q-space discrepancy/similarity | — | — | 0.544505 / 0.455495 | — | — |
| Robust b0 SNR, median | 19.469101 | 14.549609 | -5.009399 | [-5.524063, -4.488350] | dz=-1.929497; W=4.885961 |
| b=1000 normalized attenuation, median | 0.461173 | 0.511381 | +0.050142 | [0.049309, 0.051038] | dz=9.336129; W=0.050639 |
| b=2000 normalized attenuation, median | 0.266427 | 0.311895 | +0.044184 | [0.043396, 0.045401] | dz=8.384935; W=0.044927 |

Interpretation: HCP 3T and HCP 7T are not interchangeable domains. Field strength, spatial resolution, shell structure, angular sampling, robust b0 SNR, and normalized attenuation all differ. This paired shift is the appropriate internal baseline against which external IDEAS differences should be judged.

### 8.2 Protocol and q-space comparison: HCP versus IDEAS

| Comparison | Protocol distance | Protocol similarity | q-space discrepancy | q-space similarity |
|---|---:|---:|---:|---:|
| HCP 3T vs IDEAS-1 | 0.611673 | 0.388327 | 0.849262 | 0.150738 |
| HCP 3T vs IDEAS-2 | 0.593026 | 0.406974 | 0.736347 | 0.263653 |
| HCP 7T vs IDEAS-1 | 0.947368 | 0.052632 | 0.729871 | 0.270129 |
| HCP 7T vs IDEAS-2 | 0.928721 | 0.071279 | 0.518619 | 0.481381 |

Both IDEAS protocols are more distant from HCP 7T in the included spatial/field-strength protocol features. q-space comparisons are more nuanced: IDEAS-2 is the closer q-space match to both HCP acquisitions, especially HCP 7T, although “closer” does not mean equivalent.

### 8.3 Robust b0 SNR comparison

The reported difference is `IDEAS − HCP`.

| Comparison | HCP median | IDEAS median | Difference | 95% bootstrap CI | Hedges g | Wasserstein distance |
|---|---:|---:|---:|---|---:|---:|
| HCP 3T vs IDEAS-1 | 19.469101 | 17.741886 | -1.727215 | [-4.046528, -0.255564] | -0.275320 | 3.670636 |
| HCP 3T vs IDEAS-2 | 19.469101 | 13.298759 | -6.170341 | [-6.800584, -5.495198] | -2.439573 | 6.177982 |
| HCP 7T vs IDEAS-1 | 14.549609 | 17.741886 | +3.192277 | [0.837709, 4.730301] | 1.045578 | 4.367410 |
| HCP 7T vs IDEAS-2 | 14.549609 | 13.298759 | -1.250849 | [-1.977987, -0.517123] | -0.558854 | 1.370644 |

IDEAS-1 has higher robust b0 SNR than IDEAS-2. Relative to HCP 3T, both are lower, but the IDEAS-1 difference is much smaller. Relative to HCP 7T, the direction reverses for IDEAS-1: it is higher than HCP 7T, whereas IDEAS-2 remains lower. Thus no single pooled IDEAS SNR shift accurately describes both protocols.

### 8.4 Harmonized b=1000 normalized attenuation

The reported difference is `IDEAS − HCP`.

| Comparison | HCP median | IDEAS median | Difference | 95% bootstrap CI | Hedges g | Wasserstein distance |
|---|---:|---:|---:|---|---:|---:|
| HCP 3T vs IDEAS-1 | 0.461173 | 0.447764 | -0.013409 | [-0.018883, -0.009632] | -1.491936 | 0.019545 |
| HCP 3T vs IDEAS-2 | 0.461173 | 0.427525 | -0.033648 | [-0.036397, -0.030180] | -2.871388 | 0.035543 |
| HCP 7T vs IDEAS-1 | 0.511381 | 0.447764 | -0.063617 | [-0.070141, -0.060282] | -5.233633 | 0.070185 |
| HCP 7T vs IDEAS-2 | 0.511381 | 0.427525 | -0.083856 | [-0.087293, -0.080884] | -6.821467 | 0.086183 |

IDEAS-1 is closer than IDEAS-2 to HCP 3T and HCP 7T for the harmonized b=1000 attenuation measure. Both IDEAS protocols have lower normalized attenuation than both HCP acquisitions. Interpretation must retain the harmonization caveat: IDEAS-1 is interpolated from b=1200, while IDEAS-2 requires extrapolation beyond b=700.

## 9. Practical implications for modeling

1. Treat HCP 3T, HCP 7T, IDEAS-1, and IDEAS-2 as separate acquisition domains.
2. Use the paired HCP 3T/7T differences as an internal baseline for acquisition-related variability.
3. Keep IDEAS-1 and IDEAS-2 stratified in primary evaluation; pooled IDEAS results are secondary only.
4. Condition models or harmonization methods on field strength, voxel geometry, shell structure, and direction count where possible.
5. Split HCP data by subject identity across both acquisitions to prevent the same individual appearing in training and testing through different scans.
6. Use the 182 QC-valid multimodal IDEAS subjects for voxel-level external evaluation.
7. Preserve `sub-4056` in registers and flow diagrams as “DWI + anatomy present, voxel analysis excluded: malformed/truncated DWI image.”
8. Avoid raw-intensity comparisons across datasets; use normalized or within-acquisition quantities.
9. If pathology effects are discussed, state that cohort and acquisition differences are confounded in the present data.

## 10. Storage requirements

### Full IDEAS checkout

- 542 subjects.
- 3,327 tracked files.
- 1,347 annex paths representing 1,346 unique annex keys.
- Logical tracked size: 20,459,260,882 bytes = 20.459 GB = 19.054 GiB = 0.020459 TB.
- Unique annex payload: 20,388,408,469 bytes = 18.988 GiB.
- Present checkout content: 20,389,910,995 bytes = 18.990 GiB.

### Restricted 183-subject multimodal selection

- 1,423 tracked files.
- 527 annex paths representing 526 unique annex keys.
- Logical tracked size: 11,922,933,370 bytes = 11.923 GB = 11.104 GiB.
- This is 58.276% of the full logical checkout size.
- Unique annex payload: 11,852,792,922 bytes = 11.039 GiB.
- Present checkout content: 11,853,583,483 bytes = 11.040 GiB.

The “all subject files plus shared metadata” estimate equals the “session-1 anatomy plus DWI plus shared metadata” estimate for the selected subjects. Shared metadata includes `CHANGES`, `README.md`, and `dataset_description.json`.

## 11. Limitations

- Subject-level age and sex were unavailable for the intended composition analysis.
- TE, TR, and phase-encoding metadata were absent or incomplete and were not imputed.
- IDEAS lacked equivalent validated brain/tissue masks; an Otsu-derived brain mask was used.
- Tissue-specific RISH features could not be calculated.
- The replacement `S/S0` statistic is whole-brain and does not isolate white matter or other tissue classes.
- IDEAS-1 b=1000 harmonization is an interpolation from b=1200; IDEAS-2 is a limited mono-exponential extrapolation from b=300/700.
- The q-space discrepancy summarizes rather than fully represents the gradient tables.
- Robust b0 SNR measures repeated-b0 stability and should not be presented as classical background-noise SNR.
- Pathology and acquisition domain are confounded between HCP and IDEAS.
- HCHS was deferred and is not part of this analysis snapshot.

## 12. Artifact map

All relative paths below are within `stage1_dataset_analysis`.

### Human-readable reports

- `report_hcp_ideas_domain_shift.md` — latest unified HCP baseline and HCP-versus-IDEAS report.
- `report_ideas_183.md` — restricted 183-subject multimodal IDEAS report.
- `report_ideas_protocol_strata.md` — separate IDEAS protocol analysis.
- `report.md` — full Stage 1 report.

### Core machine-readable tables

- `dataset_register.csv` — dataset/protocol-level register.
- `subject_availability_inclusion.csv` — modality availability and eligibility.
- `inclusion_exclusion_log.csv` — explicit subject inclusion/exclusion provenance.
- `hcp_ideas_domain_shift_summary.csv` — unified comparison metrics.
- `ideas_183_protocol_stratum_similarity.csv` — IDEAS protocol-stratified similarity results.
- `hcp_subject_voxel_metrics.csv` — HCP subject-level voxel metrics.
- `ideas_183_subject_voxel_metrics.csv` — IDEAS multimodal subject-level voxel metrics.
- `protocol_subject_metrics.csv` — subject/protocol acquisition metrics.
- `ideas_storage_capacity.csv` and `ideas_storage_capacity.json` — capacity estimates.

### Reproducibility scripts

- `run_stage1.py`
- `calcium_storage_analysis.py`
- `calcium_refresh_qspace.py`
- `calcium_voxel_analysis.py`
- `calcium_ideas_voxel_analysis.py`
- `calcium_finalize_comparisons.py`
- `filter_ideas_183_analysis.py`
- `analyze_ideas_protocol_strata.py`
- `generate_hcp_ideas_domain_shift_report.py`

### Provenance and identifiers

- `execution.log`
- `config.json`
- `ideas_git_index_audit.txt`
- `hcp_3t_7t_intersection.csv`
- `ideas_183_dwi_anat_subject_ids.txt`

## 13. Continuation instructions for ChatGPT, Claude, or another agent

Use this section as the operational context for subsequent work.

### Fixed facts that should not be silently changed

- HCP contains 170 paired 3T/7T subjects.
- The IDEAS multimodal selection contains 183 subjects before voxel QC and 182 after QC.
- `sub-4056` contains both DWI and anatomical data, belongs to IDEAS-2, and is excluded only because its DWI image cannot be decoded.
- IDEAS-1 means raw b=1200 only; IDEAS-2 means raw b=300/700/2500.
- b≤100 s/mm² is the corrected b0 definition.
- IDEAS-1 and IDEAS-2 must be analyzed separately in primary results.
- The IDEAS annex payload is already retrieved.
- HCHS is outside the present scope.

### Before extending the analysis

1. Read `report_hcp_ideas_domain_shift.md`, `dataset_register.csv`, `subject_availability_inclusion.csv`, and `inclusion_exclusion_log.csv`.
2. Verify that the relevant source paths and NAS mount are live; do not assume a previously mounted network filesystem is still connected.
3. Treat source datasets as read-only and write derived products only to the results area.
4. Confirm the IDEAS Git/DataLad repository is clean before and after any content operation.
5. Do not alter NAS ownership or permissions; existing ownership was `nobody:UNKNOWN`.
6. Use no more than four CPU threads unless the execution environment is explicitly reconfigured. No GPU was required for the current Stage 1 analysis.
7. Preserve bootstrap seed `20260831` when reproducing the published summary values.
8. If changing metric definitions, write new outputs rather than silently overwriting the current analysis snapshot.

### Recommended next analyses

- Add demographic composition statistics if reliable subject-level age/sex metadata become available.
- Obtain or generate validated tissue segmentations and calculate tissue-specific RISH or related rotationally invariant features.
- Test sensitivity of IDEAS-2 b=1000 estimates to alternative signal models and extrapolation rules.
- Evaluate models separately on IDEAS-1 and IDEAS-2, with a secondary pooled estimate only when scientifically justified.
- Quantify whether domain adaptation reduces the shifts relative to the paired HCP acquisition baseline.
- Add HCHS only as a separately documented extension with its own inclusion and protocol characterization.

## 14. Short reusable context block

The following paragraph can be pasted into a new AI conversation:

> Continue the Stage 1 diffusion-MRI dataset analysis using `DATA_DESCRIPTION_HANDOFF.md` and the artifacts in `stage1_dataset_analysis`. HCP has 170 subjects paired across 3T and 7T. IDEAS has 183 subjects with both DWI and anatomy, but `sub-4056` (an IDEAS-2 subject) has a malformed/truncated DWI NIfTI and is excluded from voxel analysis, leaving 182 subjects. Use the labels IDEAS-1 for the raw b=1200-only protocol (59 QC-valid) and IDEAS-2 for the raw b=300/700/2500 protocol (123 QC-valid); do not pool them in primary analyses. Treat data sources as read-only, preserve all exclusion provenance, use b≤100 as b0, and retain bootstrap seed 20260831 when reproducing existing results. Read the unified domain-shift report and core CSV tables before making changes.

