# Implementation workplan for flexible diffusion MRI super resolution

Revised 5 October 2026. Scope: a 9–12 month thesis within DFG WP1, using explicit 3D feature maps to reconstruct requested spatial grids and gradient tables. 

## Data and sampling

Start with HCP3T b≈1000, then extend to acquired multi-shell data. Keep participants/families, paired field strengths and derivatives in one split; preserve locked test subjects. HCP7T transfer uses its own acquired reference. Keep IDEAS-1 and IDEAS-2 as separate external domains, with zero-shot evaluation primary and adaptation on disjoint subjects. Retain sub-4056 as QC-excluded. HCHS remains deferred until access, cohort, endpoints and permitted data/compute paths are documented. Acquisition/cohort differences are not evidence of pathology effects.

Select **6–30 DWI measurements across all shells**, with **one to at most one b0 per six DWI measurements**. Other combinations of uneven DWIs >6 may be allowed as well:

| DWI measurements | Allowed b0 images |
|---|---|
| 6 | 1 |
| 12 | 1–2 |
| 18 | 1–3 |
| 24 | 1–4 |
| 30 | 1–5 |

For intermediate counts, use `1 <= N_b0 <= floor(N_DWI / 6)`. Sample available images without replacement; do not duplicate b0 images. Vary allowed counts during training and freeze explicit combinations for evaluation. A direction measured at another b-value consumes another DWI measurement. Output counts are independent of the input budget. The maximum 35 volumes require 266 s at TR=7.6 s before overhead; verify total acquisition time for each protocol.

## Pair generation and preprocessing

- Generate LR inputs with frozen blur/voxel integration followed by downsampling. Start with identity and isotropic 2×; add anisotropic factors after geometry checks.
- Select coverage-aware angular subsets and an allowed b0 subset. Classify b0 using b≤100 s/mm², retaining actual b-values. Supervise withheld acquired directions; report observed targets separately. Missing-shell synthesis is a separate extension.
- Normalize all signals with one scalar derived only from selected input b0 images: the positive brain-voxel 95th percentile of their mean, using an input-derived brain mask. Preserve relative shell scaling and restore units at inference.
- Extract physically matched LR/HR crops. Record source/split IDs, hashes, selected indices, affines, masks, PSF, grid conventions and seeds. Native 1.25 mm references support 2.5→1.25 mm validation; interpolating targets to 1.0 mm does not create acquired 1.0 mm truth.

## Models A, B and C

Inputs have shape `[B,N,1,D,H,W]`, with b-values, gradients, `valid` and `is_b0` masks; pad only to the batch maximum, at most 35 measurements. Outputs are specified by the target gradient table and spatial grid.

Use a shared residual 3D U-Net with widths **16/32/64** at LR, LR/2 and LR/4. Each scale has two residual blocks with 3×3×3 convolutions and SiLU; use stride-2 convolutions between scales. Start without CNN normalization. Decode with trilinear resize, skip concatenation and residual blocks. Metadata include b/3000, the six independent entries of `ggᵀ`, b0 status and input/output spacing, preserving g/−g symmetry.

Encode every selected b0 image, masked-average its features at each scale, and fuse this information into every target reconstruction. B/C additionally use local b0 features in their queries. b0 receives no directional encoding.

| Model | Implementation | Purpose |
|---|---|---|
| **A: CNN baseline** | A two-layer metadata MLP (hidden width 64) maps input/target b-values and angular relationships to channel-wise modulation. Transform image features, apply SiLU, then masked-average DWI features for each target. Fuse b0 features and decode. | Variable-input/output baseline |
| **B: CNN + q-space attention** | Retain A’s backbone and fine-scale pooling. At LR/4 use two q-space self-attention blocks per spatial location followed by target cross-attention: 64 channels, four heads, FFN expansion 2. Queries combine target metadata and local b0/input features. | Test q-space interactions |
| **C: spatial + q-space attention** | Retain B’s q-space blocks. Replace coarse spatial residual blocks with two 3D window-attention blocks per measurement: 4³ windows, shift 2, four heads. Keep fine-scale CNN processing. | Test added spatial attention |

For A: `u_ij = SiLU((1 + gamma_ij) * Conv1x1(F_i) + beta_ij)`, followed by masked averaging over real DWI inputs i for each target j. Avoid acquisition-index embeddings and attention between output targets, preserving input-order and output-chunk invariance.

**Masks:** sanitize padding to finite zeros, re-mask after biased/residual layers, divide pooled sums by valid counts, and exclude padded attention keys with additive −∞ bias. Mask invalid queries and target losses; reject empty valid pools. Keep spatial-support masks separate.

## Upsampling

**Default: trilinear resize plus Conv3d.** Resize decoder features to the requested shape with `align_corners=False`, then apply `Conv3d(16,8,3,padding=1) → SiLU → Conv3d(8,1,3,padding=1)`. Use the same head across A/B/C. Begin with aligned integer-factor grids.

Two optional head comparisons:

- **Transposed convolution:** for 2×, use `ConvTranspose3d(16,8,kernel_size=4,stride=2,padding=1)`, then SiLU and a 3×3×3 output convolution. Audit alignment and periodic artifacts; configure other factors separately.
- **3D pixel shuffle:** for 2×, project to `8 × 2³ = 64` channels, rearrange into eight feature channels on the doubled 3D grid, then SiLU and an output convolution. Implement volumetric rearrangement explicitly; standard PyTorch PixelShuffle is two-dimensional. Test impulse placement and factor-specific heads.

For later nonaligned grids, use `v_in = inverse(A_in) @ A_out @ v_out`, accounting for crop origins and feature strides, and sample features before convolutional refinement. Mask out-of-support locations. Output dimensions alone do not specify physical alignment or establish generalization to unseen scales.

## Training and validation

Use matched 3D crops, varied angular/b0 subsets, input-order permutations and proper 90° rotations on compatible grids. Rotate images, gradients, masks and degradation geometry consistently. Add bounded input noise after the clean baseline; avoid elastic deformation and independent per-image/shell intensity changes.

Train with masked signal L1, balanced across subjects and shells, plus LR consistency: reconstruct acquired input directions, apply the frozen degradation operator and compare with input measurements. Select the consistency weight on validation data. Starting optimizer: AdamW, learning rate 1e-4, weight decay 1e-4. Match examples, update budgets and checkpoint criteria across models.

For 48 GB GPUs, start with batch size 1, LR 32³ patches, 1–2 target directions and mixed precision. Profile a complete forward/backward pass at 30 DWI + 5 b0, including consistency predictions; aim below approximately 40 GB reserved memory. At inference, cache encoder features, decode targets in chunks and stitch overlapping patches.

Acceptance tests: count constraints, input-only normalization, padding/permutation invariance, g/−g symmetry, gradient rotations, affine landmarks, finite gradients and target-order/chunk invariance. Test one b0 and every allowed maximum. Evaluate subject-level signal MAE/nRMSE, FA/MD, identifiable orientation errors, selected tract overlap, memory and runtime. Compare interpolation plus regularized SH, A/B/C, and sequential versus joint SR. Report missing-shell synthesis separately.

## Work packages
 Timing is relative to thesis start.

| Period | Deliverables | Acceptance gate | DFG mapping |
|---|---|---|---|
| Months 1–2 | Grouped splits, deterministic degradation, variable DWI/b0 sampler, normalization and nonlearned baseline | Budget, leakage, provenance and geometry checks pass | M1.1/M1.2 |
| Months 2–3 | Model A, masked pooling, b0 fusion and default resize–Conv3d head | Invariance tests, small development fit and maximum-input memory profile pass | M1.1/M1.2 |
| Months 4–5 | Model B, frozen training settings and bounded head comparison | Common A/B conditions; complete forward/backward fits memory budget | M1.2/M1.3 |
| Months 6–7 | A/B evaluation across direction counts, b0 counts and spatial/angular tasks | Subject-level results and costs; no test-based tuning | M1.3 |
| Months 7–9 | Model C if A/B are stable; acquired-shell conditioning and gradient-metadata ablation | Attention effects isolated; core thesis results complete | M1.1–M1.4 |
| Months 9–12 | Optional HCP7T transfer or missing-shell study; writing and reproducibility package | Code, configurations, manifests, operator hashes and evaluation outputs documented | M1.3; optional WP2 bridge |

For nine months, prioritize A/B, bound C to one matched comparison and omit optional extensions. Organize implementation into data/sampling/degradation, encoder, pooling/attention, decoder/heads, losses, inference and evaluation modules. Record software versions, seeds and source/split/config hashes.

The submitted proposal and project handoffs remain controlling. Execution requires explicit authorization and read-only remote identity, NAS, task-file and Git checks. Retain manual GPU selection, A100 exclusion and existing CPU limits.
