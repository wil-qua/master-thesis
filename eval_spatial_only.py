# eval_spatial_only.py
import os
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from dipy.core.gradients import gradient_table
from dipy.reconst.csdeconv import auto_response_ssst

import config
from dataset import (
    FlexibleDiffusionSampler,
    calculate_normalization_scalar,
    generate_and_augment_lr_patch,
)
from microstructure_inference import (
    create_3d_sinusoidal_weight,
    compute_dipy_microstructure,
    compute_fodf,
    compute_masked_mae,
)
from spatial_only_model import Model_SpatialOnlySR


@torch.no_grad()
def predict_volume_spatial_only(
    model,
    lr_dwi,
    lr_b0=None,
    downscale_factor=2,
    lr_patch_size=32,
    lr_stride=16,
    dir_batch_size=16,
    device="cuda",
):
    """
    Sliding-Window-Inferenz mit 3D-Sinus-Gewichtung (sin^2, 50% Overlap).
    Verarbeitet mehrere Richtungen pro Patch-Position gebatcht auf der GPU.
    lr_dwi: [X_lr, Y_lr, Z_lr, N_dirs]
    lr_b0:  [X_lr, Y_lr, Z_lr, N_b0] oder None
    """
    model.eval()
    X_lr, Y_lr, Z_lr, n_dirs = lr_dwi.shape
    X_hr, Y_hr, Z_hr = X_lr * downscale_factor, Y_lr * downscale_factor, Z_lr * downscale_factor
    hr_patch_size = lr_patch_size * downscale_factor

    lr_dwi_t = np.transpose(lr_dwi, (3, 0, 1, 2)).astype(np.float32)  # [N_dirs, X, Y, Z]
    lr_b0_mean = np.mean(lr_b0, axis=-1).astype(np.float32) if lr_b0 is not None else None

    xs = list(range(0, max(1, X_lr - lr_patch_size + 1), lr_stride))
    if (X_lr - lr_patch_size) not in xs and X_lr >= lr_patch_size:
        xs.append(X_lr - lr_patch_size)
    ys = list(range(0, max(1, Y_lr - lr_patch_size + 1), lr_stride))
    if (Y_lr - lr_patch_size) not in ys and Y_lr >= lr_patch_size:
        ys.append(Y_lr - lr_patch_size)
    zs = list(range(0, max(1, Z_lr - lr_patch_size + 1), lr_stride))
    if (Z_lr - lr_patch_size) not in zs and Z_lr >= lr_patch_size:
        zs.append(Z_lr - lr_patch_size)

    patch_weight = create_3d_sinusoidal_weight(hr_patch_size, min_weight=1e-3)  # [P, P, P]
    vol_sum = np.zeros((X_hr, Y_hr, Z_hr, n_dirs), dtype=np.float32)
    vol_weight = np.zeros((X_hr, Y_hr, Z_hr), dtype=np.float32)

    for x in xs:
        for y in ys:
            for z in zs:
                dwi_patch = lr_dwi_t[:, x:x + lr_patch_size, y:y + lr_patch_size, z:z + lr_patch_size]  # [N, P, P, P]
                if lr_b0_mean is not None:
                    b0_patch = lr_b0_mean[x:x + lr_patch_size, y:y + lr_patch_size, z:z + lr_patch_size]
                    b0_patch_rep = np.broadcast_to(b0_patch[None, ...], dwi_patch.shape)
                    in_batch = np.stack([dwi_patch, b0_patch_rep], axis=1)  # [N, 2, P, P, P]
                else:
                    in_batch = dwi_patch[:, None, ...]  # [N, 1, P, P, P]

                preds = []
                for start in range(0, n_dirs, dir_batch_size):
                    sub = torch.from_numpy(in_batch[start:start + dir_batch_size].copy()).to(device)
                    out = model(sub).squeeze(1).cpu().numpy()  # [b, P_hr, P_hr, P_hr]
                    preds.append(out)

                pred_all = np.concatenate(preds, axis=0)  # [N, P_hr, P_hr, P_hr]
                pred_all = np.transpose(pred_all, (1, 2, 3, 0))  # [P_hr, P_hr, P_hr, N]

                hx, hy, hz = x * downscale_factor, y * downscale_factor, z * downscale_factor
                vol_sum[hx:hx + hr_patch_size, hy:hy + hr_patch_size, hz:hz + hr_patch_size] += (
                    pred_all * patch_weight[..., None]
                )
                vol_weight[hx:hx + hr_patch_size, hy:hy + hr_patch_size, hz:hz + hr_patch_size] += patch_weight

    return vol_sum / np.maximum(vol_weight[..., None], 1e-8)


def main():
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("Results/SpatialOnly", exist_ok=True)

    USE_B0 = True
    RUN_90_DIR_REFERENCE = True  # Optional: Spatial-Only zusätzlich über alle 90 LR-Richtungen
    MODEL_PATH = "Models/model_spatial_only_best.pth"

    # 1. Exakt derselbe Validierungs-Proband wie in microstructure_inference.py
    nifti_path = config.VAL_NIFTI_FILES[0]
    bval_path = config.VAL_BVAL_FILES[0]
    bvec_path = config.VAL_BVEC_FILES[0]
    mask_path = config.VAL_MASK_FILES[0]

    bvals = np.loadtxt(bval_path).flatten()
    bvecs = np.loadtxt(bvec_path).T
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T

    mask_img = nib.load(mask_path)
    brain_mask_full = np.asanyarray(mask_img.dataobj) > 0
    affine = mask_img.affine
    full_data = np.asanyarray(nib.load(nifti_path).dataobj, dtype=np.float32)

    # 2. Exakt derselbe 30/60-Split wie bei Model A
    sampler = FlexibleDiffusionSampler(
        bvals, bvecs, config.B0_THRESHOLD, config.MIN_DWI_B, config.MAX_DWI_B
    )
    rng = np.random.RandomState(42)
    sample_spec = sampler.sample(
        num_targets=60, plot=False, rng=rng, n_dwi=30, p_b0_target=0.0
    )

    b0_full = full_data[..., sampler.global_b0_indices]
    dwi_full = full_data[..., sampler.global_dwi_indices]

    norm_scalar = calculate_normalization_scalar(b0_full[..., 0], brain_mask_full)
    b0_norm = b0_full / norm_scalar
    dwi_norm = dwi_full / norm_scalar

    # Gerade Dimensionen für Downscale-Faktor 2 (identisch zu microstructure_inference.py)
    max_x = (brain_mask_full.shape[0] // config.DOWNSCALE_FACTOR) * config.DOWNSCALE_FACTOR
    max_y = (brain_mask_full.shape[1] // config.DOWNSCALE_FACTOR) * config.DOWNSCALE_FACTOR
    max_z = (brain_mask_full.shape[2] // config.DOWNSCALE_FACTOR) * config.DOWNSCALE_FACTOR
    sl = (slice(0, max_x), slice(0, max_y), slice(0, max_z))

    target_dwi_indices = [idx for _, idx in sample_spec["target_vols_info"]]
    hr_mask = brain_mask_full[sl]
    hr_b0_in = b0_norm[sl + (sample_spec["b0_indices"],)]
    hr_dwi_in30 = dwi_norm[sl + (sample_spec["input_indices"],)]
    hr_dwi_gt60 = dwi_norm[sl + (target_dwi_indices,)]

    # 3. Identische LR-Degradation (noise_std=0.01)
    dummy_bvecs_b0 = np.zeros((hr_b0_in.shape[-1], 3))
    lr_b0_in, _, _ = generate_and_augment_lr_patch(
        hr_b0_in, hr_mask, dummy_bvecs_b0, rot_spec=None,
        downscale_factor=config.DOWNSCALE_FACTOR, noise_std=0.01
    )
    lr_dwi_in30, _, _ = generate_and_augment_lr_patch(
        hr_dwi_in30, hr_mask, sample_spec["input_bvecs"], rot_spec=None,
        downscale_factor=config.DOWNSCALE_FACTOR, noise_std=0.01
    )

    # 4. Spatial-Only-Modell laden & auf den 30 LR-Input-Richtungen inferieren
    model = Model_SpatialOnlySR(in_channels=2 if USE_B0 else 1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Modell geladen: {MODEL_PATH}")

    print("\nStarte Spatial-Only Inferenz (30 Input-Richtungen, 50% Overlap, sin^2)...")
    hr_dwi_spatial30 = predict_volume_spatial_only(
        model=model,
        lr_dwi=lr_dwi_in30,
        lr_b0=lr_b0_in if USE_B0 else None,
        downscale_factor=config.DOWNSCALE_FACTOR,
        lr_patch_size=config.PATCH_SIZE // config.DOWNSCALE_FACTOR,
        lr_stride=(config.PATCH_SIZE // config.DOWNSCALE_FACTOR) // 2,
        device=device,
    )

    # Trilineare LR-30 Baseline zum direkten Vergleich
    lr_t = torch.from_numpy(np.transpose(lr_dwi_in30, (3, 0, 1, 2))).unsqueeze(0).float()
    hr_dwi_trilin30 = np.transpose(
        F.interpolate(lr_t, scale_factor=config.DOWNSCALE_FACTOR, mode="trilinear", align_corners=False)
        .squeeze(0).numpy(),
        (1, 2, 3, 0),
    )

    # 5. Denormalisierung in physikalische Einheiten
    hr_b0_gt_phys = hr_b0_in * norm_scalar
    hr_dwi_in30_phys = hr_dwi_in30 * norm_scalar
    hr_dwi_gt60_phys = hr_dwi_gt60 * norm_scalar
    hr_dwi_gt90_phys = np.concatenate([hr_dwi_in30_phys, hr_dwi_gt60_phys], axis=-1)
    hr_dwi_spatial30_phys = hr_dwi_spatial30 * norm_scalar
    hr_dwi_trilin30_phys = hr_dwi_trilin30 * norm_scalar

    bvals_30 = sample_spec["input_bvals"]
    bvecs_30 = sample_spec["input_bvecs"]
    bvals_60 = sample_spec["target_bvals"]
    bvecs_60 = sample_spec["target_bvecs"]
    bvals_90 = np.concatenate([bvals_30, bvals_60])
    bvecs_90 = np.vstack([bvecs_30, bvecs_60])
    b0_bvals = sample_spec["b0_bvals"]
    b0_bvecs = np.zeros((len(b0_bvals), 3), dtype=np.float32)

    # 6. DTI (FA, MD) berechnen
    print("\nBerechne DTI-Karten...")
    dti_gt90 = compute_dipy_microstructure(hr_dwi_gt90_phys, hr_b0_gt_phys, bvals_90, bvecs_90, b0_bvals, hr_mask)
    dti_gt30 = compute_dipy_microstructure(hr_dwi_in30_phys, hr_b0_gt_phys, bvals_30, bvecs_30, b0_bvals, hr_mask)
    dti_trilin30 = compute_dipy_microstructure(hr_dwi_trilin30_phys, hr_b0_gt_phys, bvals_30, bvecs_30, b0_bvals, hr_mask)
    dti_spatial30 = compute_dipy_microstructure(hr_dwi_spatial30_phys, hr_b0_gt_phys, bvals_30, bvecs_30, b0_bvals, hr_mask)

    # 7. CSD-fODFs (sh_order=4, feste Response Function aus HR-GT 90)
    gtab_90 = gradient_table(np.concatenate([b0_bvals, bvals_90]), np.vstack([b0_bvecs, bvecs_90]))
    gtab_30 = gradient_table(np.concatenate([b0_bvals, bvals_30]), np.vstack([b0_bvecs, bvecs_30]))

    vol_gt90 = np.clip(np.concatenate([hr_b0_gt_phys, hr_dwi_gt90_phys], axis=-1), 0.0, None)
    vol_gt30 = np.clip(np.concatenate([hr_b0_gt_phys, hr_dwi_in30_phys], axis=-1), 0.0, None)
    vol_trilin30 = np.clip(np.concatenate([hr_b0_gt_phys, hr_dwi_trilin30_phys], axis=-1), 0.0, None)
    vol_spatial30 = np.clip(np.concatenate([hr_b0_gt_phys, hr_dwi_spatial30_phys], axis=-1), 0.0, None)

    print("\nSchätze feste Response Function auf HR-GT (90 Richtungen)...")
    response_gt90, ratio = auto_response_ssst(gtab_90, vol_gt90, roi_radii=10, fa_thr=0.7)
    print(f"Response Function Ratio: {ratio:.3f}")

    sh_order = 4
    fodf_gt90 = compute_fodf(vol_gt90, gtab_90, hr_mask, response_gt90, sh_order=sh_order)
    fodf_gt30 = compute_fodf(vol_gt30, gtab_30, hr_mask, response_gt90, sh_order=sh_order)
    fodf_trilin30 = compute_fodf(vol_trilin30, gtab_30, hr_mask, response_gt90, sh_order=sh_order)
    fodf_spatial30 = compute_fodf(vol_spatial30, gtab_30, hr_mask, response_gt90, sh_order=sh_order)

    results = [
        ("HR-GT (30 Dir, Obergrenze)", dti_gt30, fodf_gt30),
        ("LR-Trilinear (30 Dir)", dti_trilin30, fodf_trilin30),
        ("Spatial-Only (30 Dir)", dti_spatial30, fodf_spatial30),
    ]

    # Optional: Model A (60 Dir) aus Results/DWI_UNet.nii.gz direkt mit einreihen
    unet_path = "Results/DWI_UNet.nii.gz"
    if os.path.exists(unet_path):
        print("\nLade vorhandenes Model A (Results/DWI_UNet.nii.gz) für direkten 90-Dir-GT-Vergleich...")
        hr_dwi_unet_phys = np.asanyarray(nib.load(unet_path).dataobj, dtype=np.float32)
        gtab_60 = gradient_table(np.concatenate([b0_bvals, bvals_60]), np.vstack([b0_bvecs, bvecs_60]))
        vol_unet = np.clip(np.concatenate([hr_b0_gt_phys, hr_dwi_unet_phys], axis=-1), 0.0, None)
        dti_unet = compute_dipy_microstructure(hr_dwi_unet_phys, hr_b0_gt_phys, bvals_60, bvecs_60, b0_bvals, hr_mask)
        fodf_unet = compute_fodf(vol_unet, gtab_60, hr_mask, response_gt90, sh_order=sh_order)
        results.append(("Model A (60 infer. Dir)", dti_unet, fodf_unet))

    # Optional: Spatial-Only über alle 90 LR-Richtungen als Referenz
    if RUN_90_DIR_REFERENCE:
        print("\nStarte optionale Referenz: Spatial-Only auf allen 90 LR-Richtungen...")
        hr_dwi_90_norm = np.concatenate([hr_dwi_in30, hr_dwi_gt60], axis=-1)
        lr_dwi_90, _, _ = generate_and_augment_lr_patch(
            hr_dwi_90_norm, hr_mask, bvecs_90, rot_spec=None,
            downscale_factor=config.DOWNSCALE_FACTOR, noise_std=0.01
        )
        hr_dwi_spatial90 = predict_volume_spatial_only(
            model=model,
            lr_dwi=lr_dwi_90,
            lr_b0=lr_b0_in if USE_B0 else None,
            downscale_factor=config.DOWNSCALE_FACTOR,
            lr_patch_size=config.PATCH_SIZE // config.DOWNSCALE_FACTOR,
            lr_stride=(config.PATCH_SIZE // config.DOWNSCALE_FACTOR) // 2,
            device=device,
        )
        hr_dwi_spatial90_phys = hr_dwi_spatial90 * norm_scalar
        vol_spatial90 = np.clip(np.concatenate([hr_b0_gt_phys, hr_dwi_spatial90_phys], axis=-1), 0.0, None)
        dti_spatial90 = compute_dipy_microstructure(hr_dwi_spatial90_phys, hr_b0_gt_phys, bvals_90, bvecs_90, b0_bvals, hr_mask)
        fodf_spatial90 = compute_fodf(vol_spatial90, gtab_90, hr_mask, response_gt90, sh_order=sh_order)
        results.append(("Spatial-Only (90 Dir, Ref)", dti_spatial90, fodf_spatial90))
        nib.save(nib.Nifti1Image(fodf_spatial90, affine), "Results/SpatialOnly/fODF_SpatialOnly_90.nii.gz")

    # 8. Quantitative Auswertung
    print("\n================ SIGNAL MAE (30 Input-Richtungen, normiert) ================")
    print(f"LR-Trilinear (30 Dir) : {compute_masked_mae(hr_dwi_trilin30, hr_dwi_in30, hr_mask):.5f}")
    print(f"Spatial-Only (30 Dir) : {compute_masked_mae(hr_dwi_spatial30, hr_dwi_in30, hr_mask):.5f}")

    wm = hr_mask & (dti_gt90["FA"] > 0.4)
    print("\n================ MIKROSTRUKTUR MAE vs. HR-GT (90 RICHTUNGEN) ================")
    print(f"{'Methode':<28} | {'FA (Brain)':>10} | {'FA (WM)':>10} | {'MD*1e3':>10} | {'fODF':>10}")
    print("-" * 80)
    for label, dti_res, fodf_res in results:
        fa_all = compute_masked_mae(dti_res["FA"], dti_gt90["FA"], hr_mask)
        fa_wm = compute_masked_mae(dti_res["FA"], dti_gt90["FA"], wm)
        md_all = compute_masked_mae(dti_res["MD"], dti_gt90["MD"], hr_mask) * 1e3
        fodf_all = compute_masked_mae(fodf_res, fodf_gt90, hr_mask)
        print(f"{label:<28} | {fa_all:10.5f} | {fa_wm:10.5f} | {md_all:10.5f} | {fodf_all:10.5f}")
    print("=============================================================================")

    # NIfTI-Export für Visualisierung
    nib.save(nib.Nifti1Image(dti_spatial30["FA"].astype(np.float32), affine), "Results/SpatialOnly/FA_SpatialOnly_30.nii.gz")
    nib.save(nib.Nifti1Image(dti_spatial30["MD"].astype(np.float32), affine), "Results/SpatialOnly/MD_SpatialOnly_30.nii.gz")
    nib.save(nib.Nifti1Image(fodf_spatial30, affine), "Results/SpatialOnly/fODF_SpatialOnly_30.nii.gz")
    nib.save(nib.Nifti1Image(hr_dwi_spatial30_phys.astype(np.float32), affine), "Results/SpatialOnly/DWI_SpatialOnly_30.nii.gz")
    print("\nErgebnisse gespeichert unter: Results/SpatialOnly/")


if __name__ == "__main__":
    main()