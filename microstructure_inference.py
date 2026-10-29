# microstructure_reference.py
"""
Vollständige Evaluierungs- und Mikrostruktur-Pipeline.
Vergleicht Ground Truth (HR), U-Net (Model A) und Regularized SH Baseline
anhand von Signal-Metriken sowie DIPY-Mikrostrukturkarten (FA, MD, DKI).
"""

"""
Das U-Net (Modell A) mittelt im Bottleneck durch die masked_average-Operation alle unterschiedlichen Eingaberichtungen pro Voxel
zu einem einzigen Durchschnittswert zusammen und löscht damit die Information, in welcher Raumrichtung das Signal stark oder schwach war.
Dadurch sagt der Decoder für alle gesuchten Zielwinkel nahezu dasselbe, richtungslose (isotrope) Durchschnittsbild voraus,
weshalb richtungsabhängige Werte wie die Fractional Anisotropy (FA) und die fODFs (fiber orientation density function) komplett einbrechen.
"""

import os
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import dipy.reconst.dti as dti
import dipy.reconst.dki as dki
from dipy.core.gradients import gradient_table
from dipy.reconst.csdeconv import auto_response_ssst, ConstrainedSphericalDeconvModel

import config
from dataset import (
    FlexibleDiffusionSampler,
    calculate_normalization_scalar,
    generate_and_augment_lr_patch,
    compute_ggT
)
from train import ModelA_FlexibleSR
from sh_reference import RegularizedSHBaseline


# ===========================================================
# 1. INFERENCE HILFSFUNKTIONEN (Sliding Window Stitching)
# ===========================================================

@torch.no_grad()
def predict_volume_unet(
    model,
    lr_b0,
    lr_dwi,
    input_bvals,
    input_bvecs,
    target_bvals,
    target_bvecs,
    max_measurements,
    max_b0,
    downscale_factor=2,
    lr_patch_size=32,
    lr_stride=8,
    device="cuda"
):
    """
    Rekonstruiert ein komplettes 4D-Zielvolumen mit dem fertig trainierten U-Net durch überlappende
    (um die halbe Patch-Größe) 3D-Patches (Sliding Window Stitching) für jede Zielrichtung.
    lr_b0:  [X_lr, Y_lr, Z_lr, N_b0]
    lr_dwi: [X_lr, Y_lr, Z_lr, N_dwi]
    """
    model.eval()
    X_lr, Y_lr, Z_lr, n_dwi = lr_dwi.shape
    n_b0 = lr_b0.shape[-1]
    n_targets = len(target_bvals)

    X_hr = X_lr * downscale_factor
    Y_hr = Y_lr * downscale_factor
    Z_hr = Z_lr * downscale_factor
    hr_patch_size = lr_patch_size * downscale_factor

    # Globale Input-Tensoren (Reihenfolge: [N, X, Y, Z]) vorbereiten
    lr_dwi_t = np.transpose(lr_dwi, (3, 0, 1, 2))
    lr_b0_t = np.transpose(lr_b0, (3, 0, 1, 2))

    valid_mask_dwi = torch.zeros((1, max_measurements), dtype=torch.bool, device=device)
    valid_mask_dwi[0, :n_dwi] = True

    valid_mask_b0 = torch.zeros((1, max_b0), dtype=torch.bool, device=device)
    valid_mask_b0[0, :n_b0] = True

    # Koordinaten-Raster für Sliding Window
    xs = list(range(0, max(1, X_lr - lr_patch_size + 1), lr_stride))
    if (X_lr - lr_patch_size) not in xs and X_lr >= lr_patch_size:
        xs.append(X_lr - lr_patch_size)
    ys = list(range(0, max(1, Y_lr - lr_patch_size + 1), lr_stride))
    if (Y_lr - lr_patch_size) not in ys and Y_lr >= lr_patch_size:
        ys.append(Y_lr - lr_patch_size)
    zs = list(range(0, max(1, Z_lr - lr_patch_size + 1), lr_stride))
    if (Z_lr - lr_patch_size) not in zs and Z_lr >= lr_patch_size:
        zs.append(Z_lr - lr_patch_size)

    hr_predictions = np.zeros((X_hr, Y_hr, Z_hr, n_targets), dtype=np.float32)

    for t_idx in range(n_targets):
        # Metadaten für Target t_idx aufbauen
        t_bval = target_bvals[t_idx] / 3000.0
        t_ggT = compute_ggT(target_bvecs[t_idx])

        meta_ij = np.zeros((max_measurements, 14), dtype=np.float32)
        for i in range(n_dwi):
            in_bval = input_bvals[i] / 3000.0
            in_ggT = compute_ggT(input_bvecs[i])
            meta_ij[i] = np.concatenate([[in_bval], in_ggT, [t_bval], t_ggT])

        meta_tensor = torch.from_numpy(meta_ij).unsqueeze(0).to(device)  # [1, M, 14]

        vol_sum = np.zeros((X_hr, Y_hr, Z_hr), dtype=np.float32)
        vol_weight = np.zeros((X_hr, Y_hr, Z_hr), dtype=np.float32)

        for x in xs:
            for y in ys:
                for z in zs:
                    # LR-Patch ausschneiden & padden
                    padded_dwi = np.zeros((max_measurements, lr_patch_size, lr_patch_size, lr_patch_size), dtype=np.float32)
                    padded_dwi[:n_dwi] = lr_dwi_t[:, x:x+lr_patch_size, y:y+lr_patch_size, z:z+lr_patch_size]

                    padded_b0 = np.zeros((max_b0, lr_patch_size, lr_patch_size, lr_patch_size), dtype=np.float32)
                    padded_b0[:n_b0] = lr_b0_t[:, x:x+lr_patch_size, y:y+lr_patch_size, z:z+lr_patch_size]

                    x_dwi_tensor = torch.from_numpy(padded_dwi).unsqueeze(0).unsqueeze(2).to(device)
                    x_b0_tensor = torch.from_numpy(padded_b0).unsqueeze(0).unsqueeze(2).to(device)

                    # U-Net Forward Pass
                    pred_patch = model(x_dwi_tensor, x_b0_tensor, meta_tensor, valid_mask_dwi, valid_mask_b0)
                    pred_np = pred_patch.squeeze().cpu().numpy()  # [hr_patch, hr_patch, hr_patch]

                    hx, hy, hz = x * downscale_factor, y * downscale_factor, z * downscale_factor
                    vol_sum[hx:hx+hr_patch_size, hy:hy+hr_patch_size, hz:hz+hr_patch_size] += pred_np
                    vol_weight[hx:hx+hr_patch_size, hy:hy+hr_patch_size, hz:hz+hr_patch_size] += 1.0

        hr_predictions[..., t_idx] = vol_sum / np.maximum(vol_weight, 1.0)
        print(f"  U-Net Richtung [{t_idx+1}/{n_targets}] rekonstruiert.")

    return hr_predictions


def predict_volume_sh(lr_dwi, input_bvecs, target_bvecs, downscale_factor=2, lambda_reg=0.005, device="cuda"):
    """
    Rekonstruiert das 4D-Zielvolumen mit der RegularizedSHBaseline (Spherical Harmonics).
    lr_dwi: [X_lr, Y_lr, Z_lr, N_dwi]
    """
    n_dwi = lr_dwi.shape[-1]
    lmax = 4 if n_dwi >= 15 else 2

    sh_model = RegularizedSHBaseline(lmax=lmax, lambda_reg=lambda_reg, spatial_scale=float(downscale_factor))

    # Umformen auf [C, D, H, W] für fit_predict
    lr_dwi_tensor = torch.from_numpy(np.transpose(lr_dwi, (3, 0, 1, 2))).float().to(device)
    sh_pred_tensor = sh_model.fit_predict(lr_dwi_tensor, input_bvecs, target_bvecs)

    # Zurück auf [X_hr, Y_hr, Z_hr, T] für DIPY
    sh_pred_np = np.transpose(sh_pred_tensor.cpu().numpy(), (1, 2, 3, 0))
    return sh_pred_np


# ==========================================
# 2. DIPY MIKROSTRUKTUR-BERECHNUNG (DTI / DKI)
# ==========================================

def compute_dipy_microstructure(dwi_hr, b0_hr, target_bvals, target_bvecs, b0_bvals, mask):
    """
    Fügt b0- und DWI-Volumina zusammen und berechnet FA, MD (und optional MK/KFA via DKI).
    Gibt ein Dictionary mit den berechneten 3D-Parameterkarten zurück.
    """
    # 1. b0 und rekonstruierte DWIs entlang der 4. Achse verknüpfen
    full_signal = np.concatenate([b0_hr, dwi_hr], axis=-1)
    full_signal = np.clip(full_signal, a_min=0.0, a_max=None)  # Negative Ausreißer abfangen

    b0_bvecs = np.zeros((len(b0_bvals), 3), dtype=np.float32)
    full_bvals = np.concatenate([b0_bvals, target_bvals])
    full_bvecs = np.vstack([b0_bvecs, target_bvecs])

    gtab = gradient_table(bvals=full_bvals, bvecs=full_bvecs)

    # Prüfen, wie viele unterschiedliche b-Wert-Schalen vorliegen (gerundet auf 200er Schritte)
    rounded_shells = np.unique(np.round(full_bvals / 200.0) * 200)
    non_zero_shells = rounded_shells[rounded_shells > 100]

    results = {}

    if len(non_zero_shells) >= 2:
        print("  -> Multi-Shell erkannt: Berechne DKI (FA, MD, MK, KFA)...")
        dkimodel = dki.DiffusionKurtosisModel(gtab)
        dkifit = dkimodel.fit(full_signal, mask=mask)
        results["FA"] = dkifit.fa
        results["MD"] = dkifit.md
        results["MK"] = dkifit.mk(0, 3)
        results["KFA"] = dkifit.kfa
    else:
        print("  -> Single-Shell erkannt: Berechne DTI (FA, MD)...")
        tenmodel = dti.TensorModel(gtab)
        tenfit = tenmodel.fit(full_signal, mask=mask)
        results["FA"] = tenfit.fa
        results["MD"] = tenfit.md

    return results

def compute_fodf(dwi_volume, gtab, mask, response, sh_order=4):
    """
    Berechnet die fODF (Fiber Orientation Distribution Functions als SH-Koeffizienten)
    mittels Constrained Spherical Deconvolution (CSD).
    
    sh_order=4 benötigt mindestens 15 Richtungen (erzeugt 15 Koeffizienten pro Voxel).
    sh_order=6 benötigt mindestens 28 Richtungen (erzeugt 28 Koeffizienten pro Voxel).
    """
    # CSD-Modell mit der festen Referenz-Response initialisieren
    csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order_max=sh_order)
    
    # Fit nur innerhalb der Gehirnmaske durchführen (spart enorm viel Rechenzeit)
    csd_fit = csd_model.fit(dwi_volume, mask=mask)
    
    # Gibt ein 4D-Array (X, Y, Z, num_sh_coeffs) zurück
    return csd_fit.shm_coeff.astype(np.float32)

def compute_masked_mae(pred, gt, mask):
    """Berechnet den Mean Absolute Error innerhalb der Gehirnmaske."""
    valid = mask > 0
    if pred.ndim == 4:
        return float(np.mean(np.abs(pred[valid] - gt[valid])))
    return float(np.mean(np.abs(pred[valid] - gt[valid])))


# ==========================================
# 3. HAUPTPROGRAMM
# ==========================================

def main():
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("Results", exist_ok=True)

    # 1. Validierungs-Proband laden (erster Proband aus config)
    nifti_path = config.VAL_NIFTI_FILES[0]
    bval_path = config.VAL_BVAL_FILES[0]
    bvec_path = config.VAL_BVEC_FILES[0]
    mask_path = config.VAL_MASK_FILES[0]

    print(f"Lade Proband: {nifti_path}")
    bvals = np.loadtxt(bval_path).flatten()
    bvecs = np.loadtxt(bvec_path).T
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T

    mask_img = nib.load(mask_path)
    brain_mask_full = np.asanyarray(mask_img.dataobj) > 0
    affine = mask_img.affine

    img = nib.load(nifti_path)
    full_data = np.asanyarray(img.dataobj, dtype=np.float32)

    # 2. Sampler initialisieren
    sampler = FlexibleDiffusionSampler(
        bvals, bvecs, config.B0_THRESHOLD, config.MIN_DWI_B, config.MAX_DWI_B
    )
    rng = np.random.RandomState(42)
    
    n_input_dwi = 30
    # Berechnen, wie viele Richtungen in der Schale nach Abzug der 18 Inputs maximal übrig bleiben:
    max_available_targets = len(sampler.global_dwi_indices) - n_input_dwi
    print(f"Insgesamt {len(sampler.global_dwi_indices)} DWIs in der Schale -> Nutze {n_input_dwi} als Input und alle {max_available_targets} übrigen als Targets.")

    sample_spec = sampler.sample(num_targets=max_available_targets, plot=False, rng=rng, n_dwi=n_input_dwi)

    b0_full = full_data[..., sampler.global_b0_indices]
    dwi_full = full_data[..., sampler.global_dwi_indices]

    # Signal-Normalisierung nach dem ersten b0-image
    selected_b0_vol = b0_full[..., 0]
    norm_scalar = calculate_normalization_scalar(selected_b0_vol, brain_mask_full)

    b0_norm = b0_full / norm_scalar
    dwi_norm = dwi_full / norm_scalar

    # Optional: Auf einen zentralen Block zuschneiden (z.B. 64³ HR -> 32³ LR) für schnellen Durchlauf.
    # Für das komplette Gehirn einfach den Slice durch [:] ersetzen (Achtung: durch 2 teilbare Dimensionen!).
    # cx, cy, cz = [s // 2 for s in brain_mask_full.shape]
    # half_crop = 32  # Ergibt 64x64x64 HR-Block
    # sl = (slice(cx - half_crop, cx + half_crop),
    #       slice(cy - half_crop, cy + half_crop),
    #       slice(cz - half_crop, cz + half_crop))

    # Komplettes Volumen nutzen (Dimensionen auf gerade Zahlen für Downscale-Faktor 2 abrunden)
    max_x = (brain_mask_full.shape[0] // config.DOWNSCALE_FACTOR) * config.DOWNSCALE_FACTOR
    max_y = (brain_mask_full.shape[1] // config.DOWNSCALE_FACTOR) * config.DOWNSCALE_FACTOR
    max_z = (brain_mask_full.shape[2] // config.DOWNSCALE_FACTOR) * config.DOWNSCALE_FACTOR
    
    sl = (slice(0, max_x), slice(0, max_y), slice(0, max_z))

    hr_mask = brain_mask_full[sl]
    hr_b0_in = b0_norm[sl + (sample_spec["b0_indices"],)]
    hr_dwi_in = dwi_norm[sl + (sample_spec["input_indices"],)]
    hr_dwi_gt = dwi_norm[sl + (sample_spec["withheld_target_indices"],)]

    # 3. Künstliche LR-Degradation (Blur + Downsampling) erzeugen
    dummy_bvecs = np.zeros((hr_b0_in.shape[-1], 3))
    lr_b0_in, _, _ = generate_and_augment_lr_patch(
        hr_b0_in, hr_mask, dummy_bvecs, rot_spec=None,
        downscale_factor=config.DOWNSCALE_FACTOR, noise_std=0.01
    )
    lr_dwi_in, _, _ = generate_and_augment_lr_patch(
        hr_dwi_in, hr_mask, sample_spec["input_bvecs"], rot_spec=None,
        downscale_factor=config.DOWNSCALE_FACTOR, noise_std=0.01
    )

    # b0 für DIPY trilinear auf HR-Gitter hochskalieren
    lr_b0_tensor = torch.from_numpy(np.transpose(lr_b0_in, (3, 0, 1, 2))).unsqueeze(0).float()
    hr_b0_interp = F.interpolate(
        lr_b0_tensor, scale_factor=config.DOWNSCALE_FACTOR, mode="trilinear", align_corners=False
    ).squeeze(0).numpy()
    hr_b0_interp = np.transpose(hr_b0_interp, (1, 2, 3, 0))

    # 4. Rekonstruktion 1: U-Net (Model A)
    print("\nStarte U-Net Rekonstruktion...")
    model = ModelA_FlexibleSR(meta_dim=14).to(device)
    model_path = "Models/model_A.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Gewichte geladen aus: {model_path}")
    else:
        print(f"WARNUNG: {model_path} nicht gefunden! Nutze uninitialisiertes Modell.")

    hr_dwi_unet = predict_volume_unet(
        model=model,
        lr_b0=lr_b0_in,
        lr_dwi=lr_dwi_in,
        input_bvals=sample_spec["input_bvals"],
        input_bvecs=sample_spec["input_bvecs"],
        target_bvals=sample_spec["target_bvals"],
        target_bvecs=sample_spec["target_bvecs"],
        max_measurements=config.MAX_MEASUREMENTS,
        max_b0=5,
        downscale_factor=config.DOWNSCALE_FACTOR,
        lr_patch_size=config.PATCH_SIZE // config.DOWNSCALE_FACTOR,
        lr_stride=(config.PATCH_SIZE // config.DOWNSCALE_FACTOR) // 2,
        device=device
    )

    # 5. Rekonstruktion 2: Regularized SH Baseline
    print("\nStarte SH-Baseline Rekonstruktion...")
    hr_dwi_sh = predict_volume_sh(
        lr_dwi=lr_dwi_in,
        input_bvecs=sample_spec["input_bvecs"],
        target_bvecs=sample_spec["target_bvecs"],
        downscale_factor=config.DOWNSCALE_FACTOR,
        lambda_reg=0.005,
        device=device
    )

    # 6. Einheiten wiederherstellen (Denormalisierung wie im Workplan gefordert)
    hr_b0_gt_phys = hr_b0_in * norm_scalar
    hr_b0_interp_phys = hr_b0_interp * norm_scalar
    hr_dwi_gt_phys = hr_dwi_gt * norm_scalar
    hr_dwi_unet_phys = hr_dwi_unet * norm_scalar
    hr_dwi_sh_phys = hr_dwi_sh * norm_scalar

    # 7. DIPY Mikrostruktur (FA, MD, ggf. MK) für alle 3 Varianten berechnen
    print("\nBerechne DIPY Mikrostruktur für Ground Truth...")
    metrics_gt = compute_dipy_microstructure(
        hr_dwi_gt_phys, hr_b0_gt_phys, sample_spec["target_bvals"],
        sample_spec["target_bvecs"], sample_spec["b0_bvals"], hr_mask
    )

    print("Berechne DIPY Mikrostruktur für U-Net...")
    metrics_unet = compute_dipy_microstructure(
        hr_dwi_unet_phys, hr_b0_interp_phys, sample_spec["target_bvals"],
        sample_spec["target_bvecs"], sample_spec["b0_bvals"], hr_mask
    )

    print("Berechne DIPY Mikrostruktur für SH-Baseline...")
    metrics_sh = compute_dipy_microstructure(
        hr_dwi_sh_phys, hr_b0_interp_phys, sample_spec["target_bvals"],
        sample_spec["target_bvecs"], sample_spec["b0_bvals"], hr_mask
    )

    # =========================================================================
    # fODF Berechnung mittels CSD (Crossing Fibers)
    # =========================================================================
    print("\nBereite 4D-Volumina (b0 + DWI) und GradientTable für CSD vor...")
    
    # 1. Gemeinsame GradientTable (b0 + Targets) aufbauen
    b0_bvecs = np.zeros((len(sample_spec["b0_bvals"]), 3), dtype=np.float32)
    full_bvals = np.concatenate([sample_spec["b0_bvals"], sample_spec["target_bvals"]])
    full_bvecs = np.vstack([b0_bvecs, sample_spec["target_bvecs"]])
    gtab = gradient_table(bvals=full_bvals, bvecs=full_bvecs)

    # 2. Volumina (b0 + DWI) zusammenfügen und negative Ausreißer clippen
    vol_gt = np.clip(np.concatenate([hr_b0_gt_phys, hr_dwi_gt_phys], axis=-1), a_min=0.0, a_max=None)
    vol_unet = np.clip(np.concatenate([hr_b0_interp_phys, hr_dwi_unet_phys], axis=-1), a_min=0.0, a_max=None)
    vol_sh = np.clip(np.concatenate([hr_b0_interp_phys, hr_dwi_sh_phys], axis=-1), a_min=0.0, a_max=None)

    print("Berechne Response Function auf Ground Truth...")
    # Response Function NUR auf der Ground Truth schätzen!
    response, ratio = auto_response_ssst(gtab, vol_gt, roi_radii=10, fa_thr=0.7)
    print(f"Response Function geschätzt (Ratio: {ratio:.3f}). Starte CSD-Fits...")

    sh_order = 4  # Passt perfekt zu deinen num_targets=20 (benötigt mind. 15)

    print(" -> CSD Fit für Ground Truth...")
    fodf_gt = compute_fodf(vol_gt, gtab, hr_mask, response, sh_order=sh_order)

    print(" -> CSD Fit für U-Net...")
    fodf_unet = compute_fodf(vol_unet, gtab, hr_mask, response, sh_order=sh_order)

    print(" -> CSD Fit für SH-Baseline...")
    fodf_sh = compute_fodf(vol_sh, gtab, hr_mask, response, sh_order=sh_order)

    # 3. Als 4D-NIfTI im Ordner 'Results/' speichern
    nib.save(nib.Nifti1Image(fodf_gt, affine), "Results/fODF_GT.nii.gz")
    nib.save(nib.Nifti1Image(fodf_unet, affine), "Results/fODF_UNet.nii.gz")
    nib.save(nib.Nifti1Image(fodf_sh, affine), "Results/fODF_SH.nii.gz")

    # 8. Quantitative Auswertung & NIfTI Export
    print("\n================ ERGEBNISSE (MAE im Gehirn) ================")
    print(f"Signal MAE (normiert) | U-Net: {compute_masked_mae(hr_dwi_unet, hr_dwi_gt, hr_mask):.5f} "
          f"| SH: {compute_masked_mae(hr_dwi_sh, hr_dwi_gt, hr_mask):.5f}")

    for m_name in metrics_gt.keys():
        mae_unet = compute_masked_mae(metrics_unet[m_name], metrics_gt[m_name], hr_mask)
        mae_sh = compute_masked_mae(metrics_sh[m_name], metrics_gt[m_name], hr_mask)
        print(f"{m_name:6s} MAE            | U-Net: {mae_unet:.5f} | SH: {mae_sh:.5f}")

        # Als NIfTI speichern
        nib.save(nib.Nifti1Image(metrics_gt[m_name].astype(np.float32), affine), f"Results/{m_name}_GT.nii.gz")
        nib.save(nib.Nifti1Image(metrics_unet[m_name].astype(np.float32), affine), f"Results/{m_name}_UNet.nii.gz")
        nib.save(nib.Nifti1Image(metrics_sh[m_name].astype(np.float32), affine), f"Results/{m_name}_SH.nii.gz")

    # fODF MAE direkt mit in der Ergebnistabelle ausgeben
    fodf_mae_unet = compute_masked_mae(fodf_unet, fodf_gt, hr_mask)
    fodf_mae_sh = compute_masked_mae(fodf_sh, fodf_gt, hr_mask)
    print(f"fODF   MAE            | U-Net: {fodf_mae_unet:.5f} | SH: {fodf_mae_sh:.5f}")

    print("============================================================")
    # 9. Speichern der generierten 4D-DWI-Volumina als NIfTI
    nib.save(nib.Nifti1Image(hr_dwi_gt_phys.astype(np.float32), affine), "Results/DWI_GT.nii.gz")
    nib.save(nib.Nifti1Image(hr_dwi_unet_phys.astype(np.float32), affine), "Results/DWI_UNet.nii.gz")
    nib.save(nib.Nifti1Image(hr_dwi_sh_phys.astype(np.float32), affine), "Results/DWI_SH.nii.gz")
    print("Alle NIfTI-Karten (inkl. fODF) wurden im Ordner 'Results/' gespeichert.")

if __name__ == "__main__":
    main()