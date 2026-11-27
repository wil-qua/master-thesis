import nibabel as nib
import numpy as np
import dipy.reconst.dti as dti
from dipy.core.gradients import gradient_table

# --- 1. Maske und GradientTable laden ---
mask_img = nib.load("Results/nodif_brain_mask.nii.gz")
mask     = mask_img.get_fdata() > 0
affine   = mask_img.affine
suffix = "_new"

bvals = np.loadtxt(f"Results/DSI_GT{suffix}.bval").flatten()
bvecs = np.loadtxt(f"Results/DSI_GT{suffix}.bvec").T
gtab  = gradient_table(bvals=bvals, bvecs=bvecs)

# --- 2. DTI-Fit für GT, U-Net und SH berechnen ---
def fit_tensor_metrics(nii_path):
    vol = np.clip(nib.load(nii_path).get_fdata().astype(np.float32), 0.0, None)
    tenfit = dti.TensorModel(gtab).fit(vol, mask=mask)
    evals = tenfit.evals  # Shape: (X, Y, Z, 3), sortiert: L1 >= L2 >= L3
    return {
        "FA": tenfit.fa,
        "MD": tenfit.md,
        "AD (L1)": evals[..., 0],
        "L2":      evals[..., 1],
        "L3":      evals[..., 2],
        "RD":      tenfit.rd,
    }

print("Fitte Diffusionstensor für GT, U-Net und SH...")
m_gt   = fit_tensor_metrics(f"Results/DSI_GT{suffix}.nii.gz")
m_unet = fit_tensor_metrics(f"Results/DSI_UNet{suffix}.nii.gz")
m_sh   = fit_tensor_metrics(f"Results/DSI_SH{suffix}.nii.gz")

# WM-Maske anhand GT-FA (analog zu deinem bisherigen Vergleich)
wm_mask = mask & (m_gt["FA"] > 0.2)

# --- 3. Tabelle (MAE & Bias) ausgeben ---
# Diffusivitäten in 10^-3 mm^2/s skalieren (Standard in der dMRI-Literatur)
print("\n======= DIFFUSIVITY & EIGENWERTE (x 10^-3 mm^2/s, gegen GT 60 Dir) =======")
for name in ["MD", "AD (L1)", "L2", "L3", "RD"]:
    scale = 1e3
    gt_v = m_gt[name] * scale
    u_v  = m_unet[name] * scale
    s_v  = m_sh[name] * scale

    # Nur valide (nicht-NaN) Voxel in der Maske berücksichtigen
    valid_all = mask & np.isfinite(gt_v) & np.isfinite(u_v) & np.isfinite(s_v)
    valid_wm  = wm_mask & valid_all

    du_all, ds_all = u_v[valid_all] - gt_v[valid_all], s_v[valid_all] - gt_v[valid_all]
    du_wm,  ds_wm  = u_v[valid_wm]  - gt_v[valid_wm],  s_v[valid_wm]  - gt_v[valid_wm]

    print(
        f"{name:<8} | "
        f"Brain MAE -> U-Net: {np.abs(du_all).mean():.4f} ({du_all.mean():+.4f}) | "
        f"SH: {np.abs(ds_all).mean():.4f} ({ds_all.mean():+.4f}) || "
        f"WM MAE -> U-Net: {np.abs(du_wm).mean():.4f} ({du_wm.mean():+.4f}) | "
        f"SH: {np.abs(ds_wm).mean():.4f} ({ds_wm.mean():+.4f})"
    )
print("==========================================================================")

# --- 4. Optional: Karten als NIfTI für MRIcroGL / Plots speichern ---
for tag, m_dict in [("GT", m_gt), ("UNet", m_unet), ("SH", m_sh)]:
    for key, fname in [("AD (L1)", "L1"), ("L2", "L2"), ("L3", "L3"), ("RD", "RD")]:
        nib.save(
            nib.Nifti1Image(np.nan_to_num(m_dict[key]).astype(np.float32), affine),
            f"Results/{fname}_{tag}{suffix}.nii.gz"
        )
print("NIfTI-Karten (L1, L2, L3, RD) in Results/ gespeichert.")