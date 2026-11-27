import nibabel as nib
import numpy as np

suffix = "_new"

def save_diff(pred_path, gt_path, out_path, mask=None):
    img_gt = nib.load(gt_path)
    pred = nib.load(pred_path).get_fdata()
    gt = img_gt.get_fdata()
    
    diff = pred - gt  # Positiv = U-Net zu hell/hoch, Negativ = U-Net zu dunkel/niedrig
    if mask is not None:
        diff = diff * mask
        
    nib.save(nib.Nifti1Image(diff.astype(np.float32), img_gt.affine, img_gt.header), out_path)
    print(f"Gespeichert: {out_path} | Min: {diff.min():.4f}, Max: {diff.max():.4f}")

# Optional: Gehirnmaske laden, damit der Hintergrund sauber 0 bleibt
# mask = nib.load("Results/Brain_6_Zones_Markers.nii.gz").get_fdata() > 0
mask = None

# 1. FA & MD Differenzkarten (3D)
save_diff(f"Results/FA_UNet{suffix}.nii.gz", f"Results/FA_GT{suffix}.nii.gz", f"Results/Diff_FA_UNet_minus_GT{suffix}.nii.gz", mask)
save_diff(f"Results/MD_UNet{suffix}.nii.gz", f"Results/MD_GT{suffix}.nii.gz", f"Results/Diff_MD_UNet_minus_GT{suffix}.nii.gz", mask)


DWI_UNET_PATH = f"Results/DWI_UNet{suffix}.nii.gz"  # <-- Dateinamen prüfen
DWI_GT_PATH   = f"Results/DWI_GT{suffix}.nii.gz"

img_gt = nib.load(DWI_GT_PATH)
pred = nib.load(DWI_UNET_PATH).get_fdata()
gt   = img_gt.get_fdata()

print("DWI Shape:", gt.shape)  # z. B. (X, Y, Z, 60)

# Optional: Gehirnmaske anwenden, damit Hintergrund sauber 0 bleibt
# mask = nib.load("Results/Brain_6_Zones_Markers.nii.gz").get_fdata() > 0

# Eine (oder direkt mehrere) Richtungen auswählen, z. B. Index 0 und 15
for idx in [0, 15]:
    diff = pred[..., idx] - gt[..., idx]
    # diff = diff * mask  # falls Maske genutzt wird

    out_path = f"Results/Diff_DWI_dir{idx}_UNet_minus_GT{suffix}.nii.gz"
    nib.save(nib.Nifti1Image(diff.astype(np.float32), img_gt.affine, img_gt.header), out_path)

    # 99%-Perzentil hilft dir direkt für die Skala in MRIcroGL
    p99 = np.percentile(np.abs(diff), 99)
    print(f"Richtung {idx} gespeichert -> {out_path} | Vorschlag Skala: -{p99:.4f} bis +{p99:.4f}")