import nibabel as nib
import numpy as np
from dipy.core.sphere import Sphere
from dipy.reconst.shm import sf_to_sh

# 4D-DWIs und bvecs laden (erste N_b0 Spalten aus DSI_GT.bvec überspringen)
dwi_gt   = nib.load("Results/DWI_GT.nii.gz").get_fdata()
dwi_unet = nib.load("Results/DWI_UNet.nii.gz").get_fdata()
dwi_sh   = nib.load("Results/DWI_SH.nii.gz").get_fdata()
zones    = nib.load("Results/Brain_6_Zones_Markers.nii.gz").get_fdata()

full_bvals = np.loadtxt("Results/DSI_GT.bval")
full_bvecs = np.loadtxt("Results/DSI_GT.bvec").T
dwi_bvecs  = full_bvecs[full_bvals > 100]  # nur die 60 DWI-Richtungen
sphere     = Sphere(xyz=dwi_bvecs)

orders = {
    "l=0 (Mittelwert)      ": slice(0, 1),
    "l=2 (Haupt-Kontrast)  ": slice(1, 6),
    "l=4 (Kreuzungs-Signal)": slice(6, 15),
}

for z_id, z_name in [(1, "Zone 1: Einzelfaser"), (2, "Zone 2: Kreuzung")]:
    mask = (zones == z_id)
    # Direkter SH-Fit auf das rohe 60-Richtungen-DWI-Signal (ohne CSD!)
    sh_gt   = sf_to_sh(dwi_gt[mask],   sphere, sh_order_max=4)  # Spherical Function to Spherical Harmonics
    sh_unet = sf_to_sh(dwi_unet[mask], sphere, sh_order_max=4)  # zerlegt DWI-Signal mathematisch in (l=0), (l=2) und (l=4) Winkel-Anteile
    sh_base = sf_to_sh(dwi_sh[mask],   sphere, sh_order_max=4)

    print(f"\n=== Rohes DWI-Signal | {z_name} ===")
    for name, sl in orders.items():
        e_gt   = np.linalg.norm(sh_gt[:, sl],   axis=-1).mean()  # berechnet pro Voxel die Vektorlänge ($\sqrt{c_1^2 + c_2^2 + \dots}$) dieser Gruppe
        e_unet = np.linalg.norm(sh_unet[:, sl], axis=-1).mean()  # rotationsunabhängige Stärke (Energie) dieses Frequenzbands im jeweiligen Voxel.
        e_sh   = np.linalg.norm(sh_base[:, sl], axis=-1).mean()
        print(f"{name} | GT: {e_gt:7.1f} | U-Net: {e_unet:7.1f} ({100*e_unet/e_gt:5.1f}%) | "
              f"SH: {e_sh:7.1f} ({100*e_sh/e_gt:5.1f}%)")