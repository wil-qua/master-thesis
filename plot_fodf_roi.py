import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from dipy.core.sphere import Sphere
from dipy.reconst.shm import sh_to_sf

# --- 1. Pfade & Mittelpunkt im Centrum semiovale anpassen ---
FA_GT  = "Results/FA_GT.nii.gz"
FA_UNET  = "Results/FA_UNet.nii.gz"
FA_SH  = "Results/FA_SH.nii.gz"
FODF_GT   = "Results/fODF_GT.nii.gz"
FODF_UNET = "Results/fODF_UNet.nii.gz"
FODF_SH   = "Results/fODF_SH.nii.gz"

"""
Das Skript schneidet einen 12×12-Ausschnitt im Centrum semiovale aus,
rechnet die mathematischen fODF-Koeffizienten in sichtbare Faser-Glyphen um
und zeichnet GT, U-Net und SH im selben Maßstab nebeneinander auf die FA-Karte.
"""


img = nib.load(FA_GT)
X_MM, Y_MM, Z_MM = 23.75, -17.25, 31.75   # Zone 2
# X_MM, Y_MM, Z_MM = 5, 10.25, 10.5   # Zone 6
CX, CY, CZ = np.round(
    nib.affines.apply_affine(np.linalg.inv(img.affine), [X_MM, Y_MM, Z_MM])
).astype(int)

HALF = 6                  # 12x12 Voxel Ausschnitt im koronalen Schnitt (X-Z)

# --- 2. Daten laden ---
fa_gt   = nib.load(FA_GT).get_fdata()
fa_unet = nib.load(FA_UNET).get_fdata()
fa_sh = nib.load(FA_SH).get_fdata()
sh_gt   = nib.load(FODF_GT).get_fdata()
sh_unet = nib.load(FODF_UNET).get_fdata()
sh_base = nib.load(FODF_SH).get_fdata()

xs = slice(CX - HALF, CX + HALF)
zs = slice(CZ - HALF, CZ + HALF)
fa_roi = fa_gt[xs, CY, zs]

# 2D-Winkelring in der koronalen Ebene (X = links-rechts, Z = unten-oben)
angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
ring_sphere = Sphere(x=np.cos(angles), y=np.zeros_like(angles), z=np.sin(angles))

def eval_roi(sh_vol):
    roi = sh_vol[xs, CY, zs, :]
    n_c = roi.shape[-1]
    # lmax automatisch aus Anzahl der SH-Koeffizienten bestimmen (6->2, 15->4, 28->6, 45->8)
    lmax = int(round((-3 + np.sqrt(1 + 8 * n_c)) / 2))
    sf = sh_to_sf(roi, ring_sphere, lmax)
    return np.clip(sf, 0, None)

sf_gt   = eval_roi(sh_gt)
sf_unet = eval_roi(sh_unet)
sf_base = eval_roi(sh_base)

# Gemeinsame Skalierung anhand GT, damit auch Amplituden vergleichbar bleiben
scale = np.percentile(sf_gt, 99) + 1e-8

# --- 3. Nebeneinander plotten ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=220)
titles = ["Ground Truth (GT)", "U-Net", "SH Baseline"]
sfs    = [sf_gt, sf_unet, sf_base]
fas = fas = [fa_gt[xs, CY, zs], fa_unet[xs, CY, zs], fa_sh[xs, CY, zs]]
nx, nz = fa_roi.shape
for ax, title, sf, fa_bg in zip(axes, titles, sfs, fas):
    im = ax.imshow(fa_bg.T, origin="lower", cmap="gray", vmin=0, vmax=0.85,
              extent=[-0.5, nx - 0.5, -0.5, nz - 0.5])
    for ix in range(nx):
        for iz in range(nz):
            r = np.minimum(sf[ix, iz] / scale, 1.2) * 0.45
            gx = ix + r * np.cos(angles)
            gz = iz + r * np.sin(angles)
            ax.fill(gx, gz, facecolor="#00d4ff", edgecolor="#003366", alpha=0.8, linewidth=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(-0.5, nz - 0.5)

plt.tight_layout()
cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
cbar.set_label("Fractional Anisotropy (FA)", fontsize=12)
plt.savefig("Results/fodf_comparison_zone2.png", bbox_inches="tight")
print("Gespeichert unter: Results/fodf_comparison_zone2.png")