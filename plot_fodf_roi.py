import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PolyCollection
from dipy.core.sphere import Sphere
from dipy.reconst.shm import sh_to_sf

# --- 1. Pfade & Mittelpunkt im Centrum semiovale anpassen ---
FA_GT     = "Results/FA_GT_old.nii.gz"
FA_UNET   = "Results/FA_UNet_old.nii.gz"
FA_SH     = "Results/FA_SH_old.nii.gz"
FODF_GT   = "Results/fODF_GT_old.nii.gz"
FODF_UNET = "Results/fODF_UNet_old.nii.gz"
FODF_SH   = "Results/fODF_SH_old.nii.gz"

img = nib.load(FA_GT)
X_MM, Y_MM, Z_MM = 20.0, -10.0, 29.0 # 20.0, -10.0, 29.0   # Zone 2 (Centrum Semiovale)
# X_MM, Y_MM, Z_MM = 5, 10.25, 10.5       # Zone 6

CX, CY, CZ = np.round(
    nib.affines.apply_affine(np.linalg.inv(img.affine), [X_MM, Y_MM, Z_MM])
).astype(int)

HALF = 6  # 12x12 Voxel Ausschnitt im koronalen Schnitt (X-Z)
VOXEL_MM = float(img.header.get_zooms()[0])  # z.B. 1.25 mm

# Anatomische Orientierung automatisch aus der NIfTI-Affine ableiten
axcodes = nib.aff2axcodes(img.affine)
opp = {"L": "R", "R": "L", "A": "P", "P": "A", "S": "I", "I": "S"}
right_lbl, left_lbl = axcodes[0], opp[axcodes[0]]
top_lbl, bot_lbl    = axcodes[2], opp[axcodes[2]]

# --- 2. Daten laden ---
fa_gt   = img.get_fdata()
fa_unet = nib.load(FA_UNET).get_fdata()
fa_sh   = nib.load(FA_SH).get_fdata()
sh_gt   = nib.load(FODF_GT).get_fdata()
sh_unet = nib.load(FODF_UNET).get_fdata()
sh_base = nib.load(FODF_SH).get_fdata()

xs = slice(CX - HALF, CX + HALF)
zs = slice(CZ - HALF, CZ + HALF)
fa_slice_full = fa_gt[:, CY, :]  # Gesamte koronale Schicht (X, Z)

# 2D-Winkelring in der koronalen Ebene (X = links-rechts, Z = unten-oben)
angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
ring_sphere = Sphere(x=np.cos(angles), y=np.zeros_like(angles), z=np.sin(angles))

def eval_roi(sh_vol):
    roi = sh_vol[xs, CY, zs, :]
    n_c = roi.shape[-1]
    lmax = int(round((-3 + np.sqrt(1 + 8 * n_c)) / 2))
    sf = sh_to_sf(roi, ring_sphere, lmax)
    return np.clip(sf, 0, None)

sf_gt   = eval_roi(sh_gt)
sf_unet = eval_roi(sh_unet)
sf_base = eval_roi(sh_base)

# Gemeinsame Skalierung anhand GT, damit auch Amplituden vergleichbar bleiben
scale = np.percentile(sf_gt, 96) + 1e-8

# Richtungsabhängige RGB-Farben pro Winkel-Segment (Standard dMRI: Rot=L-R, Blau=S-I)
mid_angles = angles + (angles[1] - angles[0]) / 2.0
w = np.abs(np.cos(mid_angles)) ** 2  # 1 = horizontal (L-R), 0 = vertikal (S-I)
c_horiz = np.array([1.00, 0.55, 0.05])  # Kräftiges Orange-Gold
c_vert  = np.array([0.00, 0.88, 1.00])  # Leuchtendes Cyan
seg_colors = np.outer(w, c_horiz) + np.outer(1.0 - w, c_vert)

# --- 3. 4-Panel-Layout (Übersicht + 3 Zoom-Panels + Colorbar) ---
fig = plt.figure(figsize=(18, 4.8), dpi=240)
gs = fig.add_gridspec(1, 5, width_ratios=[1.1, 1, 1, 1, 0.05], wspace=0.06)

ax_loc  = fig.add_subplot(gs[0, 0])
axes    = [fig.add_subplot(gs[0, i]) for i in range(1, 4)]
ax_cbar = fig.add_subplot(gs[0, 4])

ROI_COLOR = "#ff3b30"  # Kräftiges Rot für ROI-Box & Rahmen

# --- Panel 0: Ganzhirn-Übersicht (Locator Slice) ---
ax_loc.imshow(fa_slice_full.T, origin="lower", cmap="gray", vmin=0, vmax=0.85)

# Schwarzen Rand um das Gehirn automatisch knapp zuschneiden
nz_x, nz_z = np.where(fa_slice_full > 0.02)
pad = 6
x_min, x_max = max(0, nz_x.min() - pad), min(fa_slice_full.shape[0] - 1, nz_x.max() + pad)
z_min, z_max = max(0, nz_z.min() - pad), min(fa_slice_full.shape[1] - 1, nz_z.max() + pad)
ax_loc.set_xlim(x_min, x_max)
ax_loc.set_ylim(z_min, z_max)

# Exakte ROI-Box einzeichnen
rect = Rectangle(
    (CX - HALF - 0.5, CZ - HALF - 0.5),
    2 * HALF, 2 * HALF,
    linewidth=2.0, edgecolor=ROI_COLOR, facecolor="none"
)
ax_loc.add_patch(rect)

# Anatomische Orientierungsbuchstaben (L/R, S/I)
lbl_kw = dict(color="white", fontsize=11, fontweight="bold",
              bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=1.5))
ax_loc.text(0.03, 0.50, left_lbl,  transform=ax_loc.transAxes, va="center", ha="left",   **lbl_kw)
ax_loc.text(0.97, 0.50, right_lbl, transform=ax_loc.transAxes, va="center", ha="right",  **lbl_kw)
ax_loc.text(0.50, 0.97, top_lbl,   transform=ax_loc.transAxes, va="top",    ha="center", **lbl_kw)
ax_loc.text(0.50, 0.03, bot_lbl,   transform=ax_loc.transAxes, va="bottom", ha="center", **lbl_kw)

ax_loc.set_title(f"Coronal Slice (y = {Y_MM:.1f} mm)", fontsize=12.5, fontweight="bold")
ax_loc.set_xticks([])
ax_loc.set_yticks([])

# --- Panels 1-3: Zoom-Ansichten (GT, U-Net, SH) ---
titles = ["Ground Truth (GT)", "U-Net", "SH Baseline"]
sfs    = [sf_gt, sf_unet, sf_base]
fas    = [fa_gt[xs, CY, zs], fa_unet[xs, CY, zs], fa_sh[xs, CY, zs]]
nx, nz = fas[0].shape

for idx, (ax, title, sf, fa_bg) in enumerate(zip(axes, titles, sfs, fas)):
    im = ax.imshow(
        fa_bg.T, origin="lower", cmap="gray", vmin=0, vmax=0.85,
        extent=[-0.5, nx - 0.5, -0.5, nz - 0.5]
    )

    # Alle fODF-Wedges gesammelt als PolyCollection zeichnen (schnell + richtungsfarbig)
    polys = []
    colors = []
    for ix in range(nx):
        for iz in range(nz):
            r = np.minimum(sf[ix, iz] / scale, 1.2) * 0.45
            gx = ix + r * np.cos(angles)
            gz = iz + r * np.sin(angles)

            # Dreiecks-Segmente vom Voxelzentrum zu benachbarten Randpunkten
            gx_next = np.roll(gx, -1)
            gz_next = np.roll(gz, -1)
            for k in range(len(angles)):
                polys.append([(ix, iz), (gx[k], gz[k]), (gx_next[k], gz_next[k])])
                colors.append(seg_colors[k])

            # Dünne dunkle Kontur für maximale Schärfe
            ax.plot(
                np.append(gx, gx[0]), np.append(gz, gz[0]),
                color="#111111", linewidth=0.4, alpha=0.85
            )

    pc = PolyCollection(polys, facecolors=colors, edgecolors="none", alpha=0.88)
    ax.add_collection(pc)

    # Farblich passender Rahmen zur ROI-Box links
    for spine in ax.spines.values():
        spine.set_edgecolor(ROI_COLOR)
        spine.set_linewidth(2.0)

    # Maßstabsbalken (5 mm) im ersten Zoom-Panel unten links
    if idx == 0:
        bar_mm = 5.0
        bar_vox = bar_mm / VOXEL_MM
        x0, y0 = 0.2, 0.35
        ax.plot([x0, x0 + bar_vox], [y0, y0], color="white", linewidth=3.0)
        ax.text(
            x0 + bar_vox / 2, y0 + 0.35, f"{bar_mm:.0f} mm",
            color="white", fontsize=9.5, fontweight="bold", ha="center", va="bottom",
            bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1.0)
        )

    ax.set_title(title, fontsize=12.5, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, nz - 0.5)

# --- Colorbar ---
cbar = fig.colorbar(im, cax=ax_cbar)
cbar.set_label("Fractional Anisotropy (FA)", fontsize=11.5)

out_path = "Results/fodf_comparison_zone2_old.png"
plt.savefig(out_path, bbox_inches="tight", dpi=240)
print(f"Gespeichert unter: {out_path}")