import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Einstellungen ---
VIEW = "coronal"   # "axial", "coronal" oder "sagittal"
SLICE_IDX = None   # None = automatisch mittlere Hirnschicht (oder z.B. fester Index)
# Optional: exakte mm-Koordinate wie im fODF-Plot (für coronal y = -17.25 mm)
TARGET_MM = -17.25 if VIEW == "coronal" else None

# --- 2. Bereits gespeicherte NIfTIs aus Results/ laden ---
ref_img = nib.load("Results/FA_GT_90.nii.gz")
affine  = ref_img.affine

def load_clean(path):
    return np.nan_to_num(nib.load(path).get_fdata(), nan=0.0)

mask    = load_clean("Results/nodif_brain_mask.nii.gz") > 0
fa_lr30 = load_clean("Results/FA_LR_30_trilinear.nii.gz") * mask
fa_sh   = load_clean("Results/FA_SH.nii.gz") * mask
fa_unet = load_clean("Results/FA_UNet.nii.gz") * mask
fa_gt90 = load_clean("Results/FA_GT_90.nii.gz") * mask
fa_gt30 = load_clean("Results/FA_GT_30.nii.gz") * mask

# --- 3. Schnittebene & Index bestimmen ---
view_cfg = {
    "axial":    (2, 0, 1),  # (slice_axis, horiz_axis, vert_axis)
    "coronal":  (1, 0, 2),
    "sagittal": (0, 1, 2),
}
s_ax, h_ax, v_ax = view_cfg[VIEW]

if SLICE_IDX is not None:
    idx = SLICE_IDX
elif TARGET_MM is not None:
    pt = [0.0, 0.0, 0.0]
    pt[s_ax] = TARGET_MM
    idx = int(round(nib.affines.apply_affine(np.linalg.inv(affine), pt)[s_ax]))
else:
    coords = np.where(mask)[s_ax]
    idx = int(np.median(coords))

def get_slice(vol):
    return np.take(vol, idx, axis=s_ax).T

mask_sl = get_slice(mask)

# Tight Crop um das Gehirn
nz_v, nz_h = np.where(mask_sl)
pad = 5
xlim = (max(0, nz_h.min() - pad), min(mask_sl.shape[1] - 1, nz_h.max() + pad))
ylim = (max(0, nz_v.min() - pad), min(mask_sl.shape[0] - 1, nz_v.max() + pad))

# Anatomische Orientierungsbuchstaben aus Affine
axcodes = nib.aff2axcodes(affine)
opp = {"L": "R", "R": "L", "A": "P", "P": "A", "S": "I", "I": "S"}
right_lbl, left_lbl = axcodes[h_ax], opp[axcodes[h_ax]]
top_lbl,   bot_lbl  = axcodes[v_ax], opp[axcodes[v_ax]]

# --- 4. Plot-Daten vorbereiten ---
top_panels = [
    ("LR-Input (30 Dir, trilinear)", fa_lr30),
    ("SH-Baseline (60 Dir)",         fa_sh),
    ("U-Net (60 Dir)",               fa_unet),
    ("HR-GT (90 Dir)",               fa_gt90),
]

bot_panels = [
    ("LR (30) − HR-GT (90)",       fa_lr30),
    ("SH (60) − HR-GT (90)",       fa_sh),
    ("U-Net (60) − HR-GT (90)",    fa_unet),
    ("HR-GT (30) − HR-GT (90)",    fa_gt30),
]

# Divergierende Colormap für Fehler (0 = weiß, blau = unterschätzt, rot = überschätzt)
cmap_diff = plt.get_cmap("RdBu_r").copy()
cmap_diff.set_bad("black")

# --- 5. 2x4-Abbildung erstellen ---
fig = plt.figure(figsize=(16, 8.2), dpi=240)
gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.045], wspace=0.05, hspace=0.14)

# Obere Reihe: FA-Karten
for col, (title, vol) in enumerate(top_panels):
    ax = fig.add_subplot(gs[0, col])
    im_fa = ax.imshow(get_slice(vol), origin="lower", cmap="gray", vmin=0.0, vmax=0.85)
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=6)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])

    # Orientierungsmarker im ersten Panel
    if col == 0:
        kw = dict(color="white", fontsize=10, fontweight="bold")
        ax.text(0.03, 0.50, left_lbl,  transform=ax.transAxes, va="center", ha="left",   **kw)
        ax.text(0.97, 0.50, right_lbl, transform=ax.transAxes, va="center", ha="right",  **kw)
        ax.text(0.50, 0.97, top_lbl,   transform=ax.transAxes, va="top",    ha="center", **kw)
        ax.text(0.50, 0.03, bot_lbl,   transform=ax.transAxes, va="bottom", ha="center", **kw)

cbar_fa = fig.colorbar(im_fa, cax=fig.add_subplot(gs[0, 4]))
cbar_fa.set_label("Fractional Anisotropy (FA)", fontsize=11)

# Untere Reihe: Differenzkarten gegen HR-GT (90)
for col, (title, vol) in enumerate(bot_panels):
    ax = fig.add_subplot(gs[1, col])
    diff_3d = vol - fa_gt90
    mae_3d  = np.abs(diff_3d[mask]).mean()
    bias_3d = diff_3d[mask].mean()

    diff_sl = np.where(mask_sl, get_slice(diff_3d), np.nan)
    im_diff = ax.imshow(diff_sl, origin="lower", cmap=cmap_diff, vmin=-0.25, vmax=0.25)
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=6)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])

    # 3D-Brain-MAE & Bias direkt unten links einblenden
    ax.text(
        0.03, 0.03, f"MAE: {mae_3d:.4f}\nBias: {bias_3d:+.4f}",
        transform=ax.transAxes, color="white", fontsize=9, fontweight="bold",
        va="bottom", ha="left",
        bbox=dict(facecolor="black", alpha=0.65, edgecolor="none", pad=2.5)
    )

cbar_diff = fig.colorbar(im_diff, cax=fig.add_subplot(gs[1, 4]))
cbar_diff.set_label("ΔFA (Method − HR-GT 90)", fontsize=11)

out_path = f"Results/fa_comparison_4panel_{VIEW}.png"
plt.savefig(out_path, bbox_inches="tight", dpi=240)
print(f"Gespeichert unter: {out_path}")