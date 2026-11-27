"""
plot_microstructure_comparison.py
=================================
Erzeugt einen einzigen 2x4-Vergleichsplot zwischen Ground Truth (oben) und
U-Net-Vorhersage (unten) für:
  1. Diffusionsgewichtetes Signal (DWI – gemittelt oder einzelne Richtung)
  2. Fraktionale Anisotropie (FA)
  3. Farbkodierte Fraktionale Anisotropie (Color FA / Hauptfaserrichtung)
  4. Mittlere Diffusivität (MD)

Zusätzlich wird die mittlere Winkelabweichung (Angular Error in Grad) der
Hauptfaserrichtung (erster Eigenvektor v1) in der weißen Substanz berechnet.
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
import dipy.reconst.dti as dti
from dipy.core.gradients import gradient_table
from scipy.ndimage import binary_erosion


# =============================================================================
# 1. EINSTELLUNGEN (Hier Schicht, Dateien und Ausschnitt anpassen)
# =============================================================================
SUFFIX = "_new"          # Datei-Suffix aus microstructure_inference_2.py ("_new" oder "")
VIEW = "coronal"         # Schnittebene: "coronal", "axial" oder "sagittal"
TARGET_MM = -17.25       # Position in mm (z. B. -17.25 wie im fODF-Plot) oder None für Hirnmitte
SLICE_IDX = None         # Optional: fester Voxel-Index (überschreibt TARGET_MM)

# Einstellung für die DWI-Spalte (Spalte 1):
# None       = Mittelwert über alle 60 Zielrichtungen (rauscharm, ideal neben FA/MD)
# Ganzzahl   = eine einzelne ungesehene Richtung (z. B. 0), zeigt rohes Scanner-Rauschen
DWI_DIR_IDX = None

# Optionaler Zoom-Ausschnitt in 2D-Voxelkoordinaten: (h_min, h_max, v_min, v_max)
# None = komplettes Gehirn wird automatisch knapp zugeschnitten
ZOOM_BOX = None

# Helligkeitsfaktor für die farbkodierte FA-Karte (1.0 = unverändert, z. B. 1.15 = etwas leuchtender)
# Wird für Ground Truth und U-Net identisch angewendet, damit der Vergleich fair bleibt.
COLOR_FA_GAIN = 1.0


# =============================================================================
# 2. HILFSFUNKTIONEN ZUM LADEN UND BERECHNEN
# =============================================================================
def load_3d_map(path, mask, dir_idx=None):
    """
    Lädt eine 3D- oder 4D-NIfTI-Datei, bereinigt eventuelle NaN-Werte,
    setzt negative Ausreißer auf 0 und maskiert alles außerhalb des Gehirns aus.
    Bei 4D-Dateien (z. B. DWI mit 60 Richtungen) wird entweder gemittelt (dir_idx=None)
    oder eine einzelne Richtung herausgegriffen.
    """
    data = np.nan_to_num(nib.load(path).get_fdata().astype(np.float32), nan=0.0)
    if data.ndim == 4:
        data = data.mean(axis=-1) if dir_idx is None else data[..., dir_idx]
    return np.clip(data, 0.0, None) * mask


def compute_color_fa_and_v1(dsi_nii_path, gtab, mask):
    """
    Fittet den Diffusionstensor (DTI) mit DIPY auf einem kompletten 4D-Volumen
    (b0 + alle 60 DWI-Richtungen) und berechnet daraus:
      - rgb : Farbkodierte FA-Karte der Form (X, Y, Z, 3) mit Werten in [0, 1].
              Rot = links-rechts, Grün = vorne-hinten, Blau = oben-unten.
              Die Helligkeit jedes Pixels wird mit dem FA-Wert gewichtet.
      - fa  : 3D-Karte der Fraktionalen Anisotropie.
      - v1  : Erster Eigenvektor (Hauptfaserrichtung) pro Voxel, Form (X, Y, Z, 3).
    """
    vol = np.clip(nib.load(dsi_nii_path).get_fdata().astype(np.float32), 0.0, None)
    tenfit = dti.TensorModel(gtab).fit(vol, mask=mask)

    fa = np.clip(np.nan_to_num(tenfit.fa, nan=0.0), 0.0, 1.0)

    # DIPY multipliziert die Beträge des ersten Eigenvektors (|vx|, |vy|, |vz|) mit FA
    rgb = dti.color_fa(fa, tenfit.evecs)
    rgb[~mask] = 0.0

    # tenfit.evecs hat Shape (X, Y, Z, 3, 3): letzte Achse = Index des Eigenvektors (0 = größter)
    v1 = np.nan_to_num(tenfit.evecs[..., 0], nan=0.0)
    return rgb, fa, v1


def compute_angular_error_deg(v_pred, v_ref, roi_mask):
    """
    Berechnet die mittlere Winkelabweichung (Angular Error) in Grad zwischen
    vorhergesagter Hauptfaserrichtung (v_pred) und Ground Truth (v_ref)
    innerhalb der gewählten Region (roi_mask, hier: weiße Substanz).

    - FA und MD messen nur Form/Größe des Tensors, sind aber blind für die Richtung.
    - Prüft direkt die Richtungstreue (entscheidend für Traktographie & Angular Upsampling).
    - Auswertung nur in der WM, da nur gebündelte Nervenfasern eine eindeutige Richtung
      haben (graue Substanz und Hirnwasser diffundieren nahezu isotrop).
    """
    vp = v_pred[roi_mask]
    vr = v_ref[roi_mask]

    # Vektoren zur Sicherheit auf Länge 1 normieren und ungültige Nullvektoren ignorieren
    norm_p = np.linalg.norm(vp, axis=-1)
    norm_r = np.linalg.norm(vr, axis=-1)
    valid = (norm_p > 1e-6) & (norm_r > 1e-6)

    vp = vp[valid] / norm_p[valid, None]
    vr = vr[valid] / norm_r[valid, None]

    # Betrag des Skalarprodukts |vp · vr|, da Fasern antipodal symmetrisch sind (+v == -v).
    # Clippen auf [0, 1] verhindert numerische Fehler bei arccos (z. B. 1.0000001).
    cos_sim = np.clip(np.abs(np.sum(vp * vr, axis=-1)), 0.0, 1.0)
    angles_deg = np.degrees(np.arccos(cos_sim))
    return float(angles_deg.mean()), float(np.median(angles_deg))


# =============================================================================
# 3. DATEN LADEN & FARBKODIERTE FA BERECHNEN
# =============================================================================
print("Lade Gehirnmaske und vorberechnete Karten (DWI, FA, MD)...")
mask_img = nib.load("Results/nodif_brain_mask.nii.gz")
mask = mask_img.get_fdata() > 0
affine = mask_img.affine

# Bereits gespeicherte Skalar- und Signal-Karten laden
dwi_gt   = load_3d_map(f"Results/DWI_GT{SUFFIX}.nii.gz",   mask, dir_idx=DWI_DIR_IDX)
dwi_unet = load_3d_map(f"Results/DWI_UNet{SUFFIX}.nii.gz", mask, dir_idx=DWI_DIR_IDX)

fa_gt    = load_3d_map(f"Results/FA_GT{SUFFIX}.nii.gz",    mask)
fa_unet  = load_3d_map(f"Results/FA_UNet{SUFFIX}.nii.gz",  mask)

md_gt    = load_3d_map(f"Results/MD_GT{SUFFIX}.nii.gz",    mask)
md_unet  = load_3d_map(f"Results/MD_UNet{SUFFIX}.nii.gz",  mask)

# Gradienten-Tabelle (b-Werte und Richtungsvektoren) für den DTI-Fit laden
bvals = np.loadtxt(f"Results/DSI_GT{SUFFIX}.bval").flatten()
bvecs = np.loadtxt(f"Results/DSI_GT{SUFFIX}.bvec").T
gtab = gradient_table(bvals=bvals, bvecs=bvecs)

print("Berechne Diffusionstensor-Eigenvektoren & farbkodierte FA-Karten (GT & U-Net)...")
rgb_gt,   fa_fit_gt,   v1_gt   = compute_color_fa_and_v1(f"Results/DSI_GT{SUFFIX}.nii.gz",   gtab, mask)
rgb_unet, fa_fit_unet, v1_unet = compute_color_fa_and_v1(f"Results/DSI_UNet{SUFFIX}.nii.gz", gtab, mask)

# Winkelabweichung der Hauptfaserrichtung nur in der weißen Substanz (FA > 0.2) auswerten
inner_mask = binary_erosion(mask, iterations=3)
wm_mask = inner_mask & (fa_fit_gt > 0.5) & (fa_fit_gt < 0.99)
ang_err_mean, ang_err_median = compute_angular_error_deg(v1_unet, v1_gt, wm_mask)
print(
    f"Winkelabweichung der Hauptfaserrichtung v1 (Weiße Substanz, FA > 0.2): "
    f"Mittelwert = {ang_err_mean:.2f}° | Median = {ang_err_median:.2f}°"
)


# =============================================================================
# 4. SCHICHT-INDEX & BILDAUSSCHNITT BESTIMMEN
# =============================================================================
# Zuordnung der 3D-Achsen im NIfTI-Koordinatensystem (X=sagittal, Y=coronal, Z=axial)
view_axes = {"sagittal": 0, "coronal": 1, "axial": 2}
s_ax = view_axes[VIEW]

if SLICE_IDX is not None:
    idx = SLICE_IDX
elif TARGET_MM is not None:
    # Millimeter-Koordinate über die inverse Affine-Matrix in den Voxel-Index umrechnen
    pt = [0.0, 0.0, 0.0]
    pt[s_ax] = TARGET_MM
    idx = int(round(nib.affines.apply_affine(np.linalg.inv(affine), pt)[s_ax]))
else:
    # Automatisch die Mitte des Gehirns entlang der gewählten Achse wählen
    idx = int(np.median(np.where(mask)[s_ax]))


def extract_2d_slice(vol):
    """
    Schneidet die gewählte 2D-Schicht heraus und transponiert sie passend für
    plt.imshow(..., origin='lower'). Funktioniert für 3D-Graustufen- und 4D-RGB-Volumina.
    """
    sl = np.take(vol, idx, axis=s_ax)
    if sl.ndim == 3:  # RGB-Bild: (Breite, Höhe, 3) -> (Höhe, Breite, 3)
        return np.transpose(sl, (1, 0, 2))
    return sl.T       # Graustufenbild: (Breite, Höhe) -> (Höhe, Breite)


# Bildausschnitt festlegen (entweder manuelle ZOOM_BOX oder knapp um die Gehirnmaske)
mask_sl = extract_2d_slice(mask)
if ZOOM_BOX is not None:
    xlim = (ZOOM_BOX[0], ZOOM_BOX[1])
    ylim = (ZOOM_BOX[2], ZOOM_BOX[3])
else:
    nz_v, nz_h = np.where(mask_sl)
    pad = 5  # 5 Voxel schwarzer Rand rund ums Gehirn
    xlim = (max(0, nz_h.min() - pad), min(mask_sl.shape[1] - 1, nz_h.max() + pad))
    ylim = (max(0, nz_v.min() - pad), min(mask_sl.shape[0] - 1, nz_v.max() + pad))


# =============================================================================
# 5. EINZIGEN 2x4-PLOT ERSTELLEN (MIT EXAKT GLEICH GROSSEN BILDERN)
# =============================================================================
# Gemeinsame Obergrenze für die DWI-Grauskala anhand des 99. Perzentils der Ground Truth
dwi_vmax = float(np.percentile(dwi_gt[mask], 99))
dwi_title = "Mean DWI (a.u.)" if DWI_DIR_IDX is None else f"DWI Dir {DWI_DIR_IDX} (a.u.)"

# Definition der 4 Spalten:
# is_rgb=False -> Graustufenbild mit eigener vertikaler Farbleiste (Colorbar)
# is_rgb=True  -> RGB-Bild (farbkodierte FA), keine Graustufen-Farbleiste nötig
columns = [
    {
        "title": dwi_title,
        "gt": dwi_gt, "pred": dwi_unet,
        "is_rgb": False, "vmin": 0.0, "vmax": dwi_vmax, "fmt": "%.0f",
    },
    {
        "title": "FA",
        "gt": fa_gt, "pred": fa_unet,
        "is_rgb": False, "vmin": 0.0, "vmax": 0.95, "fmt": "%.1f",
    },
    {
        "title": "Color FA",
        "gt": np.clip(rgb_gt * COLOR_FA_GAIN, 0.0, 1.0),
        "pred": np.clip(rgb_unet * COLOR_FA_GAIN, 0.0, 1.0),
        "is_rgb": True,
        "pred_note": f"WM Angular Error: {ang_err_mean:.1f}°",
    },
    {
        "title": r"MD ($\mathrm{mm}^2/\mathrm{s}$)",
        "gt": md_gt, "pred": md_unet,
        "is_rgb": False, "vmin": 0.0, "vmax": 0.0022, "fmt": "%.4f",
    },
]

# GridSpec-Layout mit 4 Zeilen und dedizierten Spalten für Bilder, Farbleisten und Abstände:
# - Zeilen 0..1 = oberes Bild (Ground Truth), Zeilen 2..3 = unteres Bild (U-Net)
# - Zeilen 1..2 = mittig zentrierte Farbleiste (exakt 50 % Gesamthöhe, wie in deiner Vorlage)
# Dadurch bleiben alle 8 Gehirnbilder exakt gleich groß, obwohl Color FA keine Colorbar hat.
fig = plt.figure(figsize=(16.5, 7.2), dpi=300)
gs = GridSpec(
    nrows=4,
    ncols=10,
    figure=fig,
    width_ratios=[1.0, 0.055, 0.24, 1.0, 0.055, 0.24, 1.0, 0.16, 1.0, 0.055],
    wspace=0.06,
    hspace=0.10,
)

# Spalten-Indizes im GridSpec: (Bild-Spalte, Colorbar-Spalte oder None)
grid_positions = [
    (0, 1),     # Spalte 1: DWI + Colorbar
    (3, 4),     # Spalte 2: FA + Colorbar
    (6, None),  # Spalte 3: Color FA (RGB, ohne Colorbar)
    (8, 9),     # Spalte 4: MD + Colorbar
]

for col_idx, (col_cfg, (img_col, cb_col)) in enumerate(zip(columns, grid_positions)):
    ax_top = fig.add_subplot(gs[0:2, img_col])
    ax_bot = fig.add_subplot(gs[2:4, img_col])

    sl_gt   = extract_2d_slice(col_cfg["gt"])
    sl_pred = extract_2d_slice(col_cfg["pred"])

    if col_cfg["is_rgb"]:
        ax_top.imshow(sl_gt,   origin="lower")
        ax_bot.imshow(sl_pred, origin="lower")

        # Dezente Angabe des Angular Errors direkt in der linken unteren Ecke des U-Net-Bildes
        ax_bot.text(
            0.03, 0.04, col_cfg["pred_note"],
            transform=ax_bot.transAxes,
            color="white", fontsize=8.5, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.65, edgecolor="none", pad=2.5),
        )
    else:
        im_top = ax_top.imshow(
            sl_gt, origin="lower", cmap="gray",
            vmin=col_cfg["vmin"], vmax=col_cfg["vmax"]
        )
        ax_bot.imshow(
            sl_pred, origin="lower", cmap="gray",
            vmin=col_cfg["vmin"], vmax=col_cfg["vmax"]
        )

        # Vertikale Farbleiste mittig rechts neben den beiden Bildern der Spalte
        cax = fig.add_subplot(gs[1:3, cb_col])
        cbar = fig.colorbar(im_top, cax=cax, format=FormatStrFormatter(col_cfg["fmt"]))
        cbar.ax.tick_params(labelsize=9)

    # Spaltentitel über der oberen Zeile
    ax_top.set_title(col_cfg["title"], fontsize=12, fontweight="bold", pad=8)

    # Einheitlicher Ausschnitt und schwarzer Hintergrund ohne Achsen-Ticks
    for ax in (ax_top, ax_bot):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("black")

    # Zeilenbeschriftung links an der ersten Spalte
    if col_idx == 0:
        ax_top.set_ylabel("Ground Truth", fontsize=12, fontweight="bold", labelpad=10)
        ax_bot.set_ylabel("U-Net",        fontsize=12, fontweight="bold", labelpad=10)

# Dezente Farblegende für die Color-FA-Spalte unterhalb der Abbildung
fig.text(
    0.5, 0.02,
    "Color FA (Hauptfaserrichtung):  Rot = Links–Rechts (L–R)  |  Grün = Vorne–Hinten (A–P)  |  Blau = Oben–Unten (S–I)",
    ha="center", va="bottom", fontsize=9.5
)

out_path = f"Results/comparison_dwi_fa_colorfa_md_{VIEW}{SUFFIX}.png"
plt.savefig(out_path, bbox_inches="tight", dpi=300)
print(f"Vergleichsplot gespeichert unter: {out_path}")