import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting

# --- 1. PFADE ANPASSEN (Typische HCP 3T Struktur) ---
main_path = "/mnt/nas_io/anarchy/lohrd/MRI_data/HCP/Diffusion/3T/"
dwi_path = main_path + "100610/T1w/Diffusion/data.nii.gz"
bvals_path = main_path + "100610/T1w/Diffusion/bvals"
mask_path = main_path + "100610/T1w/Diffusion/nodif_brain_mask.nii.gz"
output_mask_path = "b0_top1_percent_mask.nii.gz"

# --- 2. DATEN & BVALS LADEN ---
print("Lade NIfTI-Daten...")
img = nib.load(dwi_path)
data = img.get_fdata()
bvals = np.loadtxt(bvals_path)

b0_indices = np.where(bvals < 100)[0]
print(f"Gefundene b0-Volumina ({len(b0_indices)} Stück) an Indizes: {b0_indices}")

# Mittelwert über alle b0-Bilder berechnen, um das Signal-Rausch-Verhältnis zu verbessern
b0_mean = np.mean(data[:, :, :, b0_indices], axis=-1)

# --- 3. 95. PERZENTIL (TOP 5%) BERECHNEN ---
# Hintergrund ausschließen, da sonst ~70% Null-Voxel das Perzentil verfälschen
try:
    brain_mask = nib.load(mask_path).get_fdata() > 0
    print("Nutze nodif_brain_mask für die Perzentil-Berechnung...")
    valid_voxels = b0_mean[brain_mask]
except FileNotFoundError:
    print("Keine Maske gefunden. Schließe Hintergrund (Voxel <= 0) automatisch aus...")
    valid_voxels = b0_mean[b0_mean > 0]

# Top 5% der größten Werte entspricht dem 95. Perzentil
threshold_95 = np.percentile(valid_voxels, 99)
max_val = np.max(valid_voxels)
print(f"95. Perzentil Schwellenwert: {threshold_95:.2f} (Maximalwert: {max_val:.2f})")

# Binäre Maske und gefiltertes Bild für die extremen Werte erstellen
extreme_mask = b0_mean >= threshold_95

# --- 4. ALS NIFTI SPEICHERN (für FSLeyes / Freeview / MRIcroGL) ---
extreme_img = nib.Nifti1Image(extreme_mask.astype(np.uint8), img.affine, img.header)
nib.save(extreme_img, output_mask_path)
print(f"Maske der extremen Werte gespeichert unter: {output_mask_path}")

# --- 5. AUTOMATISCHE VISUALISIERUNG DES HOTSPOTS ---
# Finde die Z-Schicht (axial) mit den meisten extremen Voxeln
z_counts = np.sum(extreme_mask, axis=(0, 1))
best_z_slice = np.argmax(z_counts)

# Erstelle ein NIfTI-Objekt des gemittelten b0-Bildes für Nilearn
b0_mean_img = nib.Nifti1Image(b0_mean, img.affine)

# Plotte das b0-Bild mit den Top 5% Werten als farbiges Overlay
display = plotting.plot_roi(
    roi_img=extreme_img,
    bg_img=b0_mean_img,
    title=f"Top 1% b0-Intensitäten (> {threshold_95:.0f})",
    cmap="autumn",
    alpha=0.8,
    cut_coords=None, # Zentriert automatisch auf die größten Cluster
    display_mode="ortho"
)
plt.show()