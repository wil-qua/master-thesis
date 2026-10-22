import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

# 1. Daten und bvals laden
main_path = "/mnt/nas_io/anarchy/lohrd/MRI_data/HCP/Diffusion/3T/"
data_path = main_path + "100610/T1w/Diffusion/data.nii.gz"
bvals_path = main_path + "100610/T1w/Diffusion/bvals"
mask_path = main_path + "100610/T1w/Diffusion/nodif_brain_mask.nii.gz"
output_mask_path = "b0_top1_percent_mask.nii.gz"

img = nib.load(data_path)
data = img.get_fdata()  # 4D Array: (X, Y, Z, Zeit/Volumen)
bvals = np.loadtxt(bvals_path)

mask_img = nib.load(mask_path)
mask_data = mask_img.get_fdata()

# Prüfen, ob die Maske zum Bild passt
if data.shape[:3] != mask_data.shape:
    raise ValueError(
        f"Dimensionen passen nicht: Daten {data.shape[:3]}, "
        f"Maske {mask_data.shape}"
    )

# Bei gleicher Dimension zusätzlich Affine-Matrizen prüfen
if not np.allclose(img.affine, mask_img.affine, atol=1e-4):
    raise ValueError(
        "Die Affine-Matrizen unterscheiden sich. "
        "Prüfe, ob die Maske im selben Raum wie die Daten liegt."
    )

mask = mask_data > 0

# b0-Volumes auswählen
b0_indices = np.where(bvals < 10)[0]

# Zeitlicher Verlauf des mittleren b0-Signals
b0_means = []

for idx in b0_indices:
    vol = data[..., idx]
    valid = mask & np.isfinite(vol) & (vol > 0)
    b0_means.append(np.mean(vol[valid]))

b0_means = np.asarray(b0_means)

# Auf das erste b0-Volume normieren
relative_signal = 100 * b0_means / b0_means[0]

plt.figure(figsize=(9, 5))
plt.plot(
    np.arange(len(b0_indices)),
    relative_signal,
    marker="o"
)
plt.axhline(100, linestyle="--", color="gray")
plt.xlabel("b0-Volume (Reihenfolge)")
plt.ylabel("Mittleres Signal (% des ersten b0)")
plt.title("Signalverlauf der b0-Volumes")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('Figures/b0_Drift_Differenz.png', dpi=300)
