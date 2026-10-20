
from pathlib import Path
import csv

import nibabel as nib
import numpy as np


# ============================================================
# Konfiguration: Hier die Pfade anpassen
# ============================================================

MAIN_PATH = "/mnt/nas_io/anarchy/lohrd/MRI_data/HCP/Diffusion/3T/100610/T1w/Diffusion/"
DWI_PATH = MAIN_PATH + "data.nii.gz"
BVAL_PATH = MAIN_PATH +  "bvals"
MASK_PATH = MAIN_PATH +  "nodif_brain_mask.nii.gz"

OUTPUT_PREFIX = "100610_shell_intensities"

B0_THRESHOLD = 100

SHELL_RANGES = {
    "b0": (0, 100),
    "b1000": (900, 1100),
    "b2000": (1800, 2200),
    "b3000": (2700, 3300),
}


# ============================================================
# Daten laden
# ============================================================

img = nib.load(DWI_PATH)
data = np.asarray(img.dataobj, dtype=np.float32)
bvals = np.loadtxt(BVAL_PATH).reshape(-1)

if data.ndim != 4:
    raise ValueError(f"Erwartetes 4D-DWI, erhalten: {data.shape}")

if data.shape[3] != len(bvals):
    raise ValueError(
        f"{data.shape[3]} Bildvolumes, aber {len(bvals)} b-Werte."
    )

b0_indices = np.where(bvals <= B0_THRESHOLD)[0]

if len(b0_indices) == 0:
    raise ValueError("Keine b0-Volumen gefunden.")

# Mittelwert über alle b0-Volumen
mean_b0 = np.mean(data[..., b0_indices], axis=3)


# ============================================================
# Gehirnmaske laden oder einfache Maske erstellen
# ============================================================

if MASK_PATH:
    mask_img = nib.load(MASK_PATH)
    mask = np.asarray(mask_img.dataobj) > 0

    if mask.shape != data.shape[:3]:
        raise ValueError(
            f"Maskenform {mask.shape} passt nicht zu {data.shape[:3]}"
        )
else:
    print("WARNUNG: Keine Maske angegeben; verwende mean_b0 > 0.")
    mask = np.isfinite(mean_b0) & (mean_b0 > 0)

if not np.any(mask):
    raise ValueError("Die Gehirnmaske enthält keine gültigen Voxels.")


# ============================================================
# Mittlere Intensität pro Volume berechnen
# ============================================================

per_volume = []

for i, bval in enumerate(bvals):
    shell = "other"

    for shell_name, (b_min, b_max) in SHELL_RANGES.items():
        if b_min <= bval <= b_max:
            shell = shell_name
            break

    values = data[..., i][mask]
    values = values[np.isfinite(values)]

    if values.size == 0:
        continue

    per_volume.append({
        "volume_index": i,
        "bval": float(bval),
        "shell": shell,
        "mean_intensity": float(np.mean(values)),
        "median_intensity": float(np.median(values)),
    })


# ============================================================
# Zusammenfassung pro Shell
# ============================================================

summary = []

print(f"Bildgröße: {data.shape}")
print(f"Gehirnvoxel: {int(mask.sum())}")
print(f"Anzahl b0-Volumes: {len(b0_indices)}")
print()

print(
    f"{'Shell':<12} {'N':>4} {'Mean':>12} "
    f"{'Standard Deviation':>12} {'Median':>12}"
)
print("-" * 56)

shell_order = ["b0", "b1000", "b2000", "b3000", "other"]

for shell in shell_order:
    rows = [r for r in per_volume if r["shell"] == shell]

    if not rows:
        continue

    means = np.array(
        [r["mean_intensity"] for r in rows],
        dtype=np.float64,
    )
    medians = np.array(
        [r["median_intensity"] for r in rows],
        dtype=np.float64,
    )

    result = {
        "shell": shell,
        "n_volumes": len(rows),
        "mean_of_volume_means": float(np.mean(means)),
        "standard_deviation_of_volume_means": (
            float(np.std(means, ddof=1))
            if len(means) > 1 else 0.0
        ),
        "median_of_volume_medians": float(np.median(medians)),
    }

    summary.append(result)

    print(
        f"{shell:<12} {len(rows):>4d} "
        f"{result['mean_of_volume_means']:>12.4f} "
        f"{result['standard_deviation_of_volume_means']:>12.4f} "
        f"{result['median_of_volume_medians']:>12.4f}"
    )


# ============================================================
# Ergebnisse als CSV speichern
# ============================================================

per_volume_path = Path(f"{OUTPUT_PREFIX}_per_volume.csv")
summary_path = Path(f"{OUTPUT_PREFIX}_summary.csv")

if per_volume:
    with per_volume_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_volume[0].keys())
        writer.writeheader()
        writer.writerows(per_volume)

if summary:
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

print()
print(f"Einzelmessungen gespeichert: {per_volume_path}")
print(f"Shell-Zusammenfassung gespeichert: {summary_path}")
