import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# Dateipfade für das 4D-MRT-Bild und die 3D-Gehirnmaske
nifti_path = "/mnt/anarchy/lohrd/MRI_data/HCP/Diffusion/3T/100610/T1w/Diffusion/data.nii.gz"
mask_path = "/mnt/anarchy/lohrd/MRI_data/HCP/Diffusion/3T/100610/T1w/Diffusion/nodif_brain_mask.nii.gz"

print("Lade Daten...")
# Lade die Proxy-Objekte
img = nib.load(nifti_path)
mask_img = nib.load(mask_path)

# Berechne die mittlere Schicht auf der Z-Achse (Höhe)
mitte_z = img.shape[2] // 2

# Lade gezielt nur die mittlere Schicht des ERSTEN Volumens (Index 0, ein b0-Bild)
slice_b0_original = np.asanyarray(img.dataobj[:, :, mitte_z, 0])

# Lade die exakt gleiche Schicht aus der 3D-Gehirnmaske
slice_mask = np.asanyarray(mask_img.dataobj[:, :, mitte_z])

# Wende die Maske an: 
# Da die Maske aus 0 (Hintergrund) und 1 (Gehirn) besteht, 
# multiplizieren wir einfach das Bild mit der Maske. 
# Alles außerhalb des Gehirns wird so mit 0 multipliziert (schwarz).
slice_b0_masked = slice_b0_original * slice_mask

print("Erstelle Plot...")
# Erstelle eine Abbildung mit zwei nebeneinanderliegenden Plots (1 Zeile, 2 Spalten)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot 1: Das rohe b0-Bild (vor der Maske)
# Wir nutzen vmax, um den Kontrast leicht zu erhöhen (schneidet extrem helle Rausch-Pixel ab)
vmax_val = np.percentile(slice_b0_original, 99) 
axes[0].imshow(slice_b0_original.T, cmap="gray", origin="lower", vmax=vmax_val)
axes[0].set_title("Vor der Maske (Original b0)")
axes[0].axis("off")

# Plot 2: Das maskierte b0-Bild (nach der Maske)
axes[1].imshow(slice_b0_masked.T, cmap="gray", origin="lower", vmax=vmax_val)
axes[1].set_title("Nach der Maske (Nur Gehirn)")
axes[1].axis("off")

# Speichere das Ergebnis
output_filename = "b0_mask_comparison.png"
plt.savefig(output_filename, bbox_inches="tight", dpi=150)
plt.close()

print(f"Fertig! Bild erfolgreich gespeichert als: {output_filename}")