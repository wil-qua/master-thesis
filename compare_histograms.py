import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

number_of_files = 5

def load_masked_b0(data_path, bval_path, mask_path):
    """Lädt die Daten, extrahiert alle b0-Bilder, bildet den Mittelwert und wendet die Gehirnmaske an."""
    print(f"Lade Daten von: {data_path}")
    
    # 1. bvals laden und b0-Indizes finden (b-Wert < 100)
    bvals = np.loadtxt(bval_path).flatten()
    b0_indices = np.where(bvals < 100)[0]
    
    # 2. Nur die b0-Volumina iterativ laden
    img = nib.load(data_path)
    b0_volumes_list = []
    for idx in b0_indices:
        vol = np.asanyarray(img.dataobj[..., idx], dtype=np.float32)
        b0_volumes_list.append(vol)
        
    b0_volumes = np.stack(b0_volumes_list, axis=-1)
    
    # 3. Mittelwert über alle b0-Bilder bilden (reduziert Rauschen)
    mean_b0 = np.mean(b0_volumes, axis=-1)
    
    # 4. Gehirnmaske laden und anwenden
    mask = nib.load(mask_path).get_fdata()
    
    # Extrahiere nur die Voxel, die innerhalb der Maske liegen und größer als 0 sind
    brain_voxels = mean_b0[(mask > 0) & (mean_b0 > 0)]
    return brain_voxels

def main():
    # Liste Probanden.
    subjects = os.listdir("/home/wquan/Schreibtisch/Data/Diffusion/3T")[:number_of_files]
    print(subjects)
    
    base_path_3t = "/home/wquan/Schreibtisch/Data/Diffusion/3T"
    base_path_7t = "/home/wquan/Schreibtisch/Data/Diffusion/7T"
    
    plt.figure(figsize=(12, 7))
    
    # Flags für die Legende, damit "3T" und "7T" nicht pro Proband mehrfach auftauchen
    label_3t_added = False
    label_7t_added = False

    for subj in subjects:
        print(f"--- Verarbeite Proband {subj} ---")
        
        # 3T Verarbeitung
        path_3t = os.path.join(base_path_3t, subj, "T1w", "Diffusion")
        if os.path.exists(path_3t):
            voxels_3t = load_masked_b0(
                os.path.join(path_3t, "data.nii.gz"),
                os.path.join(path_3t, "bvals"),
                os.path.join(path_3t, "nodif_brain_mask.nii.gz")
            )
            # Zeichne das Histogramm (als Linie statt als Balken für bessere Lesbarkeit)
            plt.hist(voxels_3t, bins=200, density=True, histtype='step', 
                     color='blue', alpha=0.6, linewidth=1.5,
                     label="3T" if not label_3t_added else "")
            label_3t_added = True

        # 7T Verarbeitung
        path_7t = os.path.join(base_path_7t, subj, "T1w", "Diffusion_7T")
        if os.path.exists(path_7t):
            voxels_7t = load_masked_b0(
                os.path.join(path_7t, "data.nii.gz"),
                os.path.join(path_7t, "bvals"),
                os.path.join(path_7t, "nodif_brain_mask.nii.gz")
            )
            plt.hist(voxels_7t, bins=200, density=True, histtype='step', 
                     color='red', alpha=0.6, linewidth=1.5,
                     label="7T" if not label_7t_added else "")
            label_7t_added = True

    plt.title(f"Intensity Distribution (Histogram): 3T vs 7T b0-Images ({number_of_files} measurements)", fontsize=14)
    plt.xlabel("Voxel Intensity (a.u.)", fontsize=12)
    plt.ylabel("Density (Normalized Frequency)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    
    # X-Achse limitieren, um extreme Ausreißer (z.B. Blutgefäße) auszublenden und Fokus auf Gewebe zu legen
    plt.xlim(0, 15000) # Ggf. anpassen, je nach Skalierung der HCP-Daten
    
    ausgabe_datei = "histogramm_3T_vs_7T.png"
    plt.savefig(ausgabe_datei, dpi=300, bbox_inches='tight')
    print(f"\nFertig! Histogramm gespeichert unter: {ausgabe_datei}")

if __name__ == "__main__":
    main()