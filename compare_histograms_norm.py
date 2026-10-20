import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

number_of_files = 5

def load_and_normalize_masked_b0(data_path, bval_path, mask_path):
    bvals = np.loadtxt(bval_path).flatten()
    b0_indices = np.where(bvals < 100)[0]
    
    img = nib.load(data_path)
    b0_volumes_list = []
    for idx in b0_indices:
        vol = np.asanyarray(img.dataobj[..., idx], dtype=np.float32)
        b0_volumes_list.append(vol)
        
    b0_volumes = np.stack(b0_volumes_list, axis=-1)
    mean_b0 = np.mean(b0_volumes, axis=-1)
    mask = nib.load(mask_path).get_fdata()
    
    brain_voxels = mean_b0[(mask > 0) & (mean_b0 > 0)]
    
    # 95. Perzentil Normalisierung (wie in dataset.py)
    scalar = np.percentile(brain_voxels, 95)
    normalized_voxels = brain_voxels / scalar
    
    return normalized_voxels

def main():
    subjects = ["100610", "100307", "100408", "101107", "101309"] 
    
    base_path_3t = "/home/wquan/Schreibtisch/Data/Diffusion/3T"
    base_path_7t = "/home/wquan/Schreibtisch/Data/Diffusion/7T"
    
    plt.figure(figsize=(12, 7))
    
    all_voxels_3t = []
    all_voxels_7t = []

    for subj in subjects:
        print(f"Verarbeite Proband {subj}...")
        
        # 3T
        path_3t = os.path.join(base_path_3t, subj, "T1w", "Diffusion")
        if os.path.exists(path_3t):
            voxels_3t = load_and_normalize_masked_b0(
                os.path.join(path_3t, "data.nii.gz"),
                os.path.join(path_3t, "bvals"),
                os.path.join(path_3t, "nodif_brain_mask.nii.gz")
            )
            all_voxels_3t.append(voxels_3t)
            plt.hist(voxels_3t, bins=200, density=True, histtype='step', 
                     color='blue', alpha=0.15, linewidth=1)

        # 7T
        path_7t = os.path.join(base_path_7t, subj, "T1w", "Diffusion_7T")
        if os.path.exists(path_7t):
            voxels_7t = load_and_normalize_masked_b0(
                os.path.join(path_7t, "data.nii.gz"),
                os.path.join(path_7t, "bvals"),
                os.path.join(path_7t, "nodif_brain_mask.nii.gz")
            )
            all_voxels_7t.append(voxels_7t)
            plt.hist(voxels_7t, bins=200, density=True, histtype='step', 
                     color='red', alpha=0.15, linewidth=1)

    print("Berechne und zeichne globale Durchschnitte...")
    
    if all_voxels_3t:
        global_3t = np.concatenate(all_voxels_3t)
        plt.hist(global_3t, bins=200, density=True, histtype='step', 
                 color='blue', alpha=1.0, linewidth=2.5, label="3T (Global, Normalized)")
                 
    if all_voxels_7t:
        global_7t = np.concatenate(all_voxels_7t)
        plt.hist(global_7t, bins=200, density=True, histtype='step', 
                 color='red', alpha=1.0, linewidth=2.5, label="7T (Global, Normalized)")

    plt.title(f"Intensity Distribution (Histogram): 3T vs 7T b0-Images ({number_of_files} measurements)", fontsize=14)
    plt.xlabel("Voxel Intensity (a.u.)", fontsize=12)
    plt.ylabel("Density (Normalized Frequency)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    
    # Wertebereich liegt nun fast ausschließlich zwischen 0 und 1.5
    plt.xlim(0, 1.5)
    
    ausgabe_datei = "histogramm_3T_vs_7T_normalisiert.png"
    plt.savefig(ausgabe_datei, dpi=300, bbox_inches='tight')
    print(f"Fertig! Bild gespeichert unter: {ausgabe_datei}")

if __name__ == "__main__":
    main()