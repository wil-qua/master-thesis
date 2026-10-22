# dataset.py
"""
Modul zur Datenvorbereitung und PyTorch-Dataset-Generierung.
Umfasst Sampling von Diffusions-MRT-Daten, Signalnormalisierung 
sowie Extraktion von High-Resolution (HR) und Low-Resolution (LR) Patches.
"""

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter, affine_transform
from scipy.spatial.transform import Rotation as R


class FlexibleDiffusionSampler:
    """
    Wählt eine variable Anzahl an b0- und DWI-Bildern aus.
    DWI-Bilder werden via 'Electrostatic Repulsion' Algorithmus gezogen, 
    um eine optimale sphärische Gleichverteilung der Gradienten zu garantieren.
    """
    def __init__(self, bvals, bvecs, b0_threshold=100, min_dwi_b=900, max_dwi_b=1100):
        all_bvals = np.array(bvals)
        all_bvecs = np.array(bvecs)
        
        # Indizes für b0 und Ziel-Schale
        self.global_b0_indices = np.where(all_bvals <= b0_threshold)[0]
        self.global_dwi_indices = np.where((all_bvals >= min_dwi_b) & (all_bvals <= max_dwi_b))[0]
        
        self.b0_bvals = all_bvals[self.global_b0_indices]
        self.dwi_bvals = all_bvals[self.global_dwi_indices]
        self.dwi_bvecs = all_bvecs[self.global_dwi_indices]
        
        self.num_total_b0 = len(self.b0_bvals)
        self.num_total_dwi = len(self.dwi_bvals)

    def _electrostatic_subset_selection(self, pool_indices, n_select, rng=None, refine_steps=3):
        """
        Wählt n_select Richtungen aus dem Pool für maximale sphärische Gleichverteilung:
        1. Initialisierung via antipodalem Farthest-Point-Sampling (verhindert enge Cluster).
        2. Nachoptimierung (Swapping) über das exakte elektrostatische Jones-Potenzial.
        """
        if rng is None:
            rng = np.random

        pool_bvecs = self.dwi_bvecs[pool_indices]
        norms = np.linalg.norm(pool_bvecs, axis=1, keepdims=True)
        pool_bvecs = pool_bvecs / np.clip(norms, a_min=1e-8, a_max=None)  # Normierung
        num_pool = len(pool_indices)

        # Paarweise antipodale Skalarprodukte |u * v| vorab berechnen: Shape [N, N]
        abs_dots = np.clip(np.abs(np.dot(pool_bvecs, pool_bvecs.T)), 0.0, 1.0 - 1e-7)

        # Exakte elektrostatische Paar-Energie (Jones et al., 1999):
        # E_ij = 1 / ||u_i - u_j|| + 1 / ||u_i + u_j||
        # ||u ± v|| = sqrt(2 ± 2(u*v))
        dist_minus = np.sqrt(2.0 - 2.0 * abs_dots)
        dist_plus  = np.sqrt(2.0 + 2.0 * abs_dots)
        pair_energy = (1.0 / dist_minus) + (1.0 / dist_plus)
        np.fill_diagonal(pair_energy, 0.0)

        # --- SCHRITT 1: Farthest Point Sampling (Max-Min Winkelabstand) ---
        # Verhindert zuverlässig, dass zwei Punkte zu nah beieinander landen
        selected = [int(rng.choice(num_pool))]
        for _ in range(1, n_select):
            # Für jeden Kandidaten das maximale |u * v| (= kleinster Winkel) zu bisherigen Punkten finden
            # (Worst-Case-Check) "engster Nachbar"
            max_dot_to_selected = np.max(abs_dots[:, selected], axis=1)
            max_dot_to_selected[selected] = 2.0  # Bereits gewählte ausschließen
            
            # Wähle den Vektor, dessen nächster Nachbar am weitesten entfernt ist (kleinstes max |u*v|)
            best_next = int(np.argmin(max_dot_to_selected))
            selected.append(best_next)

        selected = np.array(selected, dtype=int)

        # --- SCHRITT 2: Elektrostatische Nachoptimierung (Coordinate Swapping) ---
        # Teste, ob ausgesuchte Vektoren wirklich die besten sind.
        # Tauscht iterativ Punkte gegen bessere Kandidaten aus dem Rest-Pool aus.
        for _ in range(refine_steps):
            swapped = False
            for i in range(n_select):
                others = np.delete(selected, i)
                # Energie jedes Pool-Kandidaten relativ zu den (n_select - 1) anderen gewählten Punkten
                candidate_energies = np.sum(pair_energy[:, others], axis=1)
                candidate_energies[others] = np.inf  # Bereits enthaltene Punkte sperren

                best_candidate = int(np.argmin(candidate_energies))
                if best_candidate != selected[i]:
                    selected[i] = best_candidate
                    swapped = True
            if not swapped:
                break

        return pool_indices[selected]

    def evaluate_spherical_distribution(self, bvecs, antipodal=True):
        """
        Berechnet die kleinsten Abstände (globales Minimum und Durchschnitt der 
        nächsten Nachbarn) für eine Menge von 3D-Vektoren auf einer Kugeloberfläche.
        
        Parameter:
        - bvecs: Array der Shape (N, 3) mit den ausgewählten Vektoren.
        - antipodal: Bool (Standard: True). Wenn True, werden u und -u als 
                    identische Richtungen behandelt (wichtig für DWI bvecs).
                    
        Rückgabe:
        - Dictionary mit Winkel- und euklidischen Abstandsmetriken sowie den 
        Indizes des engsten Vektorpaares.
        """
        bvecs = np.asarray(bvecs, dtype=float)
        if bvecs.ndim != 2 or bvecs.shape[1] != 3:
            raise ValueError(f"Erwarte Shape (N, 3), erhalten: {bvecs.shape}")
            
        n_points = len(bvecs)
        if n_points < 2:
            raise ValueError("Mindestens 2 Punkte sind für einen Abstand nötig.")

        # 1. Normierung auf die Einheitskugel (analog zu deiner Funktion)
        norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
        u = bvecs / np.clip(norms, a_min=1e-8, a_max=None)

        # 2. Paarweise Skalarprodukte berechnen: Shape (N, N)
        dots = np.dot(u, u.T)
        if antipodal:
            # Bei DWI-Gradienten zählt der kleinere Winkel zu u oder -u
            dots = np.abs(dots)
            
        # Numerisches Clipping für arccos und Wurzel
        dots = np.clip(dots, -1.0, 1.0)

        # 3. Winkelabstand in Grad berechnen
        angles_deg = np.degrees(np.arccos(dots))
        # Diagonale (Abstand zu sich selbst = 0) auf Unendlich setzen, 
        # damit sie beim Minimum ignoriert wird
        np.fill_diagonal(angles_deg, np.inf)

        # 4. Euklidische Distanz (Sehnenabstand ||u - v||) berechnen
        # Nutzt die Identität ||u - v|| = sqrt(2 - 2*(u*v))
        chord_dists = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * dots))
        np.fill_diagonal(chord_dists, np.inf)

        # 5. Für jeden Punkt den Abstand zum jeweils nächsten Nachbarn finden
        nearest_neighbor_angles = np.min(angles_deg, axis=1)
        nearest_neighbor_chords = np.min(chord_dists, axis=1)

        # 6. Globales Minimum und das zugehörige Vektor-Paar bestimmen
        min_idx_flat = np.argmin(angles_deg)
        pair_i, pair_j = np.unravel_index(min_idx_flat, angles_deg.shape)

        return {
            # --- Hauptmetrik: Das globale Minimum (Worst-Case-Paar) ---
            "global_min_angle_deg": float(angles_deg[pair_i, pair_j]),
            "global_min_chord_dist": float(chord_dists[pair_i, pair_j]),
            "closest_pair_indices": (int(pair_i), int(pair_j)),
            
            # --- Zusatzmetrik: Durchschnittlicher Abstand zum nächsten Nachbarn ---
            "mean_nn_angle_deg": float(np.mean(nearest_neighbor_angles)),
            "std_nn_angle_deg": float(np.std(nearest_neighbor_angles)),
            "mean_nn_chord_dist": float(np.mean(nearest_neighbor_chords)),
        }

    def plot_bvecs_on_sphere(self, bvecs):
        """
        Generiert eine interaktive 3D-HTML-Datei der Diffusionsvektoren.
        Ignoriert b0-Bilder.
        """
        norms = np.linalg.norm(bvecs, axis=1)

        # Filtere b0-Bilder raus
        dwi_bvecs = bvecs[norms > 0.1]
        dwi_bvecs = dwi_bvecs / np.linalg.norm(dwi_bvecs, axis=1, keepdims=True)

        fig = go.Figure()
        # 1. Kugeloberfläche
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere, 
            opacity=0.05,  # Sehr transparent
            showscale=False, colorscale='Greys', hoverinfo='skip'
        ))

        # 2. Längengrade (Meridiane - von Pol zu Pol)
        # 12 Linien (alle 30 Grad)
        for u_val in np.linspace(0, 2 * np.pi, 13)[:-1]:
            v_line = np.linspace(0, np.pi, 50)
            x_line = np.cos(u_val) * np.sin(v_line)
            y_line = np.sin(u_val) * np.sin(v_line)
            z_line = np.cos(v_line)
            fig.add_trace(go.Scatter3d(
                x=x_line, y=y_line, z=z_line, mode='lines',
                line=dict(color='gray', width=1), hoverinfo='skip', showlegend=False
            ))

        # 3. Breitengrade (Parallelen - horizontal um die Kugel)
        # 7 Linien (Äquator + je 3 Richtung Nord/Süd)
        for v_val in np.linspace(0, np.pi, 9)[1:-1]:
            u_line = np.linspace(0, 2 * np.pi, 50)
            x_line = np.cos(u_line) * np.sin(v_val)
            y_line = np.sin(u_line) * np.sin(v_val)
            z_line = np.cos(v_val) * np.ones_like(u_line)
            fig.add_trace(go.Scatter3d(
                x=x_line, y=y_line, z=z_line, mode='lines',
                line=dict(color='gray', width=1), hoverinfo='skip', showlegend=False
            ))

        # 4. Gezogene Richtungen (Rot)
        xs, ys, zs = dwi_bvecs[:, 0], dwi_bvecs[:, 1], dwi_bvecs[:, 2]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode='markers',
            marker=dict(size=6, color='red'), name='Gezogene Richtungen (v)'
        ))

        # 5. Antipodale Richtungen (Blau)
        fig.add_trace(go.Scatter3d(
            x=-xs, y=-ys, z=-zs, mode='markers',
            marker=dict(size=6, color='blue', opacity=0.4), name='Antipodale Richtungen (-v)'
        ))

        # Layout anpassen
        fig.update_layout(
            title=f"Elektrostatische Verteilung ({len(dwi_bvecs)} Richtungen)",
            scene=dict(
                xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        html_file = "Figures/interactive_bvecs.html"
        fig.write_html(html_file)
        print(f"Interaktive Kugel gespeichert unter: {html_file}")

    def sample(self, num_targets=1, plot=False, rng=None, n_dwi=None):
        """
        Zieht DWI-Bilder (optimiert) und b0-Bilder (zufällig).
        """
        if rng is None:
            rng = np.random

        if n_dwi is None:
            n_dwi = rng.randint(6, 31)
        max_b0 = max(1, n_dwi // 6)
        n_b0 = rng.randint(1, min(max_b0, self.num_total_b0) + 1) if max_b0 > 1 else 1
            
        all_local_dwi = np.arange(self.num_total_dwi)
        
        # Elektrostatisch optimierte Ziehung für DWI
        input_indices = self._electrostatic_subset_selection(all_local_dwi, n_dwi, rng=rng)

        # Disjunkte Trennung: Rest-DWIs als withheld targets
        remaining_dwi_indices = np.setdiff1d(all_local_dwi, input_indices, assume_unique=True)
        withheld_target_indices = rng.choice(
            remaining_dwi_indices, size=num_targets, replace=False
        )
        
        if plot:
            self.plot_bvecs_on_sphere(self.dwi_bvecs[input_indices])
        
        # b0-Bilder können weiterhin random gezogen werden, da sie keine Richtungs-Info (bvec=0,0,0) haben
        selected_b0 = rng.choice(self.num_total_b0, size=n_b0, replace=False)
        
        return {
            'b0_indices': selected_b0,
            'input_indices': input_indices,
            'withheld_target_indices': withheld_target_indices,
            'n_dwi': n_dwi,
            'n_b0': n_b0,
            'b0_bvals': self.b0_bvals[selected_b0],
            'input_bvals': self.dwi_bvals[input_indices],
            'input_bvecs': self.dwi_bvecs[input_indices],
            'target_bvals': self.dwi_bvals[withheld_target_indices],
            'target_bvecs': self.dwi_bvecs[withheld_target_indices]
        }

def calculate_normalization_scalar(b0_volumes, brain_mask):
    """
    Berechnet den Skalierungsfaktor für die Normalisierung basierend auf dem
    95. Perzentil der b0-Signale innerhalb der Gehirnmaske.
    """
    if b0_volumes.ndim == 3:
        # Einzelnes b0-Volumen: Shape (X, Y, Z)
        mean_b0 = b0_volumes
    elif b0_volumes.ndim == 4:
        # Mehrere b0-Volumes: Shape (X, Y, Z, N)
        mean_b0 = np.mean(b0_volumes, axis=-1)
    # Nur Voxel berücksichtigen, die in der Maske liegen und positive Signale haben
    valid_voxels = mean_b0[(brain_mask > 0) & (mean_b0 > 0)]
    scalar = np.percentile(valid_voxels, 95)
    return float(scalar)

def load_multi_subjects(nifti_paths, bval_paths, bvec_paths, mask_paths, b0_threshold, min_b_val, max_b_val):
    """
    Lädt die Daten aller Probanden in den RAM-Cache und normalisiert
    jeden Probanden anhand des 95. Perzentils seines ersten b0-Volumens
    innerhalb der Gehirnmaske.
    """
    subjects_cache = []
    
    for i, (n_path, bval_p, bvec_p, m_path) in enumerate(zip(nifti_paths, bval_paths, bvec_paths, mask_paths)):
        print(f"Lade Proband {i+1}/{len(nifti_paths)}: {n_path} ...")
        bvals = np.loadtxt(bval_p).flatten()
        bvecs = np.loadtxt(bvec_p).T
        if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
            bvecs = bvecs.T
        
        sampler = FlexibleDiffusionSampler(bvals, bvecs, b0_threshold, min_b_val, max_b_val)
        
        mask_img = nib.load(m_path)
        brain_mask = np.asanyarray(mask_img.dataobj)
        
        img = nib.load(n_path)
        full_data_4d = np.asanyarray(img.dataobj, dtype=np.float32)
        
        b0_4d_array = full_data_4d[..., sampler.global_b0_indices]
        dwi_4d_array = full_data_4d[..., sampler.global_dwi_indices]

        scalar = calculate_normalization_scalar(b0_4d_array, brain_mask)
        
        subjects_cache.append({
            'b0': b0_4d_array / scalar,
            'dwi': dwi_4d_array / scalar,
            'mask': brain_mask,
            'sampler': sampler
        })
        
    print(f"Alle {len(subjects_cache)} Probanden erfolgreich geladen und gecached.")
    return subjects_cache

def extract_random_patch(hr_data_b0, hr_data_dwi, mask, b0_indices, input_indices, withheld_target_indices, patch_size=32, fixed_center=None):
    """
    Sucht eine gültige Zentrumskoordinate innerhalb der Gehirnmaske und 
    extrahiert einen 3D-Block (Patch) aus den getrennten b0- und DWI-Tensoren.
    """
    x_dim, y_dim, z_dim, _ = hr_data_dwi.shape
    half_patch = patch_size // 2
    
    if fixed_center is not None:
        cx, cy, cz = fixed_center
    else:
        valid_coords = np.where(mask > 0)
        num_valid = len(valid_coords[0])
        
        while True:
            idx = np.random.randint(0, num_valid)
            cx = valid_coords[0][idx]
            cy = valid_coords[1][idx]
            cz = valid_coords[2][idx]
            
            if (cx - half_patch >= 0 and cx + half_patch <= x_dim and
                cy - half_patch >= 0 and cy + half_patch <= y_dim and
                cz - half_patch >= 0 and cz + half_patch <= z_dim):
                break 
            
    # Patches aus b0 und DWI herausschneiden
    hr_patch_b0 = hr_data_b0[
        cx - half_patch : cx + half_patch, 
        cy - half_patch : cy + half_patch, 
        cz - half_patch : cz + half_patch, 
        b0_indices                                  
    ]
    
    hr_patch_dwi = hr_data_dwi[
        cx - half_patch : cx + half_patch, 
        cy - half_patch : cy + half_patch, 
        cz - half_patch : cz + half_patch, 
        input_indices                                  
    ]
    
    hr_target = hr_data_dwi[
        cx - half_patch : cx + half_patch, 
        cy - half_patch : cy + half_patch, 
        cz - half_patch : cz + half_patch, 
        withheld_target_indices
    ]

    hr_mask = mask[
        cx - half_patch : cx + half_patch, 
        cy - half_patch : cy + half_patch, 
        cz - half_patch : cz + half_patch
    ]
    
    return hr_patch_b0, hr_patch_dwi, hr_target, hr_mask

def sample_random_90deg_rotation(rng=None):
    """
    Zieht zufällig eine der 24 echten 3D-90°-Rotationen (det(R) = +1, keine Spiegelung).
    Gibt zurück:
        perm:       Tuple der Länge 3 (Achsenpermutation für np.transpose)
        signs:      Tuple der Länge 3 aus {-1, +1} (Flip-Richtung pro neuer Achse)
        rot_matrix:Exakte 3x3 Rotationsmatrix (float32) für die bvecs
    """
    if rng is None:
        rng = np.random

    # Alle 6 Permutationen der 3 Raumachsen (0, 1, 2) mit ihrem Paritäts-Vorzeichen
    perms_with_parity = [
        ((0, 1, 2), +1),
        ((0, 2, 1), -1),
        ((1, 0, 2), -1),
        ((1, 2, 0), +1),
        ((2, 0, 1), +1),
        ((2, 1, 0), -1),
    ]
    perm, parity = perms_with_parity[int(rng.randint(0, 6))]

    # Erste zwei Vorzeichen frei wählen (+1 oder -1), das dritte so setzen, dass det(R) = +1
    s0 = int(rng.choice([-1, 1]))
    s1 = int(rng.choice([-1, 1]))
    s2 = parity * s0 * s1
    signs = (s0, s1, s2)

    # Zugehörige 3x3 Rotationsmatrix für die Gradienten (bvecs) aufbauen
    rot_matrix = np.zeros((3, 3), dtype=np.float32)
    for new_axis, old_axis in enumerate(perm):
        rot_matrix[new_axis, old_axis] = signs[new_axis]

    return perm, signs, rot_matrix


def apply_3d_90deg_rotation(volume, perm, signs):
    """
    Wendet die 90°-Rotation komplett verlustfrei (ohne Interpolation!) 
    auf ein 3D-Array [D, H, W] oder 4D-Array [D, H, W, C] an.
    """
    if volume.ndim == 4:
        out = np.transpose(volume, (perm[0], perm[1], perm[2], 3))
    else:
        out = np.transpose(volume, perm)

    flip_axes = [ax for ax in range(3) if signs[ax] == -1]
    if flip_axes:
        out = np.flip(out, axis=flip_axes)

    # .copy() sorgt für zusammenhängenden Speicher (wichtig für PyTorch und scipy)
    return out.copy()

def generate_and_augment_lr_patch(
    hr_patch, 
    hr_mask, 
    bvecs, 
    rot_spec=None, 
    downscale_factor=2, 
    sigma=0.8, 
    noise_std=0.01
):
    """
    - Verlustfreie 90°-Rotation von Bild, Maske und Gradienten (bvecs)
    - Fester Blur und Downsampling als Clean Baseline
    - Bounded Input Noise NACH der sauberen Baseline

    Note:
    MONAI ist primär für 3D-Volumina (wie T1w/T2w) ausgelegt.
    Diffusionsdaten sind aber 4D-Tensoren [D, H, W, N_directions],
    bei denen jede Richtung eine andere physikalische Gradientenrichtung hat.
    Man müsste dann bvecs mit drehen und dafür hat MONAI keinen fertigen wrapper. 
    """
    augmented_bvecs = bvecs.copy()
    
    # --- 1. KONSISTENTE, VERLUSTFREIE 90°-ROTATION AUF HR-EBENE ---
    if rot_spec is not None:
        perm, signs, rot_matrix = rot_spec
        hr_patch = apply_3d_90deg_rotation(hr_patch, perm, signs)
        hr_mask = apply_3d_90deg_rotation(hr_mask, perm, signs)
        augmented_bvecs = np.dot(bvecs, rot_matrix.T)

    # --- 2. DEGRADATION (Blur + Subsampling) ---
    sigma_spatial = [sigma, sigma, sigma, 0.0]
    blurred_patch = gaussian_filter(hr_patch, sigma=sigma_spatial)  # Weichzeichner/Tiefpassfilter, Monai bietet hier GaussianSmoothd für räumlichen Gauß-Blur.
    lr_patch = blurred_patch[::downscale_factor, ::downscale_factor, ::downscale_factor, :]

    # --- 3. PHYSICALLY REALISTIC dMRI NOISE (Rician Noise statt Clipped Gaussian) ---
    if noise_std > 0:
        # Optional wie in MONAI (sample_std=True): Rauschstärke zufällig zwischen 0 und noise_std ziehen
        current_std = np.random.uniform(0.0, noise_std)
        
        # Zwei unabhängige Gauß-Rauschfelder (in float32, um Speicher zu sparen)
        noise_real = np.random.normal(loc=0.0, scale=current_std, size=lr_patch.shape).astype(lr_patch.dtype)
        noise_imag = np.random.normal(loc=0.0, scale=current_std, size=lr_patch.shape).astype(lr_patch.dtype)
        
        # Betragsbildung (Magnituden-MRT): Automatisch >= 0, kein np.clip nötig
        lr_patch = np.sqrt((lr_patch + noise_real)**2 + noise_imag**2)
        
    return lr_patch, hr_mask, augmented_bvecs

def compute_ggT(bvec):
    """
    Berechnet die 6 eindeutigen Einträge der symmetrischen Matrix g*g^T.
    g ist der 3D-Vektor (gx, gy, gz).
    Rückgabe: [gx*gx, gy*gy, gz*gz, gx*gy, gx*gz, gy*gz]
    """
    gx, gy, gz = bvec
    return np.array([gx*gx, gy*gy, gz*gz, gx*gy, gx*gz, gy*gz], dtype=np.float32)

class HCPPatchDataset(Dataset):
    """
    PyTorch Dataset zur dynamischen Bereitstellung von HR/LR-Tensorpaaren
    für das Modell-Training über mehrere Probanden hinweg.
    """
    def __init__(self, subjects_cache, num_patches_per_epoch, patch_size, downscale_factor, max_measurements, num_targets=1, max_b0=5, is_validation=False):
        self.subjects_cache = subjects_cache
        self.num_patches = num_patches_per_epoch
        self.patch_size = patch_size
        self.downscale_factor = downscale_factor
        self.max_measurements = max_measurements
        self.num_targets = num_targets
        self.max_b0 = max_b0
        self.is_validation = is_validation
        
        self.frozen_val_specs = None
        if self.is_validation:
            self._freeze_validation_combinations()

    def _freeze_validation_combinations(self):
        """
        Für jeden Validierungs-Patch werden Proband, Patch-Koordinaten und gezogene DWI-/b0-Bilder einmalig
        bei der Dataset-Initialisierung festgelegt und bleiben über alle Epochen gleich.
        """
        rng = np.random.RandomState(42)
        half_patch = self.patch_size // 2
        protocol_dwi_counts = [6, 12, 18, 24, 30]  # legt fest, mit wie vielen DWI-input-directions die Validierungs-Patches getestet werden sollen.
        # BOOKMARK: Macht fixe DWI-Anzahl in Validation Sinn, oder soll es da auch flexibel gehalten werden?

        self.frozen_val_specs = []
        
        for i in range(self.num_patches):
            # Zufällig einen der Validation-Probanden wählen
            subj_idx = rng.randint(0, len(self.subjects_cache))
            subj = self.subjects_cache[subj_idx]
            
            x_dim, y_dim, z_dim, _ = subj['dwi'].shape
            valid_coords = np.where(subj['mask'] > 0)
            num_valid = len(valid_coords[0])
            
            while True:
                c_idx = rng.randint(0, num_valid)
                cx, cy, cz = valid_coords[0][c_idx], valid_coords[1][c_idx], valid_coords[2][c_idx]
                if (cx - half_patch >= 0 and cx + half_patch <= x_dim and
                    cy - half_patch >= 0 and cy + half_patch <= y_dim and
                    cz - half_patch >= 0 and cz + half_patch <= z_dim):
                    break
            
            n_dwi = protocol_dwi_counts[i % len(protocol_dwi_counts)]
            n_b0 = max(1, n_dwi // 6)
            
            sample_dict = subj['sampler'].sample(
                num_targets=self.num_targets,
                plot=False,
                rng=rng,
                n_dwi=n_dwi
            )
            
            self.frozen_val_specs.append({
                'subj_idx': subj_idx,
                'center': (cx, cy, cz),
                'sample': sample_dict
            })

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        if self.is_validation:
            spec = self.frozen_val_specs[idx]
            subj = self.subjects_cache[spec['subj_idx']]
            s = spec['sample']
            fixed_center = spec['center']
            apply_rotation = False
        else:
            subj_idx = np.random.choice(len(self.subjects_cache))
            subj = self.subjects_cache[subj_idx]
            s = subj['sampler'].sample(num_targets=self.num_targets, plot=False)
            fixed_center = None
            apply_rotation = True  # Augmentation Switch (verlustfreie 90°-Rotationen)
        
        # 1. Daten extrahieren (inklusive exakt zugeschnittenem hr_mask-Patch)
        hr_patch_b0, hr_patch_dwi, hr_target, hr_mask = extract_random_patch(
            subj['b0'], subj['dwi'], subj['mask'], 
            s['b0_indices'], s['input_indices'], s['withheld_target_indices'], 
            self.patch_size, fixed_center=fixed_center
        )
        
        # 2. Eine einzige 90°-Rotation (aus den 24 Würfel-Symmetrien) ziehen
        if apply_rotation:
            rot_spec = sample_random_90deg_rotation()
        else:
            rot_spec = None

        # 3. Augmentation & Degradation (im Validation-Modus noise_std=0 oder fest lassen)
        dummy_bvecs = np.zeros((hr_patch_b0.shape[-1], 3))
        lr_patch_b0, _, _ = generate_and_augment_lr_patch(
            hr_patch_b0, hr_mask, dummy_bvecs, 
            rot_spec=rot_spec, downscale_factor=self.downscale_factor
        )
        
        lr_patch_dwi, hr_mask_rotated, augmented_input_bvecs = generate_and_augment_lr_patch(
            hr_patch_dwi, hr_mask, s['input_bvecs'], 
            rot_spec=rot_spec, downscale_factor=self.downscale_factor
        )
        
        # 4. Target synchron und VERLUSTFREI mitrotieren (kein affine_transform Blur mehr!)
        if rot_spec is not None:
            perm, signs, rot_matrix = rot_spec
            hr_target = apply_3d_90deg_rotation(hr_target, perm, signs)
        else:
            rot_matrix = None
            hr_mask_rotated = hr_mask

        # 5. Dimensionsfolge für PyTorch anpassen: [X, Y, Z, N] -> [N, X, Y, Z]
        hr_target = np.transpose(hr_target, (3, 0, 1, 2))
        lr_patch_b0 = np.transpose(lr_patch_b0, (3, 0, 1, 2))
        lr_patch_dwi = np.transpose(lr_patch_dwi, (3, 0, 1, 2))
        
        # 6. Metadaten (FiLM) mit rotierten Gradienten (bvecs) berechnen
        n_dwi_inputs = lr_patch_dwi.shape[0]
        n_b0_inputs = lr_patch_b0.shape[0]
        
        meta_ij_list = []
        for j in range(self.num_targets):
            target_bval = s['target_bvals'][j] / 3000.0
            
            # Target-bvec ebenfalls mitrotieren, falls gedreht wurde!
            t_bvec = s['target_bvecs'][j]
            if rot_matrix is not None:
                t_bvec = np.dot(t_bvec, rot_matrix.T)
            target_ggT = compute_ggT(t_bvec)
            
            for i in range(n_dwi_inputs):
                input_bval = s['input_bvals'][i] / 3000.0
                input_ggT = compute_ggT(augmented_input_bvecs[i])
                
                meta_ij = np.concatenate([[input_bval], input_ggT, [target_bval], target_ggT])
                meta_ij_list.append(meta_ij)
                
        meta_ij_array = np.array(meta_ij_list, dtype=np.float32)
        if self.num_targets == 1:
            meta_ij_array = meta_ij_array.reshape(n_dwi_inputs, 14)

        # 7. Padding auf max_measurements
        D, H, W = lr_patch_dwi.shape[1], lr_patch_dwi.shape[2], lr_patch_dwi.shape[3]
        padded_lr_dwi = np.zeros((self.max_measurements, D, H, W), dtype=np.float32)
        padded_meta_ij = np.zeros((self.max_measurements, 14), dtype=np.float32)
        valid_mask = np.zeros(self.max_measurements, dtype=bool)
        
        padded_lr_dwi[:n_dwi_inputs] = lr_patch_dwi
        padded_meta_ij[:n_dwi_inputs] = meta_ij_array
        valid_mask[:n_dwi_inputs] = True

        padded_lr_b0 = np.zeros((self.max_b0, D, H, W), dtype=np.float32)
        padded_lr_b0[:n_b0_inputs] = lr_patch_b0

        valid_mask_b0 = np.zeros(self.max_b0, dtype=bool)
        valid_mask_b0[:n_b0_inputs] = True

        # 8. Rückgabe als PyTorch-Tensoren
        hr_target_tensor = torch.from_numpy(hr_target).float()
        lr_b0_tensor = torch.from_numpy(padded_lr_b0).float().unsqueeze(1)
        padded_lr_dwi_tensor = torch.from_numpy(padded_lr_dwi).float().unsqueeze(1)
        padded_meta_tensor = torch.from_numpy(padded_meta_ij).float()
        valid_mask_tensor = torch.from_numpy(valid_mask)
        hr_mask_tensor = torch.from_numpy(hr_mask_rotated.astype(np.float32))
        valid_mask_b0_tensor = torch.from_numpy(valid_mask_b0)

        return (
            lr_b0_tensor, 
            padded_lr_dwi_tensor, 
            padded_meta_tensor, 
            valid_mask_tensor, 
            valid_mask_b0_tensor, 
            hr_target_tensor, 
            hr_mask_tensor
        )