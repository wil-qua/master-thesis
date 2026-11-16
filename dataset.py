# dataset.py
"""
Modul zur Datenvorbereitung und PyTorch-Dataset-Generierung.
Umfasst Sampling von Diffusions-MRT-Daten, Signalnormalisierung 
sowie Extraktion von High-Resolution (HR) und Low-Resolution (LR) Patches.
"""

import numpy as np
import nibabel as nib
import torch
import os
import re
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import gc
from scipy.ndimage import gaussian_filter, distance_transform_edt
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

    def sample(self, num_targets=1, plot=False, rng=None, n_dwi=None, p_b0_target=0.1):
        """
        Zieht DWI-Bilder (optimiert mit electrostatic repulsion) und b0-Bilder (zufällig).
        Als Targets werden danach mit 10% b0-Bilder gezogen und sonst DWI-Bilder.
        """
        if rng is None:
            rng = np.random

        if n_dwi is None:
            n_dwi = rng.randint(6, 31)
        max_b0 = max(1, n_dwi // 6)
        n_b0 = rng.randint(1, min(max_b0, self.num_total_b0) + 1) if max_b0 > 1 else 1
            
        all_local_dwi = np.arange(self.num_total_dwi)
        all_local_b0 = np.arange(self.num_total_b0)
        
        # 1. Inputs ziehen
        input_indices = self._electrostatic_subset_selection(all_local_dwi, n_dwi, rng=rng)
        selected_b0 = rng.choice(all_local_b0, size=n_b0, replace=False)

        # 2. Rest-Pools für Targets ermitteln (Disjunkte Trennung)
        remaining_dwi = np.setdiff1d(all_local_dwi, input_indices, assume_unique=True)
        remaining_b0 = np.setdiff1d(all_local_b0, selected_b0, assume_unique=True)
        
        target_vols_info = []
        target_bvals = []
        target_bvecs = []
        
        # 3. Targets ziehen
        for _ in range(num_targets):
            # 10% Wahrscheinlichkeit für b0 (sofern noch b0s verfügbar sind)
            if len(remaining_b0) > 0 and rng.uniform() < p_b0_target:
                idx = rng.choice(remaining_b0)
                target_vols_info.append(('b0', idx))
                remaining_b0 = np.setdiff1d(remaining_b0, [idx], assume_unique=True)
                target_bvals.append(self.b0_bvals[idx])
                target_bvecs.append(np.array([0.0, 0.0, 0.0], dtype=np.float32)) # Dummy-bvec
            else:
                idx = rng.choice(remaining_dwi)
                target_vols_info.append(('dwi', idx))
                remaining_dwi = np.setdiff1d(remaining_dwi, [idx], assume_unique=True)
                target_bvals.append(self.dwi_bvals[idx])
                target_bvecs.append(self.dwi_bvecs[idx])
        
        if plot:
            self.plot_bvecs_on_sphere(self.dwi_bvecs[input_indices])
        
        return {
            'b0_indices': selected_b0,
            'input_indices': input_indices,
            'target_vols_info': target_vols_info,
            'n_dwi': n_dwi,
            'n_b0': n_b0,
            'b0_bvals': self.b0_bvals[selected_b0],
            'input_bvals': self.dwi_bvals[input_indices],
            'input_bvecs': self.dwi_bvecs[input_indices],
            'target_bvals': np.array(target_bvals, dtype=np.float32),
            'target_bvecs': np.array(target_bvecs, dtype=np.float32)
        }

def build_6_zone_sampling_pools(b0_4d, dwi_4d, brain_mask, patch_size=64, return_masks=False):
    """
    Teilt alle gültigen Patch-Zentren innerhalb der Gehirnmaske rein datengetrieben
    (über Geometrie, DWI-Dämpfung und Winkel-Varianz) in 6 anatomisch/physikalisch
    wichtige Zonen ein.
    """
    half_p = patch_size // 2
    mask_bool = brain_mask > 0

    valid_bbox = np.zeros_like(mask_bool, dtype=bool)
    valid_bbox[half_p:-half_p, half_p:-half_p, half_p:-half_p] = True
    eligible = mask_bool & valid_bbox
    if not np.any(eligible):
        eligible = mask_bool

    # 1. Sparsame Kennwerte (float32, nur Teilmenge der Richtungen für std -> spart massiv RAM)
    b0_mean = np.mean(b0_4d, axis=-1, dtype=np.float32) if b0_4d.ndim == 4 else b0_4d.astype(np.float32)
    b0_safe = np.maximum(b0_mean, 1e-4)

    attenuation = np.mean(dwi_4d, axis=-1, dtype=np.float32) / b0_safe
    # n_sub = min(40, dwi_4d.shape[-1])
    # angular_var = np.std(dwi_4d[..., :n_sub], axis=-1, dtype=np.float32) / b0_safe
    angular_var = np.std(dwi_4d, axis=-1, dtype=np.float32) / b0_safe
    del b0_mean, b0_safe

    # 2. Geometrie ohne riesiges np.indices-Array
    dist_to_edge = distance_transform_edt(mask_bool).astype(np.float32)
    z_1d = np.arange(mask_bool.shape[2], dtype=np.int16)[None, None, :]
    z_25 = np.percentile(np.where(mask_bool)[2], 25)

    # Schwellenwerte innerhalb der gültigen Gehirnvoxel bestimmen
    att_wm_thresh = np.percentile(attenuation[eligible], 55)     # Obere 45% = Weiße Substanz (WM)
    att_csf_thresh = np.percentile(attenuation[eligible], 18)    # Untere 18% = CSF / Ventrikelgrenze
    # Innerhalb der weißen Substanz: Top 35% Winkel-Varianz = Single-Fiber, Rest = Crossing-Fibers
    wm_voxels = eligible & (attenuation >= att_wm_thresh)
    ang_single_thresh = np.percentile(angular_var[wm_voxels], 65)
    del wm_voxels

    # --- Einteilung in 6 disjunkte Zonen ---
    # Zone 3: Äußerer Kortex-Rand & U-Fibers (oberhalb Kleinhirn, nah an Oberfläche)
    z3_cortical = eligible & (dist_to_edge <= 12) & (z_1d >= z_25)

    # Zone 5: Hirnstamm & Kleinhirn (unterstes Z-Viertel, nicht reines CSF)
    z5_brainstem_cereb = eligible & (~z3_cortical) & (z_1d < z_25) & (attenuation >= att_csf_thresh)

    # Verbleibendes tiefes Großhirn
    deep_cerebrum = eligible & (~z3_cortical) & (~z5_brainstem_cereb)

    # Zone 6: Ventrikel & CSF-Grenzflächen (starker Signalabfall im DWI)
    z6_csf = deep_cerebrum & (attenuation < att_csf_thresh)

    # Zone 1: Single-Fiber High-FA (Corpus Callosum, Capsula Interna)
    z1_single_fiber = deep_cerebrum & (attenuation >= att_wm_thresh) & (angular_var >= ang_single_thresh)

    # Zone 2: Crossing-Fibers (Centrum Semiovale, Corona Radiata)
    z2_crossing_fiber = deep_cerebrum & (attenuation >= att_wm_thresh) & (angular_var < ang_single_thresh)

    # Zone 4: Subkortikale Kerne / Deep Gray Matter (Thalamus, Putamen, Pallidum)
    z4_subcortical = deep_cerebrum & (attenuation >= att_csf_thresh) & (attenuation < att_wm_thresh)

    del dist_to_edge, attenuation, angular_var, deep_cerebrum

    zone_masks = {
        "single_fiber_wm": z1_single_fiber,
        "crossing_fiber_wm": z2_crossing_fiber,
        "cortical_boundary": z3_cortical,
        "subcortical_gm": z4_subcortical,
        "brainstem_cerebellum": z5_brainstem_cereb,
        "csf_boundary": z6_csf,
    }
    
    # Koordinaten als platzsparendes int16 speichern!
    fallback_coords = np.argwhere(eligible).astype(np.int16)
    zone_pools = {}
    for name, z_mask in zone_masks.items():
        coords = np.argwhere(z_mask).astype(np.int16)
        zone_pools[name] = coords if len(coords) > 0 else fallback_coords

    gc.collect()
    return (zone_pools, zone_masks) if return_masks else (zone_pools, None)

def calculate_normalization_scalar(b0_volumes, brain_mask):
    """
    Berechnet den Skalierungsfaktor für die Normalisierung basierend auf dem
    95. Perzentil der b0-Signale innerhalb der Gehirnmaske.
    """
    # Falls ein 4D-Array [X, Y, Z, N_b0] übergeben wird, nimm nur das erste b0-Bild (Index 0)
    first_b0 = b0_volumes[..., 0] if b0_volumes.ndim == 4 else b0_volumes
    
    # Nur Voxel berücksichtigen, die in der Maske liegen und positive Signale haben
    valid_voxels = first_b0[(brain_mask > 0) & (first_b0 > 0)]
    scalar = np.percentile(valid_voxels, 95)
    return float(scalar)

def load_multi_subjects(nifti_paths, bval_paths, bvec_paths, mask_paths, b0_threshold, min_b_val, max_b_val, patch_size=64, cache_dir="npy_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    subjects_cache = []
    
    for i, (n_path, bval_p, bvec_p, m_path) in enumerate(zip(nifti_paths, bval_paths, bvec_paths, mask_paths)):
        # Eindeutigen Namen aus dem Probanden-Ordner ableiten (z.B. 146937)
        match = re.search(r'/(\d{6})/', n_path)
        subj_id = match.group(1) if match else f"subj_{i}"

        b0_cache_file = os.path.join(cache_dir, f"{subj_id}_b0.npy")
        dwi_cache_file = os.path.join(cache_dir, f"{subj_id}_dwi.npy")
        pools_cache_file = os.path.join(cache_dir, f"{subj_id}_pools.npz")

        print(f"Lade Proband {i+1}/{len(nifti_paths)} ({subj_id}) ...")
        bvals = np.loadtxt(bval_p).flatten()
        bvecs = np.loadtxt(bvec_p).T
        if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
            bvecs = bvecs.T
        
        sampler = FlexibleDiffusionSampler(bvals, bvecs, b0_threshold, min_b_val, max_b_val)
        brain_mask = np.asanyarray(nib.load(m_path).dataobj).astype(np.uint8)

        # Nur beim allerersten Mal aus NIfTI entpacken, normalisieren und auf Disk cachen
        if not (os.path.exists(b0_cache_file) and os.path.exists(dwi_cache_file) and os.path.exists(pools_cache_file)):
            print(f"  -> Erstelle einmaligen Memmap-Cache für {subj_id} ...")
            img = nib.load(n_path)
            full_data_4d = np.asanyarray(img.dataobj, dtype=np.float32)
            
            b0_4d_array = full_data_4d[..., sampler.global_b0_indices]
            dwi_4d_array = full_data_4d[..., sampler.global_dwi_indices]
            del full_data_4d
            gc.collect()
            
            scalar = calculate_normalization_scalar(b0_4d_array, brain_mask)
            b0_norm = (b0_4d_array / scalar).astype(np.float32)
            dwi_norm = (dwi_4d_array / scalar).astype(np.float32)
            del b0_4d_array, dwi_4d_array
            gc.collect()

            zone_pools, _ = build_6_zone_sampling_pools(b0_norm, dwi_norm, brain_mask, patch_size=patch_size)
            
            np.save(b0_cache_file, b0_norm)
            np.save(dwi_cache_file, dwi_norm)
            np.savez(pools_cache_file, **zone_pools)
            del b0_norm, dwi_norm, zone_pools
            gc.collect()

        # RAM-sparend per Memory-Mapping einbinden (~0 MB RAM-Verbrauch!)
        b0_mmap = np.load(b0_cache_file, mmap_mode='r')
        dwi_mmap = np.load(dwi_cache_file, mmap_mode='r')
        with np.load(pools_cache_file) as data:
            zone_pools = {k: data[k] for k in data.files}

        subjects_cache.append({
            'b0': b0_mmap,
            'dwi': dwi_mmap,
            'mask': brain_mask,
            'sampler': sampler,
            'zone_pools': zone_pools
        })
        
    print(f"Alle {len(subjects_cache)} Probanden via Memory-Mapping eingebunden.")
    return subjects_cache

def extract_random_patch(
    hr_data_b0, hr_data_dwi, mask, b0_indices, input_indices, target_vols_info,
    patch_size=64, fixed_center=None, zone_pools=None, zone_name=None, rng=None
):
    """
    Zieht ein Patch-Zentrum stratifiziert aus einer der 6 anatomischen Zonen.
    Passt min_brain_fraction automatisch an, damit Rand-Patches (Kortex/Kleinhirn)
    nicht fälschlicherweise verworfen werden.
    """
    if rng is None:
        rng = np.random

    x_dim, y_dim, z_dim, _ = hr_data_dwi.shape
    half_patch = patch_size // 2

    if fixed_center is not None:
        cx, cy, cz = fixed_center
        hr_mask = mask[
            cx - half_patch : cx + half_patch, 
            cy - half_patch : cy + half_patch, 
            cz - half_patch : cz + half_patch
        ]
    else:
        # 1. Zone auswählen (falls nicht fest vorgegeben, gleichverteilt 1 aus 6 wählen)
        if zone_pools is not None:
            if zone_name is None:
                zone_names = list(zone_pools.keys())
                zone_name = zone_names[int(rng.randint(0, len(zone_names)))]
            coords_pool = zone_pools[zone_name]
        else:
            coords_pool = np.argwhere(mask > 0)

        # Am Kortex-Rand und Kleinhirn ragt ein 64^3-Würfel natürlich in den Hintergrund -> 0.40 erlauben
        if zone_name in ("cortical_boundary", "brainstem_cerebellum"):
            min_brain_fraction = 0.40
        else:
            min_brain_fraction = 0.75

        for attempt in range(50):
            idx = int(rng.randint(0, len(coords_pool)))
            cx, cy, cz = coords_pool[idx]

            # Sicherstellen, dass der Patch komplett im Bildvolumen liegt
            cx = int(np.clip(cx, half_patch, x_dim - half_patch))
            cy = int(np.clip(cy, half_patch, y_dim - half_patch))
            cz = int(np.clip(cz, half_patch, z_dim - half_patch))

            hr_mask = mask[
                cx - half_patch : cx + half_patch, 
                cy - half_patch : cy + half_patch, 
                cz - half_patch : cz + half_patch
            ]
            # Nach 50 Fehlversuchen akzeptieren wir den Patch trotzdem, um Endlosschleifen zu vermeiden
            if np.mean(hr_mask > 0) >= min_brain_fraction or attempt == 49:
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

    target_vols = []
    for vol_type, idx in target_vols_info:
        if vol_type == 'b0':
            vol = hr_data_b0[cx - half_patch : cx + half_patch,
                             cy - half_patch : cy + half_patch,
                             cz - half_patch : cz + half_patch, idx]
        else:
            vol = hr_data_dwi[cx - half_patch : cx + half_patch,
                              cy - half_patch : cy + half_patch,
                              cz - half_patch : cz + half_patch, idx]
        target_vols.append(vol[..., np.newaxis])
        
    hr_target = np.concatenate(target_vols, axis=-1)

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
    noise_std=0.0  # 0.01, jetzt erstmal ausgeschaltet, da HCP Daten bereits processed sind und noise bereits korrigiert wurde
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
        Legt für die Validierung feste Patches an, die exakt gleichmäßig (reihum i % 6)
        alle 6 Hirnzonen und die verschiedenen DWI-Input-Anzahlen abdecken.
        """
        rng = np.random.RandomState(42)
        half_patch = self.patch_size // 2
        protocol_dwi_counts = [6, 12, 18, 24, 30]
        zone_names = [
            "single_fiber_wm",
            "crossing_fiber_wm",
            "cortical_boundary",
            "subcortical_gm",
            "brainstem_cerebellum",
            "csf_boundary"
        ]

        self.frozen_val_specs = []
        
        for i in range(self.num_patches):
            subj_idx = rng.randint(0, len(self.subjects_cache))
            subj = self.subjects_cache[subj_idx]
            x_dim, y_dim, z_dim, _ = subj['dwi'].shape

            # Reihum eine der 6 Zonen wählen!
            zone_name = zone_names[i % len(zone_names)]
            coords_pool = subj['zone_pools'][zone_name]
            min_frac = 0.40 if zone_name in ("cortical_boundary", "brainstem_cerebellum") else 0.75

            for attempt in range(50):
                c_idx = int(rng.randint(0, len(coords_pool)))
                cx, cy, cz = coords_pool[c_idx]
                cx = int(np.clip(cx, half_patch, x_dim - half_patch))
                cy = int(np.clip(cy, half_patch, y_dim - half_patch))
                cz = int(np.clip(cz, half_patch, z_dim - half_patch))

                hr_m = subj['mask'][cx-half_patch:cx+half_patch, cy-half_patch:cy+half_patch, cz-half_patch:cz+half_patch]
                if np.mean(hr_m > 0) >= min_frac or attempt == 49:
                    break
            
            n_dwi = protocol_dwi_counts[i % len(protocol_dwi_counts)]
            sample_dict = subj['sampler'].sample(
                num_targets=self.num_targets,
                plot=False,
                rng=rng,
                n_dwi=n_dwi
            )
            
            self.frozen_val_specs.append({
                'subj_idx': subj_idx,
                'center': (cx, cy, cz),
                'zone_name': zone_name,
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
            protocol_dwi_counts = [6, 12, 18, 24, 30]
            n_dwi = int(np.random.choice(protocol_dwi_counts))

            s = subj['sampler'].sample(
                num_targets=self.num_targets,
                plot=False,
                n_dwi=n_dwi
            )
            fixed_center = None
            apply_rotation = False  # Augmentation Switch (verlustfreie 90°-Rotationen, erstmal ausgeschaltet, weil 90° Rotationen in der Praxis nicht vorkommen)
        
        # 1. Daten extrahieren (inklusive exakt zugeschnittenem hr_mask-Patch)
        hr_patch_b0, hr_patch_dwi, hr_target, hr_mask = extract_random_patch(
            subj['b0'], subj['dwi'], subj['mask'], 
            s['b0_indices'], s['input_indices'], s['target_vols_info'], 
            self.patch_size, fixed_center=fixed_center, zone_pools=subj.get('zone_pools')
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