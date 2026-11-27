import numpy as np
import plotly.graph_objects as go
def _electrostatic_subset_selection(pool_indices, n_select, rng=None, refine_steps=3):
    """
    Wählt n_select Richtungen aus dem Pool für maximale sphärische Gleichverteilung:
    1. Initialisierung via antipodalem Farthest-Point-Sampling (verhindert enge Cluster).
    2. Nachoptimierung (Swapping) über das exakte elektrostatische Jones-Potenzial.
    """
    if rng is None:
        rng = np.random

    pool_bvecs = dwi_bvecs[pool_indices]
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

def plot_bvecs_on_sphere(bvecs, n):
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
    
    html_file = f"Figures/interactive_bvecs ({n} Richtungen).html"
    fig.write_html(html_file)
    print(f"Interaktive Kugel gespeichert unter: {html_file}")

n_dwi = 30
min_dwi_b = 900
max_dwi_b = 1100
bvec_p = "/mnt/nas_io/anarchy/lohrd/MRI_data/HCP/Diffusion/3T/100610/T1w/Diffusion/bvecs"
bval_p = "/mnt/nas_io/anarchy/lohrd/MRI_data/HCP/Diffusion/3T/100610/T1w/Diffusion/bvals"
bvals = np.loadtxt(bval_p).flatten()
bvecs = np.loadtxt(bvec_p).T
all_bvals = np.array(bvals)
all_bvecs = np.array(bvecs)

# Indizes für b0 und Ziel-Schale
global_dwi_indices = np.where((all_bvals >= min_dwi_b) & (all_bvals <= max_dwi_b))[0]

dwi_bvals = all_bvals[global_dwi_indices]
dwi_bvecs = all_bvecs[global_dwi_indices]

num_total_dwi = len(dwi_bvals)
all_local_dwi = np.arange(num_total_dwi)

input_indices = _electrostatic_subset_selection(all_local_dwi, n_dwi)
plot_bvecs_on_sphere(dwi_bvecs[input_indices], n_dwi)