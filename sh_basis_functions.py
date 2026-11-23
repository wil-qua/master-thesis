import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lpmv


def real_symmetric_sh(l, m, theta, phi):
    """
    Berechnet die modifizierte reelle, symmetrische SH-Basisfunktion Y_j(theta, phi)
    für gerade Ordnungen l = 0, 2, 4, ... nach Descoteaux et al. (2007), Gl. [2] & [3].
    """
    abs_m = abs(m)
    norm = math.sqrt(
        ((2 * l + 1) / (4 * np.pi))
        * (math.factorial(l - abs_m) / math.factorial(l + abs_m))
    )
    p_lm = lpmv(abs_m, l, np.cos(theta))

    if m < 0:
        return math.sqrt(2) * norm * p_lm * np.cos(abs_m * phi)
    elif m == 0:
        return norm * p_lm
    else:
        return math.sqrt(2) * norm * p_lm * np.sin(abs_m * phi)


# 1. Kugelgitter erzeugen
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2 * np.pi, 150)
THETA, PHI = np.meshgrid(theta, phi)

X_unit = np.sin(THETA) * np.cos(PHI)
Y_unit = np.sin(THETA) * np.sin(PHI)
Z_unit = np.cos(THETA)

# 2. Plot mit minimalem Weißraum erstellen
orders = [0, 2, 4]
fig = plt.figure(figsize=(16, 6.2))
# fig.suptitle(
#     "Reelle, symmetrische Spherical Harmonics Basisfunktionen (l = 0, 2, 4)\n"
#     "Radius = |Y_j(θ, φ)|, Farbe = Vorzeichen (Rot: +, Blau: -)",
#     fontsize=13,
#     y=0.97,
# )

j_idx = 1
for row_idx, l in enumerate(orders):
    ms = range(-l, l + 1)
    for col_offset, m in enumerate(ms):
        col_idx = (9 - len(ms)) // 2 + col_offset + 1
        ax = fig.add_subplot(len(orders), 9, row_idx * 9 + col_idx, projection="3d")

        Y = real_symmetric_sh(l, m, THETA, PHI)
        #R = np.abs(Y)
        R = Y
        X = R * X_unit
        Y_coord = R * Y_unit
        Z = R * Z_unit

        colors = plt.cm.coolwarm((np.sign(Y) + 1) / 2)
        ax.plot_surface(
            X, Y_coord, Z, facecolors=colors, rstride=2, cstride=2,
            linewidth=0, antialiased=True, shade=True
        )

        # Dynamisches Limit pro Funktion: füllt jedes Teilbild maximal ohne Abschneiden
        limit = np.max(np.abs([X, Y_coord, Z])) * 1.02
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
        ax.set_box_aspect([1, 1, 1])
        ax.axis("off")
        ax.set_title(f"j={j_idx} (l={l}, m={m})", fontsize=8.5, pad=2)
        j_idx += 1

# Unsichtbare 3D-Ränder zusammenschieben
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.87, wspace=-0.22, hspace=0.08)

plt.savefig("sh_basis_functions.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.close(fig)
print("Bild erfolgreich als 'sh_basis_functions.png' gespeichert.")