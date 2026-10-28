# sh_reference.py
"""
Nutzt LR-Patches als Input und löst räumliche Skalierung (über trilineare Interpolation)
und Winkel-Projektion (Spherical Harmonics) mathematisch.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import sph_harm_y

def cart2sph(bvecs):
    """Konvertiert kartesische Vektoren in sphärische Koordinaten (theta, phi)."""
    # bvecs shape: (N, 3)
    x, y, z = bvecs[:, 0], bvecs[:, 1], bvecs[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    # Vermeide Division durch Null
    r = np.clip(r, a_min=1e-8, a_max=None)
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))  # Polarwinkel [0, pi]
    phi = np.arctan2(y, x)                        # Azimutwinkel [-pi, pi]
    return theta, phi

def real_spherical_harmonics(lmax, theta, phi):
    """
    Berechnet die reelle, gerade SH-Basis für gegebene Winkel.
    Nutzt ausschließlich gerade l (0, 2, 4...) für die g/-g Symmetrie.
    """
    N = len(theta)
    num_coeffs = int((lmax + 1) * (lmax + 2) / 2)
    B = np.zeros((N, num_coeffs), dtype=np.float32)
    
    idx = 0
    for l in range(0, lmax + 1, 2):
        for m in range(-l, l + 1):
            # l = Grad, m = Ordnung
            Y = sph_harm_y(l, m, theta, phi)

            # Wandelt die komplexen sphärischen Harmonischen in reelle sphärische Harmonische um.
            if m < 0:
                B[:, idx] = np.sqrt(2) * Y.imag
            elif m == 0:
                B[:, idx] = Y.real
            else:
                B[:, idx] = np.sqrt(2) * Y.real
            idx += 1
            
    return B

class RegularizedSHBaseline:
    """
    Interpolation + Regularized SH Baseline.
    Führt zuerst ein räumliches Upsampling der LR-Daten durch und fittet dann
    voxelweise ein regulariertes SH-Modell.
    """
    def __init__(self, lmax=4, lambda_reg=0.005, spatial_scale=2.0):
        self.lmax = lmax
        self.lambda_reg = lambda_reg
        self.spatial_scale = spatial_scale

    def fit_predict(self, lr_dwi, input_bvecs, target_bvecs):
        """
        lr_dwi: PyTorch Tensor [C, D, H, W] (C = Anzahl der Input-Richtungen)
        input_bvecs: Numpy Array [C, 3]
        target_bvecs: Numpy Array [T, 3]
        """
        device = lr_dwi.device
        C, D, H, W = lr_dwi.shape
        
        # 1. Spatial Interpolation (Trilinear) für das HR Gitter
        lr_dwi_batch = lr_dwi.unsqueeze(0)  # [1, C, D, H, W]
        hr_dwi = F.interpolate(
            lr_dwi_batch, 
            scale_factor=self.spatial_scale, 
            mode='trilinear', 
            align_corners=False
        ).squeeze(0)  # [C, D_hr, H_hr, W_hr]
        
        # Flachklopfen für Voxel-weisen Fit: [C, D, H, W] -> [C, V]
        hr_dwi_flat = hr_dwi.view(C, -1).cpu().numpy()
        
        # 2. SH Basis für Input berechnen
        theta_in, phi_in = cart2sph(input_bvecs)
        B_in = real_spherical_harmonics(self.lmax, theta_in, phi_in)  # [C, Num_Coeffs]
        
        # 3. Regularisierte Matrix-Inversion: c = (B^T B + lambda*I)^-1 B^T S
        # Nötig, aufgrund der wenigen Messungen -> B^T B lässt sich ohne Regularisierung vllt nicht invertieren
        num_coeffs = B_in.shape[1]
        L_matrix = np.eye(num_coeffs) * self.lambda_reg
        L_matrix[0, 0] = 0.0  # Den B0/Mean-Term nicht regularisieren, weil sonst das MRT-Singal künstlich abgedunklet wird.
        
        pseudo_inv = np.linalg.pinv(B_in.T @ B_in + L_matrix) @ B_in.T  # [Num_Coeffs, C], (B^T B + lambda*I)^-1 B^T
        sh_coeffs = pseudo_inv @ hr_dwi_flat  # [Num_Coeffs, V], wendet Lösungsmatrix auf Bilddaten an 
        
        # 4. SH Basis für Targets berechnen und Signal rekonstruieren
        theta_tgt, phi_tgt = cart2sph(target_bvecs)
        B_tgt = real_spherical_harmonics(self.lmax, theta_tgt, phi_tgt)  # [T, Num_Coeffs]
        
        target_pred_flat = B_tgt @ sh_coeffs  # [T, V]
        
        # 5. Zurück in 3D-Shape transformieren
        T = target_bvecs.shape[0]
        target_pred = target_pred_flat.reshape(T, hr_dwi.shape[1], hr_dwi.shape[2], hr_dwi.shape[3])
        
        return torch.from_numpy(target_pred).float().to(device)