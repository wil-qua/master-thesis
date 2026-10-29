import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import os

# Import der eigenen Module
import config
from dataset import load_multi_subjects, HCPPatchDataset
from utils import plot_training_loss

# ==========================================
# 1. BAUSTEINE FÜR MODEL A 
# ==========================================

class DegradationLayer(nn.Module):
    def __init__(self, sigma=0.8, downscale_factor=2):
        super().__init__()
        self.downscale_factor = downscale_factor
        
        # Erstelle einen 3D-Gauss-Kernel passend zu scipy.ndimage.gaussian_filter
        kernel_size = int(2 * np.ceil(2 * sigma) + 1) # meist 5x5x5 bei sigma=0.8
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        
        # 3D-Kernel bauen: [1, 1, K, K, K]
        kernel_3d = g[:, None, None] * g[None, :, None] * g[None, None, :]
        kernel_3d = kernel_3d.unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", kernel_3d)
        self.padding = kernel_size // 2

    def forward(self, hr_batch):
        # hr_batch: [B, 1, D, H, W]
        blurred = F.conv3d(hr_batch, self.kernel, padding=self.padding)
        # Subsampling durch Slicing
        lr_batch = blurred[:, :, ::self.downscale_factor, ::self.downscale_factor, ::self.downscale_factor]
        return lr_batch

class ResBlock3D(nn.Module):
    """Residual Block mit 3x3x3 Conv, SiLU, ohne Normalisierung."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.silu = nn.SiLU()
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.silu(self.conv1(x)))

class MetadataMLP(nn.Module):
    """
    2-Layer MLP (Hidden 64) für die Modulation (FiLM).
    beta und gamma skalieren und verschieben die Bildmerkmale physikalisch ("drehen" des Bildes auf das Target). 
    """
    def __init__(self, meta_dim, feature_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.SiLU(),
            nn.Linear(64, feature_channels * 2) 
        )

    def forward(self, metadata):
        out = self.mlp(metadata)
        gamma, beta = out.chunk(2, dim=-1)
        return gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), \
               beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


# ==========================================
# 2. MODEL A ARCHITEKTUR
# ==========================================

class ModelA_FlexibleSR(nn.Module): # schaue nach, was best practice / standard ist, benutze auch vortrainierten encoder
    def __init__(self, meta_dim=14):
        super().__init__()
        
        # --- Encoder ---
        self.init_conv = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.enc1 = nn.Sequential(ResBlock3D(16), ResBlock3D(16))
        
        self.down1 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.enc2 = nn.Sequential(ResBlock3D(32), ResBlock3D(32))
        
        self.down2 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc3 = nn.Sequential(ResBlock3D(64), ResBlock3D(64))
        
        # --- Metadata Modulation (FiLM) ---
        self.mod_conv1x1 = nn.Conv3d(64, 64, kernel_size=1)
        self.meta_mlp = MetadataMLP(meta_dim, feature_channels=64)
        self.mod_silu = nn.SiLU()

        # Bottleneck-Fusion für u_j_avg (64) + b0_feat_3 (64) = 128 -> 64
        self.bottleneck_fuse = nn.Conv3d(128, 64, kernel_size=1)

        # --- Decoder ---
        # 64 (von unten) + 32 (b0 Skala 2) + 32 (DWI Skip Skala 2) = 128
        self.dec2_conv = nn.Conv3d(128, 32, kernel_size=3, padding=1)
        self.dec2 = nn.Sequential(ResBlock3D(32), ResBlock3D(32))
        
        # 32 (von unten) + 16 (b0 Skala 1) + 16 (DWI Skip Skala 1) = 64
        self.dec1_conv = nn.Conv3d(64, 16, kernel_size=3, padding=1)
        self.dec1 = nn.Sequential(ResBlock3D(16), ResBlock3D(16))

        # --- Upsampling Head, um von LR zu HR zu kommen ---
        self.head_conv = nn.Sequential(
            nn.Conv3d(16, 8, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(8, 1, kernel_size=3, padding=1)
        )

    def upsample_trilinear(self, x):
        return F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)

    def masked_average(self, features, valid_mask):
        """
        features:   [B, N, C, D, H, W]
        valid_mask: [B, N] (True für echte Bilder, False für Padding)
        Berechnet Durchschnitt der Features und ignoriert Padding (Nullen).
        Nötig, um Signalstärke im Netzwerk konstant zu halten (bei versch. Anzahl an Richtungen).
        Fasse beliebig viele Input-Richtungen zu einem einzigen Feature-Tensor für den Decoder. 
        Masked_average addiert helles und dunkles Signal und teilt durch 2 -> U-Net erhält nur noch ein diffuses, ungerichtetes Grau.
        Das Netzwerk "vergisst", aus welcher Richtung das Signal ursprünglich kam, und erzeugt künstlich ein isotropes Signal ohne starke Richtungsabhängigkeit.
        """
        # Forme Maske um für Broadcasting: [B, N, 1, 1, 1, 1]
        mask_expanded = valid_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()
        
        # Nulle alle Padding-Features aus
        masked_features = features * mask_expanded
        
        # Summiere über die N-Dimension (die Measurement-Achse)
        feature_sum = masked_features.sum(dim=1) # [B, C, D, H, W]
        
        # Zähle die echten Bilder pro Batch: [B, 1, 1, 1, 1]
        valid_counts = valid_mask.sum(dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()
        
        # Vermeide Division durch 0
        valid_counts = torch.clamp(valid_counts, min=1.0)
        
        return feature_sum / valid_counts

    def forward(self, x_dwi, x_b0, meta_ij, valid_mask_dwi, valid_mask_b0):
        """
        x_dwi:      [B, N_max=35, 1, D, H, W]
        x_b0:       [B, 1, 1, D, H, W]
        meta_ij:    [B, N_max=35, 14] 
        valid_mask: [B, N_max=35]
        """
        B, N_max, _, D, H, W = x_dwi.shape
        B_b0, N_b0 = x_b0.shape[0], x_b0.shape[1]
        
        # --- 1. b0 Encoding ---
        x_b0_flat = x_b0.view(B_b0 * N_b0, 1, D, H, W)
        b0_e1 = self.enc1(self.init_conv(x_b0_flat))
        b0_e2 = self.enc2(self.down1(b0_e1))
        b0_e3 = self.enc3(self.down2(b0_e2))
        
        # Für b0
        b0_feat_1 = self.masked_average(b0_e1.view(B_b0, N_b0, 16, D, H, W), valid_mask_b0)
        b0_feat_2 = self.masked_average(b0_e2.view(B_b0, N_b0, 32, D//2, H//2, W//2), valid_mask_b0)
        b0_feat_3 = self.masked_average(b0_e3.view(B_b0, N_b0, 64, D//4, H//4, W//4), valid_mask_b0)

        # --- 2. DWI Encoding ---
        x_dwi_flat = x_dwi.view(B * N_max, 1, D, H, W)
        e1 = self.enc1(self.init_conv(x_dwi_flat))
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2)) 

        # --- 3. Modulation (Der FiLM-Schritt) ---
        meta_flat = meta_ij.view(B * N_max, -1) # [B*35, 14]
        gamma, beta = self.meta_mlp(meta_flat) 
        
        F_i_conv = self.mod_conv1x1(e3)
        u_ij = self.mod_silu((1 + gamma) * F_i_conv + beta) 
        
        # Masked Averaging
        u_ij_reshaped = u_ij.view(B, N_max, 64, D//4, H//4, W//4)
        u_j_avg = self.masked_average(u_ij_reshaped, valid_mask_dwi)

        # Bottleneck b0-Fusion auf LR/4 Ebene
        u_fused = self.bottleneck_fuse(torch.cat([u_j_avg, b0_feat_3], dim=1))
        
        # Skip-Connections mit Masked Averaging mitteln
        skip1 = self.masked_average(e1.view(B, N_max, 16, D, H, W), valid_mask_dwi)
        skip2 = self.masked_average(e2.view(B, N_max, 32, D//2, H//2, W//2), valid_mask_dwi)

        # --- 4. Decoder & b0-Fusion ---
        d2 = self.upsample_trilinear(u_fused)
        d2 = torch.cat([d2, b0_feat_2, skip2], dim=1) 
        d2 = self.dec2(self.dec2_conv(d2))
        
        d1 = self.upsample_trilinear(d2)
        d1 = torch.cat([d1, b0_feat_1, skip1], dim=1)
        d1 = self.dec1(self.dec1_conv(d1))

        # --- 5. Upsampling Head ---
        out = self.upsample_trilinear(d1)
        out = self.head_conv(out)
        
        return out


class Masked3DSSIMAndL1Loss(nn.Module):
    """
    Berechnet einen maskierten 3D-SSIM Loss (optional kombiniert mit Masked L1 Loss).
    alpha = 1.0 -> Reiner 3D-SSIM Loss
    alpha = 0.0 -> Reiner L1 Loss (MAE)
    alpha = 0.5 -> 50% SSIM + 50% L1 (Guter Kompromiss für scharfe Kanten + quantitative Treue)
    SSIM: Messung der Ähnlichkeit zwischen zwei Bildern über Mittelwerte und Varianzen. Hohe Frequenzen (Kanten, Gewebeübergänge) bleiben. 
    Bei Standard-Packages wie piq oder pytorch-mssim scheint es Probleme mit 3D-Volumina oder hr_mask zu geben.
    """
    def __init__(self, window_size=7, sigma=1.5, data_range=1.0, alpha=0.5):
        super().__init__()
        self.window_size = window_size
        self.padding = window_size // 2
        self.C1 = (0.01 * data_range) ** 2
        self.C2 = (0.03 * data_range) ** 2
        self.alpha = alpha
        
        # 3D Gauß-Kernel
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel_3d = g[:, None, None] * g[None, :, None] * g[None, None, :]
        kernel_3d = kernel_3d.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, K, K, K]
        self.register_buffer("window", kernel_3d)  # fester Filter, der automatisch auf die GPU geschoben wird

    def forward(self, pred, target, mask):
        """
        pred:   [B, C, D, H, W]
        target: [B, C, D, H, W]
        mask:   [B, 1, D, H, W]
        """
        # Für numerische Stabilität bei Mixed Precision (AMP) in float32 rechnen
        pred_f = pred.float()
        target_f = target.float()
        mask_f = mask.float()
        
        C = pred_f.shape[1]
        window = self.window.expand(C, 1, -1, -1, -1).to(pred_f.dtype)

        # Lokale Mittelwerte (Luminanz)
        mu1 = F.conv3d(pred_f, window, padding=self.padding, groups=C)
        mu2 = F.conv3d(target_f, window, padding=self.padding, groups=C)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # Lokale Varianzen und Kovarianz (Kontrast & Struktur)
        # clamp(min=0) verhindert negative Varianzen durch numerische Ungenauigkeiten
        sigma1_sq = torch.clamp(
            F.conv3d(pred_f * pred_f, window, padding=self.padding, groups=C) - mu1_sq, min=0.0
        )
        sigma2_sq = torch.clamp(
            F.conv3d(target_f * target_f, window, padding=self.padding, groups=C) - mu2_sq, min=0.0
        )
        sigma12 = F.conv3d(pred_f * target_f, window, padding=self.padding, groups=C) - mu1_mu2

        # Voxel-weise SSIM-Map berechnen
        ssim_num = (2.0 * mu1_mu2 + self.C1) * (2.0 * sigma12 + self.C2)
        ssim_den = (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        ssim_map = ssim_num / (ssim_den + 1e-8)

        # SSIM liegt bei [-1, 1] (1 = identisch). Als Loss wollen wir (1 - SSIM) minimieren:
        ssim_loss_map = 1.0 - ssim_map

        # Maskierung auf Gehirn-Voxel
        denom = mask_f.sum() * C + 1e-8
        masked_ssim_loss = (ssim_loss_map * mask_f).sum() / denom  # Zuerst 3D-SSIM-Map auf unmaskiertem Patch berechnen und dann erst Maske anwenden.

        if self.alpha >= 1.0:
            return masked_ssim_loss

        # Optionaler L1-Anteil
        l1_loss_map = torch.abs(pred_f - target_f)
        masked_l1_loss = (l1_loss_map * mask_f).sum() / denom

        return self.alpha * masked_ssim_loss + (1.0 - self.alpha) * masked_l1_loss

def validate(model, val_loader, criterion, device, use_amp=False):
    """Führt die Validation auf den Validierungs-Probanden durch."""
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch_data in val_loader:
            x_b0, x_dwi_padded, meta_padded, valid_mask_dwi, valid_mask_b0, hr_target, hr_mask = batch_data
            
            x_b0 = x_b0.to(device)
            x_dwi_padded = x_dwi_padded.to(device)
            meta_padded = meta_padded.to(device)
            valid_mask_dwi = valid_mask_dwi.to(device)
            valid_mask_b0 = valid_mask_b0.to(device)
            hr_target = hr_target.to(device)
            hr_mask = hr_mask.to(device)

            if hr_mask.ndim == 4:
                hr_mask = hr_mask.unsqueeze(1)
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    sr_out = model(x_dwi_padded, x_b0, meta_padded, valid_mask_dwi, valid_mask_b0)
                    loss = criterion(sr_out, hr_target, hr_mask)
            else:
                sr_out = model(x_dwi_padded, x_b0, meta_padded, valid_mask_dwi, valid_mask_b0)
                loss = criterion(sr_out, hr_target, hr_mask)
                
            val_loss += loss.item()
            
    return val_loss / len(val_loader)

# ==========================================
# 3. TRAINING LOOP
# ==========================================

def main():
    # Reproduzierbarkeit sicherstellen
    # np.random.seed(20260831)
    # torch.manual_seed(20260831)
    os.makedirs("Figures", exist_ok=True)
    os.makedirs("Models", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        total_mem = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        print(f"\nNutze Gerät: {device} ({gpu_name} | {total_mem:.1f} GB VRAM)")
    else:
        print(f"\nNutze Gerät: CPU")
        
    
    print("Lade Trainings-Probanden...")
    train_cache = load_multi_subjects(
        config.TRAIN_NIFTI_FILES, config.TRAIN_BVAL_FILES, 
        config.TRAIN_BVEC_FILES, config.TRAIN_MASK_FILES, 
        config.B0_THRESHOLD, config.MIN_DWI_B, config.MAX_DWI_B
    )
    train_cache[0]['sampler'].sample(plot=True)
    
    print("\nLade Validierungs-Probanden...")
    val_cache = load_multi_subjects(
        config.VAL_NIFTI_FILES, config.VAL_BVAL_FILES, 
        config.VAL_BVEC_FILES, config.VAL_MASK_FILES, 
        config.B0_THRESHOLD, config.MIN_DWI_B, config.MAX_DWI_B
    )
    
    # PyTorch Datasets initialisieren
    train_dataset = HCPPatchDataset(
        subjects_cache=train_cache,
        num_patches_per_epoch=config.NUM_PATCHES_PER_EPOCH, 
        patch_size=config.PATCH_SIZE,
        downscale_factor=config.DOWNSCALE_FACTOR,
        max_measurements=config.MAX_MEASUREMENTS,
        is_validation=False
    )
    
    val_dataset = HCPPatchDataset(
        subjects_cache=val_cache,
        num_patches_per_epoch=getattr(config, 'NUM_VAL_PATCHES', 25), 
        patch_size=config.PATCH_SIZE,
        downscale_factor=config.DOWNSCALE_FACTOR,
        max_measurements=config.MAX_MEASUREMENTS,
        is_validation=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, 
                            num_workers=0, pin_memory=True)
    
    model = ModelA_FlexibleSR(meta_dim=14).to(device)

    # Starte mit reinem L1 Loss (alpha=0.0)
    criterion = Masked3DSSIMAndL1Loss(window_size=7, data_range=1.0, alpha=0.0).to(device)

    # Adam + Decoupled Weight Decay (schützt vor Overfitting)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4) 

    # AMP schaltet bei großen convolutions von 32-bit auf 16-bit runter
    # scaler multipliziert kleine losses mit grißem Faktor, sodass auch kleine Gradienten in 16-bit Format passen
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    # 4. TRAININGSSCHLEIFE
    EPOCHS = config.NUM_EPOCHS
    
    print(f"\nStarte Training für {EPOCHS} Epochen...")
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    
    degradation = DegradationLayer(sigma=0.8, downscale_factor=config.DOWNSCALE_FACTOR).to(device)
    LAMBDA_DC = 0.5  # Gewichtung für den Data Consistency Loss

    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch_data in enumerate(train_loader):
            global_step += 1

            if global_step == 500:
                criterion.alpha = 0.5
                print(f"\n[Update {global_step}] Loss-Funktion umgestellt auf 50% SSIM + 50% L1 (alpha=0.5)\n")

            x_b0, x_dwi_padded, meta_padded, valid_mask_dwi, valid_mask_b0, hr_target, hr_mask = batch_data
            
            x_b0 = x_b0.to(device)
            x_dwi_padded = x_dwi_padded.to(device)
            meta_padded = meta_padded.to(device)
            valid_mask_dwi = valid_mask_dwi.to(device)
            valid_mask_b0 = valid_mask_b0.to(device)
            hr_target = hr_target.to(device)
            hr_mask = hr_mask.to(device)

            if hr_mask.ndim == 4:
                hr_mask = hr_mask.unsqueeze(1)

            optimizer.zero_grad()
            
            # --- Metadaten der INPUT-Richtungen für den DC-Term bauen ---
            # Für den DC-Term: Ziel = dem Input. 
            # Wir kopieren die Input-Metadaten (Indizes 0-6) auf die Target-Metadaten (Indizes 7-13).
            # meta_padded: Die ersten 7 Werte beschreiben die Eingaberichtung (Ist-Zustand), die letzten 7 beschreiben die unbekannte Zielrichtung (Soll-Zustand).
            # meta_input_self: Kopiert Tensor und ersetzt die Ziel-Werte durch die Eingabe-Werte.
            # Index 0 ist immer ein gültiger Input-Kanal (kein Padding)
            # Wir setzen die Input-Richtung 0 als gemeinsames Target für ALLE Kanäle:
            meta_input_self = meta_padded.clone()
            target_meta_0 = meta_padded[:, 0:1, 0:7].expand(-1, meta_padded.shape[1], -1)
            meta_input_self[:, :, 7:14] = target_meta_0

            # === Masken für b0 und DWI vor dem Loss berechnen ===
            # Target b-Wert liegt an Index 7. Wegen der Normalisierung in dataset.py 
            # müssen wir ihn für den Vergleich mit dem Threshold wieder mit 3000 multiplizieren.
            target_bvals = meta_padded[:, 0, 7] * 3000.0
            is_b0 = (target_bvals < config.B0_THRESHOLD)
            is_dwi = ~is_b0            

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    # 1. PASS: Vorhersage der withheld Targets
                    sr_out = model(x_dwi_padded, x_b0, meta_padded, valid_mask_dwi, valid_mask_b0)
                    
                    # === Separierter und skalierter SR-Loss ===
                    loss_sr = torch.tensor(0.0, device=device)
                    
                    if is_b0.any():
                        loss_b0 = criterion(sr_out[is_b0], hr_target[is_b0], hr_mask[is_b0])
                        loss_sr += config.WEIGHT_B0 * loss_b0
                    
                    if is_dwi.any():
                        loss_dwi = criterion(sr_out[is_dwi], hr_target[is_dwi], hr_mask[is_dwi])
                        loss_sr += config.WEIGHT_DWI * loss_dwi

                    # 2. PASS: Vorhersage der 0-ten Input-DWI in HR (für DC)
                    lr_to_hr = model(x_dwi_padded, x_b0, meta_input_self, valid_mask_dwi, valid_mask_b0)
                    pred_lr = degradation(lr_to_hr)
                    lr_target_0 = x_dwi_padded[:, 0]
                    
                    diff = torch.abs(pred_lr - lr_target_0)
                    mask_brain_lr = hr_mask[:, :, ::config.DOWNSCALE_FACTOR, ::config.DOWNSCALE_FACTOR, ::config.DOWNSCALE_FACTOR]
                    loss_dc = (diff * mask_brain_lr).sum() / (mask_brain_lr.sum() + 1e-8)

                    # Gesamt-Loss
                    loss = loss_sr + LAMBDA_DC * loss_dc
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Dasselbe nochmal ohne AMP
                sr_out = model(x_dwi_padded, x_b0, meta_padded, valid_mask_dwi, valid_mask_b0)
                
                loss_sr = torch.tensor(0.0, device=device)
                if is_b0.any():
                    loss_sr += config.WEIGHT_B0 * criterion(sr_out[is_b0], hr_target[is_b0], hr_mask[is_b0])
                if is_dwi.any():
                    loss_sr += config.WEIGHT_DWI * criterion(sr_out[is_dwi], hr_target[is_dwi], hr_mask[is_dwi])
                
                lr_to_hr = model(x_dwi_padded, x_b0, meta_input_self, valid_mask_dwi, valid_mask_b0)
                pred_lr = degradation(lr_to_hr)
                lr_target_0 = x_dwi_padded[:, 0]  # [B, 1, D_lr, H_lr, W_lr] (5D)
                
                diff = torch.abs(pred_lr - lr_target_0)
                mask_brain_lr = hr_mask[:, :, ::config.DOWNSCALE_FACTOR, ::config.DOWNSCALE_FACTOR, ::config.DOWNSCALE_FACTOR]
                loss_dc = (diff * mask_brain_lr).sum() / (mask_brain_lr.sum() + 1e-8)
                
                loss = loss_sr + LAMBDA_DC * loss_dc
                
                loss.backward()
                optimizer.step()
                
            epoch_loss += loss_sr.item() # Loggt den reinen SR-Loss passend zur Validation-Kurve
            
            # === Getrennte SR-Losses für Logging berechnen ===
            with torch.no_grad():
                if is_b0.any():
                    b0_loss_val = criterion(sr_out[is_b0], hr_target[is_b0], hr_mask[is_b0]).item()
                else:
                    b0_loss_val = 0.0
                    
                if is_dwi.any():
                    dwi_loss_val = criterion(sr_out[is_dwi], hr_target[is_dwi], hr_mask[is_dwi]).item()
                else:
                    dwi_loss_val = 0.0
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{EPOCHS}], Batch [{batch_idx}/{len(train_loader)}]")
                print(f"  Gesamt SR-Loss: {loss_sr.item():.4f}")
                
                if is_b0.any():
                    print(f"  -> b0 SR-Loss:  {b0_loss_val:.4f} (basierend auf {is_b0.sum().item()} Bildern)")
                if is_dwi.any():
                    print(f"  -> DWI SR-Loss: {dwi_loss_val:.4f} (basierend auf {is_dwi.sum().item()} Bildern)")
                    
                print(f"  -> DC-Loss:     {loss_dc.item():.4f}")
                
        avg_train_loss = epoch_loss / len(train_loader)
        
        # ---  VALIDATION ---
        avg_val_loss = validate(model, val_loader, criterion, device, use_amp=(scaler is not None))
        # bookmark: soll der validate loss auch den DC-Term enthalten, oder nur den "normalen" loss, weil das eig. Ziel SR ist?
        # DC-Loss als Hilfsmittel zum Training, oder als fester Loss-Term?
        
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        
        # Bestes Modell abspeichern, wenn Val-Loss einen Rekord aufstellt
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "Models/model_A_best.pth")
            best_marker = " [* Best Val *]"
        else:
            best_marker = ""
            
        print(f"==> Ende Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}{best_marker}\n")
        # Bestes Modell abspeichern, wenn Val-Loss einen Rekord aufstellt
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "Models/model_A_best.pth")
            best_marker = " [* Best Val *]"
        else:
            best_marker = ""
            
        print(f"==> Ende Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}{best_marker}")

        # === Checkpoint und Plot alle 20 Epochen speichern ===
        CHECKPOINT_FREQ = 20
        if epoch % CHECKPOINT_FREQ == 0:
            ckpt_path = f"Models/model_A_epoch_{epoch}.pth"
            plot_path = f"Figures/training_loss_model_A_epoch_{epoch}.png"
            
            # Modell-Gewichte für diese spezifische Epoche sichern
            torch.save(model.state_dict(), ckpt_path)
            
            # Trainingsverlauf bis hierhin plotten
            plot_training_loss(
                train_loss_history, 
                val_loss_history=val_loss_history, 
                config=config, 
                save_path=plot_path
            )
            print(f"  [Checkpoint] Status gesichert: {ckpt_path} und {plot_path}\n")
        else:
            print()

    # === MODELL SPEICHERN UND PLOTTEN (mit beiden Kurven) ===
    plot_training_loss(
        train_loss_history, 
        val_loss_history=val_loss_history, 
        config=config, 
        save_path="Figures/training_loss_model_A.png"
    )

    save_path = "Models/model_A.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nFinales Modell gespeichert unter: {save_path} (Bestes Val-Modell unter: model_A_best.pth)")

if __name__ == "__main__":
    main()