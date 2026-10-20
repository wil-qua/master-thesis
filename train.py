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

        # --- Upsampling Head ---
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
        Fasse beliebig viele Input-Richtungen zu einem einzigen Feature-Tensor für den Decoder. 
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
                    loss_map = criterion(sr_out, hr_target)
                    loss = (loss_map * hr_mask).sum() / (hr_mask.sum() * sr_out.shape[1] + 1e-8)
            else:
                sr_out = model(x_dwi_padded, x_b0, meta_padded, valid_mask_dwi, valid_mask_b0)
                loss_map = criterion(sr_out, hr_target)
                loss = (loss_map * hr_mask).sum() / (hr_mask.sum() * sr_out.shape[1] + 1e-8)
                
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
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        total_mem = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        print(f"\nNutze Gerät: {device} ({gpu_name} | {total_mem:.1f} GB VRAM)")
    else:
        print(f"\nNutze Gerät: CPU")
    
    model = ModelA_FlexibleSR(meta_dim=14).to(device)

    # L1: MAE
    # besser als MSE (L2), weil da große Abweichungen extrem gewichtet werden ->  verschwommene Kanten
    criterion = nn.L1Loss(reduction='none')  # jedes Voxel behält eigenen Fehlerwert

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
    
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch_data in enumerate(train_loader):
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
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    sr_out = model(x_dwi_padded, x_b0, meta_padded, valid_mask_dwi, valid_mask_b0)

                    # Elementweiser L1-Loss
                    loss_map = criterion(sr_out, hr_target)
                    
                    # Masked Loss: Fehler für Voxel innerhalb des Hirns durch Anzahl der Gehirn-Voxel teilen
                    loss = (loss_map * hr_mask).sum() / (hr_mask.sum() * sr_out.shape[1] + 1e-8)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                sr_out = model(x_dwi_padded, x_b0, meta_padded, valid_mask_dwi, valid_mask_b0)
                loss_map = criterion(sr_out, hr_target)
                loss = (loss_map * hr_mask).sum() / (hr_mask.sum() * sr_out.shape[1] + 1e-8)
                loss.backward()
                optimizer.step()
                
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{EPOCHS}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        avg_train_loss = epoch_loss / len(train_loader)
        
        # ---  VALIDATION ---
        avg_val_loss = validate(model, val_loader, criterion, device, use_amp=(scaler is not None))
        
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