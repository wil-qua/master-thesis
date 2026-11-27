# train_spatial_only.py
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

import config
from dataset import load_multi_subjects, release_subject_memmap_pages
from train import DegradationLayer, Masked3DSSIMAndL1Loss, save_checkpoint
from spatial_only_model import Model_SpatialOnlySR
from spatial_only_dataset import SpatialOnlyPatchDataset
from utils import plot_training_loss


@torch.no_grad()
def validate_spatial_only(model, val_loader, criterion, device, use_amp=False):
    model.eval()
    val_loss_sum = 0.0
    num_batches = 0

    for lr_in, hr_target, hr_mask in val_loader:
        lr_in = lr_in.to(device)
        hr_target = hr_target.to(device)
        hr_mask = hr_mask.to(device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            sr_out = model(lr_in)
            loss_sr = criterion(sr_out, hr_target, hr_mask)

        val_loss_sum += loss_sr.item()
        num_batches += 1

    return val_loss_sum / max(num_batches, 1)


def main():
    os.makedirs("Figures", exist_ok=True)
    os.makedirs("Models", exist_ok=True)

    USE_B0 = True
    NOISE_STD = 0.0  # noise turned off

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nNutze Gerät: {device}")

    print("Lade Trainings-Probanden...")
    train_cache = load_multi_subjects(
        config.TRAIN_NIFTI_FILES, config.TRAIN_BVAL_FILES,
        config.TRAIN_BVEC_FILES, config.TRAIN_MASK_FILES,
        config.B0_THRESHOLD, config.MIN_DWI_B, config.MAX_DWI_B,
        patch_size=config.PATCH_SIZE,
    )

    print("\nLade Validierungs-Probanden...")
    val_cache = load_multi_subjects(
        config.VAL_NIFTI_FILES, config.VAL_BVAL_FILES,
        config.VAL_BVEC_FILES, config.VAL_MASK_FILES,
        config.B0_THRESHOLD, config.MIN_DWI_B, config.MAX_DWI_B,
        patch_size=config.PATCH_SIZE,
    )

    train_dataset = SpatialOnlyPatchDataset(
        subjects_cache=train_cache,
        num_patches_per_epoch=config.NUM_PATCHES_PER_EPOCH,
        patch_size=config.PATCH_SIZE,
        downscale_factor=config.DOWNSCALE_FACTOR,
        noise_std=NOISE_STD,
        use_b0=USE_B0,
        is_validation=False,
    )

    val_dataset = SpatialOnlyPatchDataset(
        subjects_cache=val_cache,
        num_patches_per_epoch=getattr(config, "NUM_VAL_PATCHES", 100),
        patch_size=config.PATCH_SIZE,
        downscale_factor=config.DOWNSCALE_FACTOR,
        noise_std=NOISE_STD,
        use_b0=USE_B0,
        is_validation=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False
    )

    in_channels = 2 if USE_B0 else 1
    model = Model_SpatialOnlySR(in_channels=in_channels).to(device)
    criterion = Masked3DSSIMAndL1Loss(window_size=7, data_range=1.0, alpha=0.0).to(device)
    degradation = DegradationLayer(sigma=0.8, downscale_factor=config.DOWNSCALE_FACTOR).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    CHECKPOINT_PATH = "Models/latest_checkpoint_spatial_only.pth"
    BEST_MODEL_PATH = "Models/model_spatial_only_best.pth"

    epoch = 0
    global_step = 0
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    if os.path.exists(CHECKPOINT_PATH):
        print(f"\nLade Checkpoint: {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scaler is not None and ckpt["scaler_state_dict"] is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        best_val_loss = ckpt["best_val_loss"]
        epochs_without_improvement = ckpt["epochs_without_improvement"]
        criterion.alpha = ckpt["criterion_alpha"]
        train_loss_history = ckpt["train_loss_history"]
        val_loss_history = ckpt["val_loss_history"]

    run = wandb.init(
        project="master-thesis",
        entity="william-quan18-uke",
        name="Model_SpatialOnlySR_ablation",
        config={
            "model": "Model_SpatialOnlySR",
            "in_channels": in_channels,
            "use_b0": USE_B0,
            "noise_std": NOISE_STD,
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "batch_size": config.BATCH_SIZE,
            "patch_size": config.PATCH_SIZE,
            "downscale_factor": config.DOWNSCALE_FACTOR,
            "num_patches_per_epoch": config.NUM_PATCHES_PER_EPOCH,
            "lambda_dc": config.LAMBDA_DC,
            "degradation_sigma": 0.8,
        },
    )

    PATIENCE = 15

    while True:
        epoch += 1
        model.train()
        epoch_sr_loss = 0.0
        epoch_dc_loss = 0.0

        for batch_idx, (lr_in, hr_target, hr_mask) in enumerate(train_loader):
            global_step += 1

            if global_step >= 500 and criterion.alpha < 0.5:
                criterion.alpha = 0.5
                print(f"\n[Update {global_step}] Loss umgestellt auf 50% SSIM + 50% L1 (alpha=0.5)\n")
                best_val_loss = float("inf")
                epochs_without_improvement = 0

            lr_in = lr_in.to(device)
            hr_target = hr_target.to(device)
            hr_mask = hr_mask.to(device)
            mask_brain_lr = hr_mask[
                :, :, ::config.DOWNSCALE_FACTOR, ::config.DOWNSCALE_FACTOR, ::config.DOWNSCALE_FACTOR
            ]

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=(scaler is not None)):
                sr_out = model(lr_in)

                # 1. Super-Resolution Loss (identisch zu Model A)
                loss_sr = criterion(sr_out, hr_target, hr_mask)

                # 2. Data Consistency Loss (identischer Forward-Operator auf Kanal 0 = LR-DWI)
                pred_lr = degradation(sr_out)
                diff = torch.abs(pred_lr - lr_in[:, 0:1])
                loss_dc = (diff * mask_brain_lr).sum() / (mask_brain_lr.sum() + 1e-8)

                loss = (1.0 - config.LAMBDA_DC) * loss_sr + config.LAMBDA_DC * loss_dc

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_sr_loss += loss_sr.item()
            epoch_dc_loss += loss_dc.item()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch}] Batch [{batch_idx}/{len(train_loader)}] | "
                    f"SR-Loss: {loss_sr.item():.4f} | DC-Loss: {loss_dc.item():.4f}"
                )

        avg_train_loss = epoch_sr_loss / len(train_loader)
        avg_dc_loss = epoch_dc_loss / len(train_loader)
        avg_val_loss = validate_spatial_only(
            model, val_loader, criterion, device, use_amp=(scaler is not None)
        )

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            best_marker = " [* Best Val *]"
        else:
            epochs_without_improvement += 1
            best_marker = ""

        run.log({
            "epoch": epoch,
            "train/sr_loss": avg_train_loss,
            "train/dc_loss": avg_dc_loss,
            "val/sr_loss": avg_val_loss,
            "val/best_sr_loss": best_val_loss,
        })

        print(
            f"==> Ende Epoch {epoch} | Train SR: {avg_train_loss:.4f} | "
            f"Train DC: {avg_dc_loss:.4f} | Val SR: {avg_val_loss:.4f}{best_marker}\n"
        )

        save_checkpoint(
            path=CHECKPOINT_PATH,
            epoch=epoch,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            best_val_loss=best_val_loss,
            epochs_without_improvement=epochs_without_improvement,
            train_loss_history=train_loss_history,
            val_loss_history=val_loss_history,
        )

        release_subject_memmap_pages(train_cache)
        release_subject_memmap_pages(val_cache)

        if criterion.alpha == 0.5 and epochs_without_improvement >= PATIENCE:
            print(f"\nEarly Stopping nach Epoche {epoch} ({PATIENCE} Epochen ohne Verbesserung).")
            break

    plot_training_loss(
        train_loss_history,
        val_loss_history=val_loss_history,
        config=config,
        save_path="Figures/training_loss_spatial_only.png",
    )
    torch.save(model.state_dict(), "Models/model_spatial_only.pth")
    run.finish()


if __name__ == "__main__":
    main()