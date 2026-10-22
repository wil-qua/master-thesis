import matplotlib.pyplot as plt

def plot_training_loss(loss_history, val_loss_history=None, config=None, save_path="Figures/training_loss_model_A.png"):
    """
    Erstellt ein aussagekräftiges Liniendiagramm des Trainingsverlaufs
    inklusive einer Infobox mit allen relevanten Hyperparametern.
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    epochs = range(1, len(loss_history) + 1)
    ax.plot(epochs, loss_history, marker='o', linestyle='-', color='#1f77b4', 
            linewidth=2, markersize=4, label=f"Training {getattr(config, 'LOSS_FUNCTION')} Loss")

    # Kennzahlen berechnen
    min_loss = min(loss_history)
    min_epoch = loss_history.index(min_loss) + 1
    final_loss = loss_history[-1]

    if val_loss_history is not None and len(val_loss_history) > 0:
        ax.plot(epochs, val_loss_history, marker='s', linestyle='--', color='#ff7f0e',
                linewidth=2, markersize=4, label=f"Validation {getattr(config, 'LOSS_FUNCTION')} Loss")
        min_val_loss = min(val_loss_history)
        min_val_epoch = val_loss_history.index(min_val_loss) + 1
        final_val_loss = val_loss_history[-1]
        ax.plot(min_val_epoch, min_val_loss, marker='*', markersize=12, color='#d62728', 
                label=f'Min Val Loss: {min_val_loss:.4f} (Epoche {min_val_epoch})')
    else:
        # Besten Punkt markieren
        ax.plot(min_epoch, min_loss, marker='*', markersize=12, color='#d62728', 
                label=f'Min Loss: {min_loss:.4f} (Epoche {min_epoch})')

    # Titel und Achsen
    ax.set_title('Model A – Training Curve & Hyperparameter', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(f"Average {getattr(config, 'LOSS_FUNCTION')} Loss", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    if len(loss_history) > 1:
        ax.set_xlim(1, len(loss_history))
    else:
        ax.set_xlim(0.5, 1.5)

    # --- Parameter-Textblock zusammenstellen ---
    param_lines = ["Trainings-Parameter:"]

    if config is not None:
        # Attribute aus dem config-Modul oder einem Dict auslesen
        def get_val(key, default="n/a"):
            if isinstance(config, dict):
                return config.get(key, default)
            return getattr(config, key, default)

        param_lines.append(f"• Batch Size: {get_val('BATCH_SIZE')}")
        param_lines.append(f"• Patch Size: {get_val('PATCH_SIZE')}³")
        param_lines.append(f"• Downscale Factor: {get_val('DOWNSCALE_FACTOR')}x")
        param_lines.append(f"• Patches / Epoche: {get_val('NUM_PATCHES_PER_EPOCH')}")
        param_lines.append(f"• Val-Patches: {get_val('NUM_VAL_PATCHES', 25)}")
        param_lines.append(f"• Epochs : {get_val('NUM_EPOCHS', len(loss_history))}")
        param_lines.append(f"• Training Probands: {get_val('NUM_TRAIN_PROBANDS', len(loss_history))}")
        param_lines.append(f"• Validation Probands: {get_val('NUM_VAL_PROBANDS', len(loss_history))}")

        if hasattr(config, 'MAX_MEASUREMENTS') or (isinstance(config, dict) and 'MAX_MEASUREMENTS' in config):
            param_lines.append(f"• Max Measurements: {get_val('MAX_MEASUREMENTS')}")

    param_lines.append("--------------------")
    param_lines.append(f"Final Train: {final_loss:.4f} (Min: {min_loss:.4f})")
    if val_loss_history is not None and len(val_loss_history) > 0:
        param_lines.append(f"Final Val:   {final_val_loss:.4f} (Min: {min_val_loss:.4f})")

    info_text = "\n".join(param_lines)

    # Infobox oben rechts im Diagramm platzieren
    ax.text(
        0.97, 0.95, info_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='right',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', edgecolor='#cccccc', alpha=0.92)
    )

    ax.legend(loc='upper center', framealpha=0.9)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\nLoss-Plot mit Parametern erfolgreich gespeichert unter: {save_path}")