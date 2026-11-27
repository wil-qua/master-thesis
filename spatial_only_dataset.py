# spatial_only_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from dataset import generate_and_augment_lr_patch


class SpatialOnlyPatchDataset(Dataset):
    """
    Liefert (lr_input, hr_target, hr_mask) für reines Spatial SR:
    - Zieht Patches stratifiziert aus denselben 6 anatomischen Zonen wie Model A.
    - Nutzt alle verfügbaren DWI-Richtungen (90) als Trainingsbeispiele.
    - Kanal 0: LR-DWI (eine Richtung), Kanal 1 (optional): gemitteltes LR-b0.
    """
    def __init__(
        self,
        subjects_cache,
        num_patches_per_epoch,
        patch_size=64,
        downscale_factor=2,
        noise_std=0.00,  # noise is turned off
        use_b0=True,
        max_b0=5,
        is_validation=False,
    ):
        self.subjects_cache = subjects_cache
        self.num_patches = num_patches_per_epoch
        self.patch_size = patch_size
        self.downscale_factor = downscale_factor
        self.noise_std = noise_std
        self.use_b0 = use_b0
        self.max_b0 = max_b0
        self.is_validation = is_validation

        self.frozen_val_specs = None
        if self.is_validation:
            self._freeze_validation_combinations()

    def _freeze_validation_combinations(self):
        rng = np.random.RandomState(42)
        half_p = self.patch_size // 2
        zone_names = [
            "single_fiber_wm",
            "crossing_fiber_wm",
            "cortical_boundary",
            "subcortical_gm",
            "brainstem_cerebellum",
            "csf_boundary",
        ]

        self.frozen_val_specs = []
        for i in range(self.num_patches):
            subj_idx = int(rng.randint(0, len(self.subjects_cache)))
            subj = self.subjects_cache[subj_idx]
            x_dim, y_dim, z_dim, n_dwi_total = subj["dwi"].shape
            n_b0_total = subj["b0"].shape[-1]

            zone_name = zone_names[i % len(zone_names)]
            coords_pool = subj["zone_pools"][zone_name]
            min_frac = 0.40 if zone_name in ("cortical_boundary", "brainstem_cerebellum") else 0.75

            for attempt in range(50):
                c_idx = int(rng.randint(0, len(coords_pool)))
                cx, cy, cz = coords_pool[c_idx]
                cx = int(np.clip(cx, half_p, x_dim - half_p))
                cy = int(np.clip(cy, half_p, y_dim - half_p))
                cz = int(np.clip(cz, half_p, z_dim - half_p))

                hr_m = subj["mask"][cx - half_p:cx + half_p, cy - half_p:cy + half_p, cz - half_p:cz + half_p]
                if np.mean(hr_m > 0) >= min_frac or attempt == 49:
                    break

            dwi_idx = int(rng.randint(0, n_dwi_total))
            n_b0 = int(rng.randint(1, min(self.max_b0, n_b0_total) + 1))
            b0_indices = rng.choice(n_b0_total, size=n_b0, replace=False)

            self.frozen_val_specs.append({
                "subj_idx": subj_idx,
                "center": (cx, cy, cz),
                "dwi_idx": dwi_idx,
                "b0_indices": b0_indices,
            })

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        half_p = self.patch_size // 2

        if self.is_validation:
            spec = self.frozen_val_specs[idx]
            subj = self.subjects_cache[spec["subj_idx"]]
            cx, cy, cz = spec["center"]
            dwi_idx = spec["dwi_idx"]
            b0_indices = spec["b0_indices"]
            current_noise_std = self.noise_std
        else:
            subj_idx = int(np.random.randint(0, len(self.subjects_cache)))
            subj = self.subjects_cache[subj_idx]
            x_dim, y_dim, z_dim, n_dwi_total = subj["dwi"].shape
            n_b0_total = subj["b0"].shape[-1]

            zone_names = list(subj["zone_pools"].keys())
            zone_name = zone_names[int(np.random.randint(0, len(zone_names)))]
            coords_pool = subj["zone_pools"][zone_name]
            min_frac = 0.40 if zone_name in ("cortical_boundary", "brainstem_cerebellum") else 0.75

            for attempt in range(50):
                c_idx = int(np.random.randint(0, len(coords_pool)))
                cx, cy, cz = coords_pool[c_idx]
                cx = int(np.clip(cx, half_p, x_dim - half_p))
                cy = int(np.clip(cy, half_p, y_dim - half_p))
                cz = int(np.clip(cz, half_p, z_dim - half_p))

                hr_m = subj["mask"][cx - half_p:cx + half_p, cy - half_p:cy + half_p, cz - half_p:cz + half_p]
                if np.mean(hr_m > 0) >= min_frac or attempt == 49:
                    break

            dwi_idx = int(np.random.randint(0, n_dwi_total))
            n_b0 = int(np.random.randint(1, min(self.max_b0, n_b0_total) + 1))
            b0_indices = np.random.choice(n_b0_total, size=n_b0, replace=False)
            current_noise_std = self.noise_std

        sx = slice(cx - half_p, cx + half_p)
        sy = slice(cy - half_p, cy + half_p)
        sz = slice(cz - half_p, cz + half_p)

        hr_mask = subj["mask"][sx, sy, sz]
        hr_dwi = subj["dwi"][sx, sy, sz, dwi_idx:dwi_idx + 1]  # [P, P, P, 1]

        # Identische LR-Degradation (Blur sigma=0.8 + Subsampling + Rician Noise)
        dummy_bvec_dwi = np.zeros((1, 3), dtype=np.float32)
        lr_dwi, _, _ = generate_and_augment_lr_patch(
            hr_dwi,
            hr_mask,
            dummy_bvec_dwi,
            rot_spec=None,
            downscale_factor=self.downscale_factor,
            sigma=0.8,
            noise_std=current_noise_std,
        )  # [P/2, P/2, P/2, 1]

        if self.use_b0:
            hr_b0 = subj["b0"][sx, sy, sz, b0_indices]  # [P, P, P, n_b0]
            dummy_bvec_b0 = np.zeros((hr_b0.shape[-1], 3), dtype=np.float32)
            lr_b0, _, _ = generate_and_augment_lr_patch(
                hr_b0,
                hr_mask,
                dummy_bvec_b0,
                rot_spec=None,
                downscale_factor=self.downscale_factor,
                sigma=0.8,
                noise_std=current_noise_std,
            )
            # Über die gezogenen b0-Bilder mitteln -> [P/2, P/2, P/2, 1]
            lr_b0_mean = np.mean(lr_b0, axis=-1, keepdims=True).astype(np.float32)
            lr_in = np.concatenate([lr_dwi, lr_b0_mean], axis=-1)  # [P/2, P/2, P/2, 2]
        else:
            lr_in = lr_dwi  # [P/2, P/2, P/2, 1]

        # [X, Y, Z, C] -> [C, X, Y, Z]
        lr_tensor = torch.from_numpy(np.transpose(lr_in, (3, 0, 1, 2))).float()
        hr_target_tensor = torch.from_numpy(np.transpose(hr_dwi, (3, 0, 1, 2))).float()
        hr_mask_tensor = torch.from_numpy(hr_mask.astype(np.float32)).unsqueeze(0)

        return lr_tensor, hr_target_tensor, hr_mask_tensor