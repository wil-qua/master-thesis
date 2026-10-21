# config.py
import os

# Daten- und Trainings-Parameter
B0_THRESHOLD = 100
MIN_DWI_B = 900
MAX_DWI_B = 1100
BATCH_SIZE = 5
PATCH_SIZE = 32
DOWNSCALE_FACTOR = 2
NUM_PATCHES_PER_EPOCH = 100
NUM_VAL_PATCHES = 100
NUM_EPOCHS = 50
MAX_MEASUREMENTS = 35
NUM_TRAIN_PROBANDS = 60
NUM_VAL_PROBANDS = 12
LOSS_FUNCTION = "Masked L1 + SSIM"

MAIN_PATH = "/mnt/nas_io/anarchy/lohrd/MRI_data/HCP/Diffusion/3T/"  # UKE: "/home/wquan/Schreibtisch/Data/Diffusion/3T/"  # HO: "/mnt/nas_io/anarchy/lohrd/MRI_data/HCP/Diffusion/3T/"
TRAIN_SUBJECTS = sorted(os.listdir(MAIN_PATH))[:NUM_TRAIN_PROBANDS]
VAL_SUBJECTS = sorted(os.listdir(MAIN_PATH))[NUM_TRAIN_PROBANDS:NUM_TRAIN_PROBANDS+NUM_VAL_PROBANDS]


# --- TRAININGS-PFADE ---
TRAIN_NIFTI_FILES = [ os.path.join(MAIN_PATH, subj, "T1w", "Diffusion/data.nii.gz") for subj in TRAIN_SUBJECTS ]
TRAIN_BVAL_FILES = [ os.path.join(MAIN_PATH, subj, "T1w", "Diffusion/bvals") for subj in TRAIN_SUBJECTS ]
TRAIN_BVEC_FILES = [ os.path.join(MAIN_PATH, subj, "T1w", "Diffusion/bvecs") for subj in TRAIN_SUBJECTS ]
TRAIN_MASK_FILES = [ os.path.join(MAIN_PATH, subj, "T1w", "Diffusion/nodif_brain_mask.nii.gz") for subj in TRAIN_SUBJECTS ]


# --- VALIDIERUNGS-PFADE ---
VAL_NIFTI_FILES = [ os.path.join(MAIN_PATH, subj, "T1w", "Diffusion/data.nii.gz") for subj in VAL_SUBJECTS ]
VAL_BVAL_FILES = [ os.path.join(MAIN_PATH, subj, "T1w", "Diffusion/bvals") for subj in VAL_SUBJECTS ]
VAL_BVEC_FILES = [ os.path.join(MAIN_PATH, subj, "T1w", "Diffusion/bvecs") for subj in VAL_SUBJECTS ]
VAL_MASK_FILES = [ os.path.join(MAIN_PATH, subj, "T1w", "Diffusion/nodif_brain_mask.nii.gz") for subj in VAL_SUBJECTS ]
