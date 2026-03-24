from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
RESULTS_DIR = DATA_DIR / "results"
PAPER_DIR = PROJECT_ROOT / "an-unsupervised-embeddings-model-for-ocean-crust-lithology"

# Zenodo record IDs
ZENODO_RECORD_LILY = 8408297
ZENODO_RECORD_DENSITY = 10001855

# Training variables (order must match tensor columns)
TRAINING_VARIABLES = [
    "bulk_density",
    "magnetic_susceptibility",
    "NGR",
    "R",
    "G",
    "B",
]

# Model architecture
INPUT_DIM = 6
LATENT_DIM = 10
HIDDEN_DIMS = [128, 64]
NUM_LITHOLOGY_CLASSES = 139

# Training
NUM_EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MASK_FRACTION = 0.1
MAX_GRAD_NORM = 1.0

# Beta annealing (KL weight)
BETA_START = 1e-10
BETA_END = 0.075
BETA_ANNEAL_EPOCHS = 50

# Alpha annealing (classification weight, semi-supervised only)
ALPHA_START = 0.0
ALPHA_END = 0.01
ALPHA_ANNEAL_EPOCHS = 50

# Decoder log-variance clamp range
LOGVAR_CLAMP_MIN = -6.0
LOGVAR_CLAMP_MAX = 6.0

# Data preprocessing
BIN_SIZE_CM = 20

# Bootstrap evaluation
BOOTSTRAP_ITERATIONS = 100
BOOTSTRAP_TEST_FRACTION = 0.2

# Random seed
SEED = 42
