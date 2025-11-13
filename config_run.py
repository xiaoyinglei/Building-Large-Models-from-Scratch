"""
Training configuration parameters.

Modify these settings directly in this file to customize training behavior.
No command-line arguments needed - just edit this file and run main.py.
"""

# Device configuration
DEVICE = None  # None = auto-select (mps > cuda > cpu), or specify: "cpu", "cuda", "mps"

# Model configuration
USE_SMALL = True  # True = small model (testing), False = full model
USE_DEFAULT = True  # Use default large config (overrides USE_SMALL if True)
CONFIG_PATH = "config.json"  # Custom config file path (only used if USE_SMALL=False and USE_DEFAULT=False)

# Training mode
QUICK_MODE = True  # True = 1 epoch quick test, False = full training config

# Batch size (optional override)
BATCH_SIZE = None  # None = use default from config, or specify a number like 8, 16, 32

# Random seed
SEED = 123

# Logging
LOG_EVERY = 0  # Log batch loss every N steps (0 = no per-batch logging)
