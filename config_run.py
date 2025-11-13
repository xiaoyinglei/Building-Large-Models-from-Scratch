"""
Training configuration parameters.

Modify these settings directly in this file to customize training behavior.
No command-line arguments needed - just edit this file and run main.py.

Three training modes available:
  1. QUICK_MODE=True  → Quick test (1 epoch, small batch, 128 context)
  2. QUICK_MODE=False → Full training (10 epochs, 32 batch, 256 context)
  3. CUSTOM_MODE=True → Custom/free training with user-defined parameters
"""

# Device configuration
DEVICE = None  # None = auto-select (mps > cuda > cpu), or specify: "cpu", "cuda", "mps"

# Model configuration
USE_SMALL = True  # True = small model (testing), False = full model
USE_DEFAULT = True  # Use default large config (overrides USE_SMALL if True)
CONFIG_PATH = "config.json"  # Custom config file path (only used if USE_SMALL=False and USE_DEFAULT=False)

# Training mode selector
QUICK_MODE = False  # True = 1 epoch quick test, False = full training config
CUSTOM_MODE = True  # True = use custom_params (overrides QUICK_MODE and USE_SMALL)

# Custom training parameters (only used when CUSTOM_MODE=True)
# Uncomment and modify to customize training. All parameters are optional and will override defaults.
custom_params = {
    # === Training hyperparameters ===
    'num_epochs': 10,              # Number of training epochs
    'batch_size': 64,             # Batch size (8, 16, 32, 64, ...)
    'learning_rate': 5e-4,        # Learning rate (e.g., 3e-4, 1e-3, 5e-4)
    'weight_decay': 0.05,         # L2 regularization strength (0.0 to 0.5)
    
    # === Evaluation and logging ===
    'eval_freq': 25,              # Evaluate every N training steps
    'eval_iter': 2,               # Number of batches for evaluation
    
    # === Data and context ===
    'max_length': 512,            # Maximum sequence length (128, 256, 512, 1024)
    'stride': 256,                # Stride for sliding window (max_length // 2 typical)
    'start_context': "Once upon a time",  # Starting text for generation
    
    # === Computation ===
    'num_workers': 0,             # DataLoader workers (0 for CPU, 2-4 for GPU)
    
    # === Model architecture overrides (optional, advanced) ===
    # Uncomment to override model architecture:
    # 'model_overrides': {
    #     'n_heads': 8,           # Number of attention heads (must divide emb_dim)
    #     'n_layers': 6,          # Number of transformer layers
    #     'emb_dim': 512,         # Embedding dimension
    #     'drop_rate': 0.1,       # Dropout rate (0.0 to 0.5)
    # }
}

# Batch size (optional override for quick/full mode only)
BATCH_SIZE = None  # None = use default from config, or specify a number like 8, 16, 32

# === Text Generation Configuration ===
# Customize text generation behavior during training (used for generating sample text at each epoch)
# You can either set custom_gen_params (dict) or use a pre-defined strategy

# Option 1: Use pre-defined generation strategy (set CUSTOM_GEN_MODE to one of these)
GENERATION_STRATEGY = 'greedy'  # Options: 'greedy', 'top_k', 'top_p'


# Option 2: Custom generation parameters (only used when this dict is not empty)
custom_gen_params = {
    # Uncomment to customize generation:
    # 'max_new_tokens': 100,      # Number of tokens to generate (default: 50)
    # 'strategy': 'top_k',        # Sampling strategy: 'greedy' | 'top_k' | 'top_p' (default: 'greedy')
    # 'top_k': 50,                # Top-k value for top_k sampling (default: 50)
    # 'top_p': 0.9,               # Top-p value for nucleus sampling (default: 0.9)
    # 'temperature': 0.8,         # Temperature scaling (<1=sharper, >1=flatter, default: 1.0)
}

# Random seed
SEED = 123

# Logging
LOG_EVERY = 0  # Log batch loss every N steps (0 = no per-batch logging)

