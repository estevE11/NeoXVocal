import os

# =============================================================================
# Data Paths
# =============================================================================
BASE_DIR = '/home/usuaris/veussd/roger.esteve.sanchez/adresso/processed_data/diagnosis/train'
AD_TEXT_DIR = '/home/usuaris/veussd/roger.esteve.sanchez/adresso/processed_data/diagnosis/train/ad'
CN_TEXT_DIR = '/home/usuaris/veussd/roger.esteve.sanchez/adresso/processed_data/diagnosis/train/cn'
AD_CSV = '/home/usuaris/veussd/roger.esteve.sanchez/adresso/processed_data/diagnosis/train/ad/audio_features_ad.csv'
CN_CSV = '/home/usuaris/veussd/roger.esteve.sanchez/adresso/processed_data/diagnosis/train/cn/audio_features_cn.csv'
AD_EMBEDDING_CSV = '/home/usuaris/veussd/roger.esteve.sanchez/adresso/processed_data/diagnosis/train/ad/audio_embeddings_ad.csv'
CN_EMBEDDING_CSV = '/home/usuaris/veussd/roger.esteve.sanchez/adresso/processed_data/diagnosis/train/cn/audio_embeddings_cn.csv'

# =============================================================================
# Text Model Configuration
# =============================================================================
TEXT_EMBEDDING_MODEL = 'microsoft/deberta-v3-base'

# =============================================================================
# Feature Dimensions
# =============================================================================
NUM_MFCC_FEATURES = 47
NUM_EMBEDDING_FEATURES = 768
AUDIO_CHANNELS = 1

# =============================================================================
# Device Configuration
# =============================================================================
CUDA = True

# =============================================================================
# Training Hyperparameters (set via CLI or here)
# =============================================================================
# These are placeholder strings - set actual values via CLI args or replace here
BATCH_SIZE = 'Desired number of samples per training batch'
EPOCHS = 'Total number of training epochs'
LEARNING_RATE = 'Learning rate for the optimizer'
WEIGHT_DECAY = 'Weight decay (L2 regularization) rate'
NUM_FOLDS = 'Number of folds for cross-validation'
SAVE_BEST_MODEL = 'Flag to save only the best-performing model (True/False)'
EARLY_STOPPING_PATIENCE = 'Number of epochs with no improvement to trigger early stopping'
 
# =============================================================================
# Model Architecture Defaults
# These values are used if not overridden via CLI arguments
# =============================================================================

# --- Transformer Encoder ---
TRANSFORMER_NUM_HEADS = 8           # Number of attention heads
TRANSFORMER_NUM_LAYERS = 2          # Number of transformer encoder layers
TRANSFORMER_DIM_FEEDFORWARD = None  # Feedforward dimension (None = 4 * hidden_size)
TRANSFORMER_DROPOUT = 0.35          # Dropout in transformer encoder
TRANSFORMER_ACTIVATION = 'gelu'     # Activation function ('relu' or 'gelu')

# --- Feature Projection ---
FEATURE_PROJECTION_DROPOUT = 0.3    # Dropout in audio/embedding projection layers

# --- Classifier Head ---
CLASSIFIER_HIDDEN_LAYERS = 1        # Number of hidden layers (0 = linear projection)
CLASSIFIER_HIDDEN_WIDTH = None      # Width of hidden layers (None = hidden_size // 2)
CLASSIFIER_DROPOUT = 0.45           # Dropout in classifier
CLASSIFIER_ACTIVATION = 'relu'      # Activation ('relu', 'gelu', 'tanh', 'leaky_relu')
NUM_CLASSES = 1                     # Output classes (1 for binary with BCEWithLogitsLoss)

# --- Text Model ---
FREEZE_TEXT_MODEL = False           # Freeze entire text model
FREEZE_TEXT_MODEL_LAYERS = None     # Freeze N layers from bottom (None = don't freeze)

# --- Pooling ---
POOLING_STRATEGY = 'first'          # 'first', 'mean', 'max', 'cls_token'

# --- Training Schedule ---
GRADIENT_CLIP_NORM = 1.0            # Max norm for gradient clipping
SCHEDULER_FACTOR = 0.5              # Factor for ReduceLROnPlateau
SCHEDULER_PATIENCE = 5              # Patience for ReduceLROnPlateau

# =============================================================================
# Output Paths
# =============================================================================
SAVE_MODEL_PATH = 'path/to/results/folder/model'
LOG_PATH = 'path/to/results/folder/training.log'

