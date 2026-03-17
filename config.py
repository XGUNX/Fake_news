# =============================================================================
# config.py
# -----------------------------------------------------------------------------
# Central configuration file for the Fake News Detection project.
# All paths, hyperparameters, and constants are defined here.
# Every other module imports from this file — nothing is hardcoded elsewhere.
# =============================================================================

import os

# =============================================================================
# PATHS
# =============================================================================

# Root directory of the project (directory where this file lives)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Where the raw CSV files from Kaggle are stored — never modify these files
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

# Where we save the cleaned/merged data after preprocessing
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Trained model weights are saved here
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Evaluation outputs: plots, metrics, confusion matrices
RESULTS_DIR = os.path.join(BASE_DIR, "results")


# =============================================================================
# DATASET FILES
# -----------------------------------------------------------------------------
# The Kaggle FakeNewsNet dataset contains 4 CSV files.
# Labels are NOT inside the files — they come from the filename itself:
#   *_fake_* → label 0  (fake news)
#   *_real_* → label 1  (real news)
# We assign labels manually when loading in preprocess.py.
# The .txt files in the dataset are tweet IDs — we ignore them (no Twitter API).
# =============================================================================

RAW_FILES = {
    "buzzfeed_fake":   "BuzzFeed_fake_news_content.csv",
    "buzzfeed_real":   "BuzzFeed_real_news_content.csv",
    "politifact_fake": "PolitiFact_fake_news_content.csv",
    "politifact_real": "PolitiFact_real_news_content.csv",
}

# Numeric label assigned to each file based on its name
FILE_LABELS = {
    "buzzfeed_fake":   0,
    "buzzfeed_real":   1,
    "politifact_fake": 0,
    "politifact_real": 1,
}

# Source tag to track which dataset each row comes from (useful in EDA)
FILE_SOURCES = {
    "buzzfeed_fake":   "buzzfeed",
    "buzzfeed_real":   "buzzfeed",
    "politifact_fake": "politifact",
    "politifact_real": "politifact",
}

# Column names in the CSV files that contain the text content
# (exact names will be confirmed in 01_exploration.ipynb)
TITLE_COLUMN = "title"
TEXT_COLUMN  = "text"

# Columns to keep after loading; everything else is dropped
COLUMNS_TO_KEEP = ["title", "text", "author", "published", "site_url", "domain_rank"]

# Name of the label column we create ourselves during loading
LABEL_COLUMN  = "label"     # 0 = fake, 1 = real

# Name of the source column we create to distinguish BuzzFeed vs PolitiFact
SOURCE_COLUMN = "source"    # "buzzfeed" or "politifact"

# Human-readable label names used in plots and the Streamlit app
LABEL_NAMES = {0: "Fake", 1: "Real"}


# =============================================================================
# DATA SPLITS
# =============================================================================

TRAIN_SIZE  = 0.8    # 80% for training
VAL_SIZE    = 0.1    # 10% for validation
TEST_SIZE   = 0.1    # 10% for final testing

# Fixed seed — same split every run for reproducibility
RANDOM_SEED = 42


# =============================================================================
# TEXT PREPROCESSING
# =============================================================================

# Template for combining title and body into one input string.
# We separate them with [SEP] so the model sees both but knows the boundary.
TEXT_COMBINATION_TEMPLATE = "{title} [SEP] {text}"

# Maximum number of tokens/words per sample.
# Texts longer than this are truncated; shorter ones are padded.
MAX_SEQUENCE_LENGTH = 256

# Minimum word frequency to include a word in the vocabulary (BiLSTM only)
MIN_WORD_FREQ = 2

# Vocabulary size cap (BiLSTM only — RoBERTa has its own built-in tokenizer)
MAX_VOCAB_SIZE = 20_000


# =============================================================================
# BASELINE MODEL — TF-IDF + Logistic Regression
# =============================================================================

BASELINE_MAX_FEATURES = 10_000   # number of TF-IDF features to keep
BASELINE_NGRAM_RANGE  = (1, 2)   # use both unigrams and bigrams
BASELINE_MAX_ITER     = 1_000    # max solver iterations for convergence
BASELINE_C            = 1.0      # regularization strength (lower = stronger)


# =============================================================================
# BiLSTM MODEL
# =============================================================================

BILSTM_EMBEDDING_DIM  = 100     # must match the GloVe file you download
                                  # available: 50 / 100 / 200 / 300
BILSTM_HIDDEN_DIM     = 128     # LSTM hidden units per direction
BILSTM_NUM_LAYERS     = 2       # number of stacked BiLSTM layers
BILSTM_DROPOUT        = 0.3     # dropout between layers (regularization)
BILSTM_BIDIRECTIONAL  = True    # read sequences in both directions

BILSTM_BATCH_SIZE     = 32
BILSTM_EPOCHS         = 15
BILSTM_LEARNING_RATE  = 1e-3
BILSTM_PATIENCE       = 3       # early stopping: halt after N bad epochs


# =============================================================================
# RoBERTa MODEL
# =============================================================================

# roberta-base is well suited here: the dataset is small, base is enough
ROBERTA_MODEL_NAME    = "roberta-base"
ROBERTA_MAX_LENGTH    = 256        # tokenizer will truncate/pad to this length

ROBERTA_BATCH_SIZE    = 16         # smaller than BiLSTM — transformer is heavier
ROBERTA_EPOCHS        = 5          # fine-tuning converges in very few epochs
ROBERTA_LEARNING_RATE = 2e-5       # standard lr for fine-tuning transformers
ROBERTA_WARMUP_STEPS  = 50         # linear warmup before the scheduler starts
ROBERTA_WEIGHT_DECAY  = 0.01       # L2 regularization on the model weights
ROBERTA_PATIENCE      = 2          # early stopping patience


# =============================================================================
# TRAINING — SHARED SETTINGS
# =============================================================================

# Device for PyTorch — set to "cpu" if you have no GPU
DEVICE = "cuda"

# Which metric to use when selecting the best checkpoint during training
BEST_MODEL_METRIC = "f1"    # options: "accuracy", "f1", "loss"


# =============================================================================
# STREAMLIT APP
# =============================================================================

# Model the app loads by default for live inference
APP_DEFAULT_MODEL = "roberta"    # options: "baseline", "bilstm", "roberta"

# Labels shown in the app UI
LABEL_MAP = {
    0: "🔴 Fake News",
    1: "🟢 Real News",
}


# =============================================================================
# AUTO-CREATE OUTPUT DIRECTORIES
# -----------------------------------------------------------------------------
# Executed every time config.py is imported — guarantees all folders exist
# before any other module tries to read or write files.
# =============================================================================

for _dir in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(_dir, exist_ok=True)