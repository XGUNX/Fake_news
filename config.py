# =============================================================================
# config.py
# -----------------------------------------------------------------------------
# Fichier de configuration central du projet.
# Tous les chemins, hyperparamètres et constantes sont définis ici.
# Tous les autres fichiers importent depuis config.py — rien n'est codé en dur ailleurs.
#
# Datasets utilisés :
#   1. FakeNewsNet    (Kaggle — mdepak)         ~422  articles
#   2. Fake and Real  (Kaggle — clmentbisaillon) ~44 000 articles
#
# Les deux sont fusionnés dans preprocess.py pour former un seul dataset.
# =============================================================================

import os

# =============================================================================
# CHEMINS
# =============================================================================

# racine du projet — dossier où se trouve ce fichier
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# fichiers CSV bruts — on ne les modifie jamais
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')

# données nettoyées sauvegardées après preprocessing
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# poids des modèles entraînés
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# graphiques, matrices de confusion, métriques
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


# =============================================================================
# DATASET 1 — FakeNewsNet (Kaggle — mdepak)
# -----------------------------------------------------------------------------
# 4 fichiers CSV, labels encodés dans le nom du fichier.
# _fake_ → 0 (fake), _real_ → 1 (real)
# Colonnes utiles : title, text
# =============================================================================

RAW_FILES = {
    'buzzfeed_fake'  : 'BuzzFeed_fake_news_content.csv',
    'buzzfeed_real'  : 'BuzzFeed_real_news_content.csv',
    'politifact_fake': 'PolitiFact_fake_news_content.csv',
    'politifact_real': 'PolitiFact_real_news_content.csv',
}

# label associé à chaque fichier FakeNewsNet
FILE_LABELS = {
    'buzzfeed_fake'  : 0,
    'buzzfeed_real'  : 1,
    'politifact_fake': 0,
    'politifact_real': 1,
}

# source associée à chaque fichier — utile pour l'analyse dans l'exploration
FILE_SOURCES = {
    'buzzfeed_fake'  : 'buzzfeed',
    'buzzfeed_real'  : 'buzzfeed',
    'politifact_fake': 'politifact',
    'politifact_real': 'politifact',
}


# =============================================================================
# DATASET 2 — Fake and Real News (Kaggle — clmentbisaillon)
# -----------------------------------------------------------------------------
# 2 fichiers CSV, un pour le fake et un pour le real.
# Contrairement à FakeNewsNet, les colonnes sont : title, text, subject, date
# On n'utilise que title et text pour rester cohérent avec FakeNewsNet.
# =============================================================================

EXTRA_FILES = {
    'extra_fake': 'Fake.csv',
    'extra_real': 'True.csv',
}

# label associé à chaque fichier du dataset supplémentaire
EXTRA_LABELS = {
    'extra_fake': 0,
    'extra_real': 1,
}


# =============================================================================
# COLONNES ET LABELS
# =============================================================================

TITLE_COLUMN  = 'title'
TEXT_COLUMN   = 'text'
LABEL_COLUMN  = 'label'   # 0 = fake, 1 = real — créé à l'étape de chargement
SOURCE_COLUMN = 'source'  # 'buzzfeed', 'politifact', ou 'extra'

# noms lisibles utilisés dans les graphiques et l'app Streamlit
LABEL_NAMES = {0: 'Fake', 1: 'Real'}


# =============================================================================
# SPLITS
# =============================================================================

TRAIN_SIZE  = 0.8   # 80% entraînement
VAL_SIZE    = 0.1   # 10% validation
TEST_SIZE   = 0.1   # 10% test

# graine aléatoire — garantit les mêmes splits à chaque exécution
RANDOM_SEED = 42


# =============================================================================
# PREPROCESSING DU TEXTE
# =============================================================================

# template pour combiner titre et corps en un seul texte d'entrée
TEXT_COMBINATION_TEMPLATE = '{title} [SEP] {text}'

# longueur max en tokens/mots — les textes plus longs sont tronqués
# avec ~44 000 articles on peut se permettre d'être plus strict sur la longueur
MAX_SEQUENCE_LENGTH = 256

# fréquence minimale d'un mot pour être inclus dans le vocabulaire BiLSTM
# on monte à 3 car avec un grand corpus les mots rares sont vraiment du bruit
MIN_WORD_FREQ = 3

# taille max du vocabulaire BiLSTM
# on augmente car le corpus est beaucoup plus grand
MAX_VOCAB_SIZE = 50_000


# =============================================================================
# BASELINE — TF-IDF + Régression Logistique
# =============================================================================

BASELINE_MAX_FEATURES = 10_000   # on repasse à 10 000 — justifié avec ~35 000 articles train
BASELINE_NGRAM_RANGE  = (1, 2)
BASELINE_MAX_ITER     = 1_000
BASELINE_C            = 1.0      # on revient à 1.0 — le dataset est assez grand maintenant


# =============================================================================
# BILSTM
# =============================================================================

BILSTM_EMBEDDING_DIM  = 100
BILSTM_HIDDEN_DIM     = 128
BILSTM_NUM_LAYERS     = 2
BILSTM_DROPOUT        = 0.3
BILSTM_BIDIRECTIONAL  = True

BILSTM_BATCH_SIZE     = 64      # on augmente le batch — on a plus de données
BILSTM_EPOCHS         = 15
BILSTM_LEARNING_RATE  = 1e-3
BILSTM_PATIENCE       = 3


# =============================================================================
# ROBERTA
# =============================================================================

ROBERTA_MODEL_NAME    = 'roberta-base'
ROBERTA_MAX_LENGTH    = 256

ROBERTA_BATCH_SIZE    = 32
ROBERTA_EPOCHS        = 5
ROBERTA_LEARNING_RATE = 2e-5
ROBERTA_WARMUP_STEPS  = 200     # plus de warmup steps car on a plus de batches
ROBERTA_WEIGHT_DECAY  = 0.01
ROBERTA_PATIENCE      = 2


# =============================================================================
# ENTRAÎNEMENT
# =============================================================================

DEVICE            = 'cuda'   # changer en 'cpu' si pas de GPU
BEST_MODEL_METRIC = 'f1'


# =============================================================================
# APP STREAMLIT
# =============================================================================

APP_DEFAULT_MODEL = 'roberta'

LABEL_MAP = {
    0: '🔴 Fake News',
    1: '🟢 Real News',
}


# =============================================================================
# CRÉATION AUTOMATIQUE DES DOSSIERS
# -----------------------------------------------------------------------------
# S'exécute à chaque import de config.py — garantit que les dossiers existent.
# =============================================================================

for _dir in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(_dir, exist_ok=True)
