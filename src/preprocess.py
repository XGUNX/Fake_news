# =============================================================================
# preprocess.py
# -----------------------------------------------------------------------------
# Tout ce qui touche au chargement et nettoyage des données.
# On a testé toutes ces fonctions dans 02_experiments.ipynb avant de les écrire ici.
# =============================================================================

import os
import re
import pickle

import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config


# -----------------------------------------------------------------------------
# Chargement
# -----------------------------------------------------------------------------

def load_data():
    # on charge les 4 fichiers CSV et on les fusionne en un seul dataframe
    frames = []

    for key, filename in config.RAW_FILES.items():
        path = os.path.join(config.DATA_RAW_DIR, filename)

        if not os.path.exists(path):
            print(f'[ATTENTION] Fichier introuvable : {path}')
            continue

        df = pd.read_csv(path)

        # le label et la source viennent du nom du fichier, pas des colonnes
        df['label']  = config.FILE_LABELS[key]
        df['source'] = config.FILE_SOURCES[key]

        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)

    # on garde seulement les colonnes dont on a besoin
    df_all = df_all[['title', 'text', 'label', 'source']]

    print(f'Données chargées : {len(df_all)} articles')
    print(f'  Fake : {(df_all["label"] == 0).sum()}')
    print(f'  Real : {(df_all["label"] == 1).sum()}')

    return df_all


# -----------------------------------------------------------------------------
# Nettoyage
# -----------------------------------------------------------------------------

def clean_text(text):
    # nettoie un texte brut : minuscules, URLs, caractères spéciaux, espaces
    if pd.isna(text):
        return ''

    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)       # suppression des URLs
    text = re.sub(r'\S+@\S+', '', text)               # suppression des emails
    text = re.sub(r'[^a-z0-9\s]', ' ', text)          # on garde lettres et chiffres
    text = re.sub(r'\s+', ' ', text).strip()           # espaces multiples

    return text


def clean_dataframe(df):
    # applique le nettoyage sur les colonnes title et text
    df = df.copy()
    df['title_clean'] = df['title'].apply(clean_text)
    df['text_clean']  = df['text'].apply(clean_text)

    # combinaison titre + corps avec le séparateur [SEP]
    df['combined'] = df.apply(
        lambda row: config.TEXT_COMBINATION_TEMPLATE.format(
            title=row['title_clean'],
            text=row['text_clean']
        ),
        axis=1
    )

    print('Nettoyage terminé')
    return df


# -----------------------------------------------------------------------------
# Split
# -----------------------------------------------------------------------------

def split_data(df):
    # découpe le dataframe en train / val / test (80 / 10 / 10)
    # on utilise stratify pour garder l'équilibre des classes dans chaque split

    X = df['combined'].values
    y = df['label'].values

    # premier split : train vs reste
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(1 - config.TRAIN_SIZE),
        random_state=config.RANDOM_SEED,
        stratify=y
    )

    # deuxième split : val vs test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=config.RANDOM_SEED,
        stratify=y_temp
    )

    print(f'Train : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}')

    return X_train, X_val, X_test, y_train, y_val, y_test


# -----------------------------------------------------------------------------
# Vocabulaire pour le BiLSTM
# -----------------------------------------------------------------------------

def build_vocab(X_train):
    # construit le vocabulaire uniquement depuis le train set
    # on ne regarde jamais val ou test pour éviter le data leakage

    word_counts = Counter()
    for text in X_train:
        word_counts.update(text.split())

    # tokens spéciaux en premier
    vocab = ['<PAD>', '<UNK>']

    for word, count in word_counts.items():
        if count >= config.MIN_WORD_FREQ:
            vocab.append(word)

    # on tronque si on dépasse la taille max
    vocab = vocab[:config.MAX_VOCAB_SIZE]

    # dictionnaire mot → index
    word2idx = {word: idx for idx, word in enumerate(vocab)}

    print(f'Vocabulaire construit : {len(word2idx)} mots')
    return word2idx


def text_to_indices(text, word2idx):
    # convertit un texte en liste d'indices de longueur fixe (MAX_SEQUENCE_LENGTH)
    tokens  = text.split()[:config.MAX_SEQUENCE_LENGTH]
    indices = [word2idx.get(token, word2idx['<UNK>']) for token in tokens]

    # padding si le texte est plus court que MAX_SEQUENCE_LENGTH
    pad_len = config.MAX_SEQUENCE_LENGTH - len(indices)
    indices += [word2idx['<PAD>']] * pad_len

    return indices


# -----------------------------------------------------------------------------
# Sauvegarde
# -----------------------------------------------------------------------------

def save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, word2idx):
    # sauvegarde les splits et le vocabulaire dans data/processed/

    splits = {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
    }

    for name, data in splits.items():
        path = os.path.join(config.DATA_PROCESSED_DIR, f'{name}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    # sauvegarde du vocabulaire séparément (utilisé par le BiLSTM)
    vocab_path = os.path.join(config.DATA_PROCESSED_DIR, 'word2idx.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump(word2idx, f)

    print(f'Données sauvegardées dans : {config.DATA_PROCESSED_DIR}')


def load_processed_data():
    # recharge les données déjà traitées pour éviter de tout recalculer
    splits = {}
    names  = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']

    for name in names:
        path = os.path.join(config.DATA_PROCESSED_DIR, f'{name}.pkl')
        with open(path, 'rb') as f:
            splits[name] = pickle.load(f)

    vocab_path = os.path.join(config.DATA_PROCESSED_DIR, 'word2idx.pkl')
    with open(vocab_path, 'rb') as f:
        word2idx = pickle.load(f)

    print('Données traitées rechargées')
    return splits, word2idx


# -----------------------------------------------------------------------------
# Pipeline complète
# -----------------------------------------------------------------------------

def run_preprocessing():
    # fonction principale — exécute tout dans l'ordre
    print('=== Démarrage du preprocessing ===\n')

    df        = load_data()
    df        = clean_dataframe(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    word2idx  = build_vocab(X_train)

    save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, word2idx)

    print('\n=== Preprocessing terminé ===')
    return X_train, X_val, X_test, y_train, y_val, y_test, word2idx


# si on lance ce fichier directement : python src/preprocess.py
if __name__ == '__main__':
    run_preprocessing()
