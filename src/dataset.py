# =============================================================================
# dataset.py
# -----------------------------------------------------------------------------
# Ce fichier contient les classes Dataset pour PyTorch.
# Un Dataset c'est juste un objet qui sait comment donner un exemple à la fois
# au modèle pendant l'entraînement.
#
# On a besoin de DEUX datasets différents :
#   - BiLSTMDataset  : retourne des indices de mots (vocabulaire qu'on a construit)
#   - RobertaDataset : retourne des tokens RoBERTa (tokenizer HuggingFace)
#
# PyTorch attend que tout Dataset implémente trois méthodes :
#   - __init__  : initialisation, on stocke les données
#   - __len__   : retourne le nombre d'exemples dans le dataset
#   - __getitem__: retourne un seul exemple (le DataLoader l'appelle en boucle)
# =============================================================================

import os
import sys

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
from src.preprocess import text_to_indices


# -----------------------------------------------------------------------------
# Dataset pour le BiLSTM
# -----------------------------------------------------------------------------

class BiLSTMDataset(Dataset):
    # Dataset qui convertit les textes en séquences d'indices pour le BiLSTM
    # Le BiLSTM a besoin d'indices entiers qui pointent vers notre vocabulaire

    def __init__(self, texts, labels, word2idx):
        # texts   : liste de textes nettoyés (strings)
        # labels  : liste de labels (0 ou 1)
        # word2idx: dictionnaire mot → index construit dans preprocess.py

        self.texts   = texts
        self.labels  = labels
        self.word2idx = word2idx

    def __len__(self):
        # PyTorch appelle ça pour savoir combien d'exemples il y a dans le dataset
        return len(self.texts)

    def __getitem__(self, idx):
        # PyTorch appelle ça à chaque fois qu'il veut un exemple précis
        # idx : l'indice de l'exemple qu'il veut (entre 0 et len-1)

        text  = self.texts[idx]
        label = self.labels[idx]

        # on convertit le texte en liste d'indices de longueur fixe
        # les mots inconnus deviennent <UNK>, le padding remplit le reste
        indices = text_to_indices(text, self.word2idx)

        # on retourne des tenseurs PyTorch, pas des listes Python
        # LongTensor pour les indices (entiers 64 bits — requis par nn.Embedding)
        return {
            'input_ids': torch.LongTensor(indices),
            'label'    : torch.LongTensor([label])[0]  # scalaire, pas un tableau
        }


# -----------------------------------------------------------------------------
# Dataset pour RoBERTa
# -----------------------------------------------------------------------------

class RobertaDataset(Dataset):
    # Dataset qui tokenise les textes avec le tokenizer HuggingFace de RoBERTa
    # RoBERTa a son propre vocabulaire de 50 000 tokens — on n'utilise pas le nôtre

    def __init__(self, texts, labels, tokenizer):
        # texts    : liste de textes nettoyés (strings)
        # labels   : liste de labels (0 ou 1)
        # tokenizer: instance de RobertaTokenizer déjà chargée

        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text  = self.texts[idx]
        label = self.labels[idx]

        # le tokenizer de HuggingFace fait tout en une ligne :
        #   - découpage en tokens
        #   - conversion en indices
        #   - padding jusqu'à max_length
        #   - truncation si le texte est trop long
        #   - attention_mask : 1 pour les vrais tokens, 0 pour le padding
        encoded = self.tokenizer(
            text,
            max_length=config.ROBERTA_MAX_LENGTH,
            padding='max_length',      # complète avec des 0 si trop court
            truncation=True,           # coupe si trop long
            return_tensors='pt'        # retourne directement des tenseurs PyTorch
        )

        # .squeeze(0) : enlève la dimension batch ajoutée par return_tensors='pt'
        # sans ça on aurait shape (1, 256) au lieu de (256,)
        return {
            'input_ids'     : encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label'         : torch.tensor(label, dtype=torch.long)
        }


# -----------------------------------------------------------------------------
# Fonctions utilitaires pour créer les DataLoaders
# -----------------------------------------------------------------------------

def get_bilstm_loaders(X_train, X_val, X_test, y_train, y_val, y_test, word2idx):
    # crée les trois DataLoaders (train / val / test) pour le BiLSTM
    #
    # Un DataLoader c'est un itérateur qui :
    #   - regroupe les exemples en batches
    #   - mélange les données à chaque epoch (shuffle=True pour le train)
    #   - gère le chargement en parallèle (num_workers)

    train_dataset = BiLSTMDataset(X_train, y_train, word2idx)
    val_dataset   = BiLSTMDataset(X_val,   y_val,   word2idx)
    test_dataset  = BiLSTMDataset(X_test,  y_test,  word2idx)

    # shuffle=True seulement pour le train
    # si on mélange val/test les résultats restent les mêmes mais c'est une bonne habitude
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BILSTM_BATCH_SIZE,
        shuffle=True   # on mélange le train à chaque epoch pour éviter l'overfitting
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BILSTM_BATCH_SIZE,
        shuffle=False  # pas besoin de mélanger pour l'évaluation
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BILSTM_BATCH_SIZE,
        shuffle=False
    )

    print(f'DataLoaders BiLSTM créés')
    print(f'  Train : {len(train_dataset)} exemples, {len(train_loader)} batches')
    print(f'  Val   : {len(val_dataset)}   exemples, {len(val_loader)} batches')
    print(f'  Test  : {len(test_dataset)}  exemples, {len(test_loader)} batches')

    return train_loader, val_loader, test_loader


def get_roberta_loaders(X_train, X_val, X_test, y_train, y_val, y_test):
    # crée les trois DataLoaders pour RoBERTa
    # le tokenizer est chargé ici une seule fois et partagé entre les trois datasets

    print(f'Chargement du tokenizer {config.ROBERTA_MODEL_NAME}...')
    tokenizer = RobertaTokenizer.from_pretrained(config.ROBERTA_MODEL_NAME)

    train_dataset = RobertaDataset(X_train, y_train, tokenizer)
    val_dataset   = RobertaDataset(X_val,   y_val,   tokenizer)
    test_dataset  = RobertaDataset(X_test,  y_test,  tokenizer)

    # batch_size plus petit pour RoBERTa car le modèle est beaucoup plus lourd
    # si vous avez des erreurs de mémoire GPU, réduisez encore cette valeur dans config.py
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.ROBERTA_BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.ROBERTA_BATCH_SIZE,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.ROBERTA_BATCH_SIZE,
        shuffle=False
    )

    print(f'DataLoaders RoBERTa créés')
    print(f'  Train : {len(train_dataset)} exemples, {len(train_loader)} batches')
    print(f'  Val   : {len(val_dataset)}   exemples, {len(val_loader)} batches')
    print(f'  Test  : {len(test_dataset)}  exemples, {len(test_loader)} batches')

    return train_loader, val_loader, test_loader, tokenizer


# -----------------------------------------------------------------------------
# Test rapide
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    print('=== Test de dataset.py ===\n')

    from src.preprocess import load_processed_data

    splits, word2idx = load_processed_data()

    # test BiLSTM
    print('-- BiLSTM Dataset --')
    train_loader, val_loader, test_loader = get_bilstm_loaders(
        splits['X_train'], splits['X_val'], splits['X_test'],
        splits['y_train'], splits['y_val'], splits['y_test'],
        word2idx
    )

    # on regarde le premier batch pour vérifier les shapes
    batch = next(iter(train_loader))
    print(f'\nPremier batch BiLSTM :')
    print(f'  input_ids shape : {batch["input_ids"].shape}')   # (batch_size, seq_len)
    print(f'  label shape     : {batch["label"].shape}')        # (batch_size,)

    # test RoBERTa
    print('\n-- RoBERTa Dataset --')
    train_loader_r, _, _, _ = get_roberta_loaders(
        splits['X_train'], splits['X_val'], splits['X_test'],
        splits['y_train'], splits['y_val'], splits['y_test']
    )

    batch_r = next(iter(train_loader_r))
    print(f'\nPremier batch RoBERTa :')
    print(f'  input_ids shape      : {batch_r["input_ids"].shape}')        # (batch_size, max_length)
    print(f'  attention_mask shape : {batch_r["attention_mask"].shape}')   # (batch_size, max_length)
    print(f'  label shape          : {batch_r["label"].shape}')            # (batch_size,)

    print('\nTest OK')
