# =============================================================================
# dataset.py
# -----------------------------------------------------------------------------
# Ce fichier définit la classe PyTorch Dataset pour le modèle BiLSTM.
#
# Objectif :
# Préparer les données textuelles pour l'entraînement du BiLSTM en convertissant
# les textes en tenseurs d'indices et en gérant le batching.
#
# Étapes suivies dans ce fichier :
# . Créer une classe TextDataset qui hérite de torch.utils.data.Dataset
# . Convertir les textes en indices lors du chargement
# . Retourner les paires (indices, label) pour chaque échantillon
# . Permettre le chargement efficace avec un DataLoader
# =============================================================================

import torch
from torch.utils.data import Dataset

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
from src.preprocess import text_to_indices


class TextDataset(Dataset):
    """
    Dataset PyTorch pour les textes du projet.

    Args:
        texts (list): Liste des textes combinés (titre + corps)
        labels (list): Liste des labels (0 pour fake, 1 pour real)
        word2idx (dict): Dictionnaire mot -> index
    """

    def __init__(self, texts, labels, word2idx):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx

    def __len__(self):
        """Retourne le nombre total d'échantillons."""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retourne un échantillon (indices, label).

        Args:
            idx (int): Index de l'échantillon

        Returns:
            tuple: (indices_tensor, label_tensor)
                - indices_tensor: tenseur d'indices de longueur MAX_SEQUENCE_LENGTH
                - label_tensor: tenseur du label (0 ou 1)
        """
        # Récupérer le texte et le label
        text = self.texts[idx]
        label = self.labels[idx]

        # Convertir le texte en indices (déjà gère le padding et la troncature)
        indices = text_to_indices(text, self.word2idx)

        # Convertir en tenseurs PyTorch
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return indices_tensor, label_tensor


def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, word2idx):
    """
    Crée les DataLoaders pour l'entraînement, la validation et le test.

    Args:
        X_train, X_val, X_test: Listes de textes
        y_train, y_val, y_test: Listes de labels
        word2idx (dict): Dictionnaire mot -> index

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Créer les datasets
    train_dataset = TextDataset(X_train, y_train, word2idx)
    val_dataset = TextDataset(X_val, y_val, word2idx)
    test_dataset = TextDataset(X_test, y_test, word2idx)

    # Créer les dataloaders
    # shuffle=True pour le train (mélange les échantillons à chaque époque)
    # shuffle=False pour val et test (on garde l'ordre pour l'évaluation)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BILSTM_BATCH_SIZE,
        shuffle=True,
        num_workers=0  # mettre 0 pour éviter les problèmes sur Windows
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.BILSTM_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BILSTM_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(f"DataLoaders créés :")
    print(f"  Train : {len(train_dataset)} échantillons - {len(train_loader)} batches")
    print(f"  Val   : {len(val_dataset)} échantillons - {len(val_loader)} batches")
    print(f"  Test  : {len(test_dataset)} échantillons - {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


# Test rapide si le fichier est exécuté directement
if __name__ == '__main__':
    print("=== Test de dataset.py ===\n")

    # Importer preprocess pour charger les données
    from src.preprocess import load_processed_data

    # Charger les données
    splits, word2idx = load_processed_data()

    # Créer les dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        splits['X_train'], splits['X_val'], splits['X_test'],
        splits['y_train'], splits['y_val'], splits['y_test'],
        word2idx
    )

    # Tester un batch
    print("\nTest d'un batch :")
    indices, labels = next(iter(train_loader))
    print(f"  Shape des indices : {indices.shape}")  # (batch_size, MAX_SEQUENCE_LENGTH)
    print(f"  Shape des labels  : {labels.shape}")
    print(f"  Labels du batch   : {labels.tolist()}")