# =============================================================================
# bilstm.py
# -----------------------------------------------------------------------------
# Ce fichier définit l'architecture du modèle BiLSTM pour la classification
# de fake news.
#
# Objectif :
# Implémenter un réseau de neurones récurrent bidirectionnel (BiLSTM) qui
# lit les textes dans les deux sens pour capturer le contexte complet.
#
# Architecture :
# . Embedding : convertit chaque mot en vecteur dense
# . BiLSTM : deux LSTM qui lisent le texte dans les deux sens
# . Dropout : régularisation pour éviter le surapprentissage
# . Fully Connected : couche dense pour la classification finale
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config


class BiLSTM(nn.Module):
    """
    Modèle BiLSTM pour la classification binaire (Fake vs Real).

    Args:
        vocab_size (int): Taille du vocabulaire
        embedding_dim (int): Dimension des vecteurs de mots
        hidden_dim (int): Dimension de la couche cachée du LSTM
        num_layers (int): Nombre de couches LSTM empilées
        dropout (float): Taux de dropout
        pad_idx (int): Index du token <PAD>
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=config.BILSTM_EMBEDDING_DIM,
        hidden_dim=config.BILSTM_HIDDEN_DIM,
        num_layers=config.BILSTM_NUM_LAYERS,
        dropout=config.BILSTM_DROPOUT,
        pad_idx=0  # <PAD> est toujours à l'index 0
    ):
        super(BiLSTM, self).__init__()

        # Couche d'embedding
        # Chaque mot devient un vecteur de dimension embedding_dim
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx  # Les tokens <PAD> ne participent pas à l'apprentissage
        )

        # Dropout avant le LSTM
        self.dropout = nn.Dropout(dropout)

        # BiLSTM
        # batch_first=True : les tenseurs sont de forme (batch, sequence, feature)
        # bidirectional=True : le LSTM lit dans les deux sens
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Couche de sortie
        # hidden_dim * 2 car on concatène les sorties des deux directions
        # On a 2 classes (Fake=0, Real=1)
        self.fc = nn.Linear(hidden_dim * 2, 2)

        # Dropout final
        self.dropout_final = nn.Dropout(dropout)

    def forward(self, x):
        """
        Passage avant (forward pass).

        Args:
            x (torch.Tensor): Tenseur d'indices de forme (batch_size, seq_len)

        Returns:
            torch.Tensor: Logits de forme (batch_size, 2)
        """
        # Embedding : (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)

        # Dropout
        embedded = self.dropout(embedded)

        # LSTM : (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, hidden_dim * 2) pour la sortie
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # On prend le dernier état caché de chaque direction
        # hidden est de forme (num_layers * 2, batch_size, hidden_dim)
        # On concatène les deux dernières couches (forward et backward)
        hidden_last = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # hidden_last shape : (batch_size, hidden_dim * 2)

        # Dropout
        hidden_last = self.dropout_final(hidden_last)

        # Couche fully connected : (batch_size, hidden_dim * 2) -> (batch_size, 2)
        output = self.fc(hidden_last)

        return output

    def predict(self, x):
        """
        Prédit la classe pour un batch d'échantillons.

        Args:
            x (torch.Tensor): Tenseur d'indices de forme (batch_size, seq_len)

        Returns:
            tuple: (predictions, probabilities)
                - predictions: tenseur des classes prédites
                - probabilities: tenseur des probabilités (softmax)
        """
        logits = self.forward(x)
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        return predictions, probabilities


# Test rapide si le fichier est exécuté directement
if __name__ == '__main__':
    print("=== Test de bilstm.py ===\n")

    # Créer un petit modèle de test
    vocab_size = 1000
    model = BiLSTM(vocab_size=vocab_size)

    print(f"Architecture du modèle :")
    print(f"  Vocabulaire : {vocab_size} mots")
    print(f"  Embedding dim : {config.BILSTM_EMBEDDING_DIM}")
    print(f"  Hidden dim : {config.BILSTM_HIDDEN_DIM}")
    print(f"  Nombre de paramètres : {sum(p.numel() for p in model.parameters()):,}")

    # Tester un forward pass
    batch_size = 4
    seq_len = config.MAX_SEQUENCE_LENGTH
    test_input = torch.randint(0, vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        output = model(test_input)

    print(f"\nTest forward pass :")
    print(f"  Input shape  : {test_input.shape}")
    print(f"  Output shape : {output.shape}")
    print(f"  Output sample: {output[0]}")