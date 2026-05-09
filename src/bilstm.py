# =============================================================================
# bilstm.py
# -----------------------------------------------------------------------------
# Architecture du modèle BiLSTM pour classifier les articles en fake ou real.
#
# Pourquoi un BiLSTM ?
# Un LSTM classique lit le texte de gauche à droite.
# Un BiLSTM le lit dans les deux sens (gauche→droite ET droite→gauche)
# et concatène les deux résultats — il capture mieux le contexte d'un mot.
#
# Structure du modèle :
#   1. Embedding   : transforme chaque indice de mot en vecteur dense
#   2. BiLSTM      : lit la séquence dans les deux sens
#   3. Dropout     : désactive des neurones aléatoirement pour éviter l'overfitting
#   4. Linear      : couche finale qui sort 2 scores (fake / real)
# =============================================================================

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config


class BiLSTM(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=config.BILSTM_EMBEDDING_DIM,
        hidden_dim=config.BILSTM_HIDDEN_DIM,
        num_layers=config.BILSTM_NUM_LAYERS,
        dropout=config.BILSTM_DROPOUT,
        pad_idx=0   # <PAD> est toujours à l'index 0 dans notre vocabulaire
    ):
        super(BiLSTM, self).__init__()

        # --- Couche Embedding ---
        # convertit chaque indice de mot en vecteur de taille embedding_dim
        # padding_idx=pad_idx : les tokens <PAD> ne contribuent pas au gradient
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx
        )

        # dropout appliqué après l'embedding pour régulariser dès le début
        self.dropout = nn.Dropout(dropout)

        # --- Couche BiLSTM ---
        # batch_first=True : les tenseurs ont la forme (batch, séquence, features)
        # bidirectional=True : deux LSTM en parallèle, un dans chaque sens
        # dropout entre les couches LSTM (seulement si num_layers > 1)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # --- Couche de classification finale ---
        # hidden_dim * 2 car on concatène les sorties forward et backward du BiLSTM
        # la sortie finale a 2 neurones : un score pour fake, un score pour real
        self.fc = nn.Linear(hidden_dim * 2, 2)

        # dropout final avant la classification
        self.dropout_final = nn.Dropout(dropout)

    def forward(self, x):
        # x : tenseur d'indices de mots, forme (batch_size, seq_len)

        # 1. embedding : (batch_size, seq_len) → (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(self.embedding(x))

        # 2. BiLSTM
        # lstm_out : sortie à chaque pas de temps, forme (batch_size, seq_len, hidden_dim * 2)
        # hidden   : état caché final, forme (num_layers * 2, batch_size, hidden_dim)
        #            *2 car bidirectionnel
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # on récupère le dernier état caché des deux directions
        # hidden[-2] = dernière couche, sens forward
        # hidden[-1] = dernière couche, sens backward
        # on les concatène → (batch_size, hidden_dim * 2)
        hidden_last = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        # 3. dropout + classification
        # (batch_size, hidden_dim * 2) → (batch_size, 2)
        output = self.fc(self.dropout_final(hidden_last))

        return output

    def predict(self, x):
        # retourne la classe prédite et la probabilité associée
        # utilisé pendant l'évaluation et dans l'app Streamlit

        logits        = self.forward(x)
        probabilities = F.softmax(logits, dim=1)     # convertit les scores en probabilités
        predictions   = torch.argmax(probabilities, dim=1)  # prend la classe avec la proba la plus haute

        return predictions, probabilities


# -----------------------------------------------------------------------------
# Test rapide
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    print('=== Test du BiLSTM ===\n')

    # on crée un petit modèle avec un vocabulaire fictif pour vérifier les shapes
    vocab_size = 1000
    model      = BiLSTM(vocab_size=vocab_size)

    nb_params = sum(p.numel() for p in model.parameters())
    print(f'Taille du vocabulaire   : {vocab_size}')
    print(f'Dimension embedding     : {config.BILSTM_EMBEDDING_DIM}')
    print(f'Dimension cachée (LSTM) : {config.BILSTM_HIDDEN_DIM}')
    print(f'Nombre de paramètres    : {nb_params:,}')

    # on simule un batch de 4 articles de longueur MAX_SEQUENCE_LENGTH
    batch_size = 4
    seq_len    = config.MAX_SEQUENCE_LENGTH
    faux_input = torch.randint(0, vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        sortie = model(faux_input)

    print(f'\nTest forward pass :')
    print(f'  Input shape  : {faux_input.shape}')   # (4, 256)
    print(f'  Output shape : {sortie.shape}')        # (4, 2) — 2 scores par article
    print('\nTest OK — les shapes sont corrects')
