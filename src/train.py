# =============================================================================
# train.py
# -----------------------------------------------------------------------------
# Boucle d'entraînement pour le BiLSTM et RoBERTa.
#
# On écrit la boucle à la main en PyTorch plutôt que d'utiliser le Trainer
# de HuggingFace — c'est plus long mais beaucoup plus clair à expliquer
# et on contrôle exactement ce qui se passe à chaque étape.
#
# Une epoch d'entraînement ça ressemble à ça :
#   1. On prend un batch de textes
#   2. On fait passer le batch dans le modèle (forward pass)
#   3. On calcule l'erreur (loss)
#   4. On calcule les gradients (backward pass)
#   5. On met à jour les poids du modèle (optimizer.step)
#   6. On répète pour tous les batches
#   7. On évalue sur le val set à la fin de chaque epoch
#   8. On sauvegarde le meilleur modèle (early stopping)
# =============================================================================

import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import RobertaForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config


# -----------------------------------------------------------------------------
# Fonctions utilitaires communes
# -----------------------------------------------------------------------------

def get_device():
    # vérifie si un GPU est disponible, sinon on utilise le CPU
    # sur GPU l'entraînement est 10x à 50x plus rapide
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f'Device utilisé : {device}')
    return device


def compute_metrics(all_labels, all_preds):
    # calcule accuracy et F1 depuis deux listes de labels
    # on utilise F1 comme métrique principale car c'est plus robuste que l'accuracy
    # sur des problèmes de classification binaire

    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    f1       = f1_score(all_labels, all_preds, average='binary')

    return accuracy, f1


# -----------------------------------------------------------------------------
# Entraînement du BiLSTM
# -----------------------------------------------------------------------------

def train_one_epoch_bilstm(model, loader, optimizer, criterion, device):
    # entraîne le modèle sur tous les batches du train loader
    # retourne la loss moyenne et le F1 sur tout le train set

    model.train()   # mode entraînement : active le dropout
    total_loss = 0
    all_preds  = []
    all_labels = []

    for batch in loader:
        # on transfère les données sur le bon device (GPU ou CPU)
        input_ids = batch['input_ids'].to(device)
        labels    = batch['label'].to(device)

        # --- Forward pass ---
        # le modèle prédit des scores (logits) pour chaque classe
        optimizer.zero_grad()          # on remet les gradients à zéro avant chaque batch
        logits = model(input_ids)      # forme : (batch_size, 2)

        # --- Calcul de la loss ---
        # CrossEntropyLoss est standard pour la classification multi-classe
        # elle combine softmax + negative log likelihood
        loss = criterion(logits, labels)

        # --- Backward pass ---
        # calcule les gradients de la loss par rapport à chaque paramètre
        loss.backward()

        # gradient clipping : évite les gradients qui explosent (problème fréquent avec les LSTM)
        # on plafonne la norme des gradients à 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # --- Mise à jour des poids ---
        optimizer.step()

        # on accumule la loss et les prédictions pour calculer les métriques à la fin
        total_loss += loss.item()
        preds       = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss         = total_loss / len(loader)
    accuracy, f1     = compute_metrics(all_labels, all_preds)

    return avg_loss, accuracy, f1


def evaluate_bilstm(model, loader, criterion, device):
    # évalue le modèle sur val ou test — pas de mise à jour des poids ici

    model.eval()    # mode évaluation : désactive le dropout
    total_loss = 0
    all_preds  = []
    all_labels = []

    # torch.no_grad() : on n'a pas besoin de calculer les gradients en évaluation
    # ça économise de la mémoire et accélère le calcul
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            labels    = batch['label'].to(device)

            logits = model(input_ids)
            loss   = criterion(logits, labels)

            total_loss += loss.item()
            preds       = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss     = total_loss / len(loader)
    accuracy, f1 = compute_metrics(all_labels, all_preds)

    return avg_loss, accuracy, f1


def train_bilstm(model, train_loader, val_loader):
    # pipeline complète d'entraînement du BiLSTM avec early stopping

    print('=== Entraînement du BiLSTM ===\n')

    device    = get_device()
    model     = model.to(device)

    # CrossEntropyLoss : loss standard pour la classification
    criterion = nn.CrossEntropyLoss()

    # Adam : optimiseur adaptatif, très utilisé en deep learning
    optimizer = Adam(model.parameters(), lr=config.BILSTM_LEARNING_RATE)

    # ReduceLROnPlateau : réduit le learning rate si la val loss ne s'améliore plus
    # ça aide le modèle à converger vers un meilleur minimum
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

    # variables pour l'early stopping
    best_val_f1    = 0.0
    epochs_no_improve = 0
    best_model_path = os.path.join(config.MODELS_DIR, 'bilstm_best.pt')

    for epoch in range(1, config.BILSTM_EPOCHS + 1):
        start = time.time()

        # --- Phase train ---
        train_loss, train_acc, train_f1 = train_one_epoch_bilstm(
            model, train_loader, optimizer, criterion, device
        )

        # --- Phase validation ---
        val_loss, val_acc, val_f1 = evaluate_bilstm(
            model, val_loader, criterion, device
        )

        # on donne la val_loss au scheduler pour qu'il ajuste le lr si besoin
        scheduler.step(val_loss)

        elapsed = time.time() - start

        print(
            f'Epoch {epoch:02d}/{config.BILSTM_EPOCHS} | '
            f'Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | '
            f'Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | '
            f'Temps: {elapsed:.1f}s'
        )

        # --- Early stopping et sauvegarde du meilleur modèle ---
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0

            # on sauvegarde seulement les poids, pas le modèle entier
            # c'est plus flexible car on peut recharger dans n'importe quelle architecture
            torch.save(model.state_dict(), best_model_path)
            print(f'  → Meilleur modèle sauvegardé (Val F1 = {best_val_f1:.4f})')
        else:
            epochs_no_improve += 1
            print(f'  → Pas d\'amélioration ({epochs_no_improve}/{config.BILSTM_PATIENCE})')

            # on arrête si le modèle ne s'améliore plus depuis PATIENCE epochs
            if epochs_no_improve >= config.BILSTM_PATIENCE:
                print(f'\nEarly stopping déclenché à l\'epoch {epoch}')
                break

    print(f'\nMeilleur Val F1 : {best_val_f1:.4f}')
    print(f'Poids sauvegardés : {best_model_path}')

    return best_model_path


# -----------------------------------------------------------------------------
# Entraînement de RoBERTa
# -----------------------------------------------------------------------------

def train_one_epoch_roberta(model, loader, optimizer, scheduler, device):
    # même logique que pour le BiLSTM mais RoBERTa prend aussi l'attention_mask en entrée
    # l'attention_mask dit au modèle quels tokens sont du vrai contenu et quels sont du padding

    model.train()
    total_loss = 0
    all_preds  = []
    all_labels = []

    for batch in loader:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        optimizer.zero_grad()

        # RobertaForSequenceClassification retourne un objet avec .loss et .logits
        # on lui passe les labels directement et il calcule la loss tout seul
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss   = outputs.loss
        logits = outputs.logits

        loss.backward()

        # gradient clipping — encore plus important pour les transformers
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # le scheduler de warmup avance d'un pas à chaque batch (pas à chaque epoch)
        scheduler.step()

        total_loss += loss.item()
        preds       = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss     = total_loss / len(loader)
    accuracy, f1 = compute_metrics(all_labels, all_preds)

    return avg_loss, accuracy, f1


def evaluate_roberta(model, loader, device):
    # évaluation de RoBERTa — même logique que BiLSTM
    # RoBERTa calcule sa propre loss si on lui passe les labels

    model.eval()
    total_loss = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()
            preds       = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss     = total_loss / len(loader)
    accuracy, f1 = compute_metrics(all_labels, all_preds)

    return avg_loss, accuracy, f1


def train_roberta(train_loader, val_loader):
    # pipeline complète d'entraînement de RoBERTa avec early stopping

    print('=== Fine-tuning RoBERTa ===\n')

    device = get_device()

    # on charge RoBERTa avec une tête de classification binaire (num_labels=2)
    # les poids pré-entraînés sont téléchargés automatiquement la première fois
    print(f'Chargement de {config.ROBERTA_MODEL_NAME}...')
    model = RobertaForSequenceClassification.from_pretrained(
        config.ROBERTA_MODEL_NAME,
        num_labels=2   # fake ou real
    )
    model = model.to(device)

    # nombre total de steps d'entraînement (utile pour le scheduler)
    total_steps = len(train_loader) * config.ROBERTA_EPOCHS

    # AdamW : variante d'Adam avec weight decay — recommandée pour les transformers
    # weight_decay : régularisation L2 sur les poids (évite l'overfitting)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.ROBERTA_LEARNING_RATE,
        weight_decay=config.ROBERTA_WEIGHT_DECAY
    )

    # scheduler avec warmup linéaire :
    # le lr monte progressivement au début (warmup) puis redescend
    # c'est une pratique standard pour le fine-tuning de transformers
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.ROBERTA_WARMUP_STEPS,
        num_training_steps=total_steps
    )

    # early stopping
    best_val_f1       = 0.0
    epochs_no_improve = 0
    best_model_path   = os.path.join(config.MODELS_DIR, 'roberta_best.pt')

    for epoch in range(1, config.ROBERTA_EPOCHS + 1):
        start = time.time()

        # --- Phase train ---
        train_loss, train_acc, train_f1 = train_one_epoch_roberta(
            model, train_loader, optimizer, scheduler, device
        )

        # --- Phase validation ---
        val_loss, val_acc, val_f1 = evaluate_roberta(
            model, val_loader, device
        )

        elapsed = time.time() - start

        print(
            f'Epoch {epoch:02d}/{config.ROBERTA_EPOCHS} | '
            f'Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | '
            f'Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | '
            f'Temps: {elapsed:.1f}s'
        )

        # sauvegarde du meilleur modèle + early stopping
        if val_f1 > best_val_f1:
            best_val_f1       = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f'  → Meilleur modèle sauvegardé (Val F1 = {best_val_f1:.4f})')
        else:
            epochs_no_improve += 1
            print(f'  → Pas d\'amélioration ({epochs_no_improve}/{config.ROBERTA_PATIENCE})')

            if epochs_no_improve >= config.ROBERTA_PATIENCE:
                print(f'\nEarly stopping déclenché à l\'epoch {epoch}')
                break

    print(f'\nMeilleur Val F1 : {best_val_f1:.4f}')
    print(f'Poids sauvegardés : {best_model_path}')

    return model, best_model_path
