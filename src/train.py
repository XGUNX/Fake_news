# =============================================================================
# train.py
# -----------------------------------------------------------------------------
# Ce fichier gère l'entraînement du modèle BiLSTM.
#
# Objectif :
# Entraîner le BiLSTM sur les données prétraitées, évaluer ses performances,
# et sauvegarder les meilleurs poids.
#
# Étapes suivies dans ce fichier :
# . Charger les données et créer les DataLoaders
# . Initialiser le modèle, l'optimiseur et la fonction de perte
# . Boucle d'entraînement avec early stopping
# . Évaluation à chaque époque sur la validation
# . Sauvegarde du meilleur modèle
# . Évaluation finale sur le test
# . Génération des graphiques des courbes d'apprentissage
# =============================================================================

import os
import sys
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
from src.preprocess import load_processed_data
from src.dataset import create_dataloaders
from src.bilstm import BiLSTM


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Entraîne le modèle pendant une époque.

    Args:
        model: Modèle PyTorch
        dataloader: DataLoader pour les données d'entraînement
        criterion: Fonction de perte
        optimizer: Optimiseur
        device: 'cuda' ou 'cpu'

    Returns:
        tuple: (loss_moyenne, accuracy_moyenne)
    """
    model.train()  # Mode entraînement (active dropout, batch norm, etc.)
    total_loss = 0
    correct = 0
    total = 0

    for indices, labels in dataloader:
        # Déplacer les données sur le device (GPU ou CPU)
        indices = indices.to(device)
        labels = labels.to(device)

        # Réinitialiser les gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(indices)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping pour éviter l'explosion des gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Mettre à jour les poids
        optimizer.step()

        # Statistiques
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """
    Évalue le modèle sur un jeu de données (validation ou test).

    Args:
        model: Modèle PyTorch
        dataloader: DataLoader pour les données
        criterion: Fonction de perte
        device: 'cuda' ou 'cpu'

    Returns:
        tuple: (loss_moyenne, accuracy_moyenne)
    """
    model.eval()  # Mode évaluation (désactive dropout)
    total_loss = 0
    correct = 0
    total = 0

    # Désactiver le calcul des gradients pour économiser la mémoire
    with torch.no_grad():
        for indices, labels in dataloader:
            indices = indices.to(device)
            labels = labels.to(device)

            outputs = model(indices)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def get_predictions(model, dataloader, device):
    """
    Récupère toutes les prédictions et les labels réels.

    Args:
        model: Modèle PyTorch
        dataloader: DataLoader
        device: 'cuda' ou 'cpu'

    Returns:
        tuple: (all_predictions, all_labels, all_probabilities)
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for indices, labels in dataloader:
            indices = indices.to(device)
            labels = labels.to(device)

            outputs = model(indices)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)


def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """
    Trace les courbes d'apprentissage.

    Args:
        train_losses, val_losses: Listes des pertes
        train_accs, val_accs: Listes des accuracies
    """
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Courbes de perte
    ax1.plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation', linewidth=2)
    ax1.set_xlabel('Époque')
    ax1.set_ylabel('Perte')
    ax1.set_title('Courbes de perte')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Courbes d'accuracy
    ax2.plot(epochs, train_accs, 'b-', label='Train', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation', linewidth=2)
    ax2.set_xlabel('Époque')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Courbes d\'accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Courbes d\'apprentissage - BiLSTM', fontsize=14)
    plt.tight_layout()

    # Sauvegarder
    save_path = os.path.join(config.RESULTS_DIR, 'bilstm_training_curves.png')
    plt.savefig(save_path)
    plt.show()
    print(f"Courbes sauvegardées : {save_path}")


def save_bilstm_model(model, word2idx):
    """
    Sauvegarde le modèle BiLSTM et le vocabulaire.

    Args:
        model: Modèle PyTorch
        word2idx: Dictionnaire mot -> index
    """
    model_path = os.path.join(config.MODELS_DIR, 'bilstm_model.pth')
    vocab_path = os.path.join(config.MODELS_DIR, 'bilstm_word2idx.pkl')

    # Sauvegarder les poids du modèle
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': len(word2idx),
        'embedding_dim': config.BILSTM_EMBEDDING_DIM,
        'hidden_dim': config.BILSTM_HIDDEN_DIM,
        'num_layers': config.BILSTM_NUM_LAYERS,
        'dropout': config.BILSTM_DROPOUT
    }, model_path)

    # Sauvegarder le vocabulaire
    with open(vocab_path, 'wb') as f:
        pickle.dump(word2idx, f)

    print(f"Modèle BiLSTM sauvegardé : {model_path}")
    print(f"Vocabulaire sauvegardé : {vocab_path}")


def load_bilstm_model():
    """
    Recharge le modèle BiLSTM sauvegardé.

    Returns:
        tuple: (model, word2idx)
    """
    model_path = os.path.join(config.MODELS_DIR, 'bilstm_model.pth')
    vocab_path = os.path.join(config.MODELS_DIR, 'bilstm_word2idx.pkl')

    # Charger le vocabulaire
    with open(vocab_path, 'rb') as f:
        word2idx = pickle.load(f)

    # Reconstruire le modèle avec les mêmes paramètres
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model = BiLSTM(
        vocab_size=checkpoint['vocab_size'],
        embedding_dim=checkpoint['embedding_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers'],
        dropout=checkpoint['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, word2idx


def run_bilstm_training():
    """
    Pipeline complète d'entraînement du BiLSTM.
    """
    print("=== Entraînement du BiLSTM ===\n")

    # 1. Charger les données prétraitées
    print("1. Chargement des données...")
    splits, word2idx = load_processed_data()
    X_train, X_val, X_test = splits['X_train'], splits['X_val'], splits['X_test']
    y_train, y_val, y_test = splits['y_train'], splits['y_val'], splits['y_test']

    # 2. Créer les DataLoaders
    print("\n2. Création des DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        word2idx
    )

    # 3. Initialiser le modèle
    print("\n3. Initialisation du modèle...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device : {device}")

    vocab_size = len(word2idx)
    model = BiLSTM(vocab_size=vocab_size).to(device)

    print(f"  Vocabulaire : {vocab_size} mots")
    print(f"  Paramètres : {sum(p.numel() for p in model.parameters()):,}")

    # 4. Définir l'optimiseur et la fonction de perte
    criterion = nn.CrossEntropyLoss()  # Classification multi-classes
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.BILSTM_LEARNING_RATE,
        weight_decay=config.BILSTM_WEIGHT_DECAY
    )

    # Réduire le learning rate si la perte de validation stagne
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )

    # 5. Boucle d'entraînement
    print("\n4. Démarrage de l'entraînement...")
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config.BILSTM_EPOCHS):
        print(f"\n--- Époque {epoch + 1}/{config.BILSTM_EPOCHS} ---")

        # Entraînement
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Sauvegarder le meilleur modèle
            save_bilstm_model(model, word2idx)
            print(f"  -> Meilleur modèle sauvegardé!")
        else:
            patience_counter += 1
            if patience_counter >= config.BILSTM_PATIENCE:
                print(f"\nEarly stopping déclenché après {epoch + 1} époques")
                break

        # Ajuster le learning rate
        scheduler.step(val_loss)

    # 6. Tracer les courbes d'apprentissage
    print("\n5. Génération des courbes d'apprentissage...")
    plot_training_history(train_losses, val_losses, train_accs, val_accs)

    # 7. Évaluation sur le test
    print("\n6. Évaluation sur le test...")
    model, word2idx = load_bilstm_model()  # Charger le meilleur modèle
    model = model.to(device)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n--- Résultats sur le Test ---")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.2f}%")

    # Récupérer toutes les prédictions pour les métriques détaillées
    predictions, labels, probabilities = get_predictions(model, test_loader, device)

    # Afficher les métriques détaillées
    from sklearn.metrics import classification_report, confusion_matrix
    print(f"\nRapport de classification :")
    print(classification_report(labels, predictions, target_names=['Fake', 'Real']))

    # Matrice de confusion
    cm = confusion_matrix(labels, predictions)
    import seaborn as sns
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fake', 'Real'],
                yticklabels=['Fake', 'Real'])
    plt.title('Matrice de Confusion - BiLSTM')
    plt.ylabel('Label réel')
    plt.xlabel('Label prédit')
    plt.tight_layout()
    cm_path = os.path.join(config.RESULTS_DIR, 'confusion_matrix_bilstm.png')
    plt.savefig(cm_path)
    plt.show()

    print(f"\n=== Entraînement BiLSTM terminé ===")
    print(f"  Meilleure accuracy sur validation : {max(val_accs):.2f}%")
    print(f"  Accuracy sur test : {test_acc:.2f}%")

    return test_acc, test_loss


# Lancement direct : python src/train.py
if __name__ == '__main__':
    run_bilstm_training()