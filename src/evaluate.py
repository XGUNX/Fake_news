# =============================================================================
# evaluate.py
# -----------------------------------------------------------------------------
# Ce fichier s'occupe de l'évaluation finale des trois modèles :
#   - Baseline (TF-IDF + LR)
#   - BiLSTM
#   - RoBERTa
#
# On calcule les mêmes métriques pour chacun pour pouvoir les comparer
# objectivement. À la fin on génère un graphique de comparaison.
#
# Les métriques qu'on utilise :
#   - Accuracy  : proportion d'articles bien classifiés (simple mais peut tromper)
#   - Precision : parmi les articles qu'on a dit "fake", combien le sont vraiment ?
#   - Recall    : parmi les vrais "fake", combien on en a trouvé ?
#   - F1 Score  : moyenne harmonique de precision et recall — la plus fiable des trois
# =============================================================================

import os
import sys
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from transformers import RobertaForSequenceClassification

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
from src.preprocess import load_processed_data
from src.bilstm import BiLSTM
from src.dataset import get_bilstm_loaders, get_roberta_loaders


# -----------------------------------------------------------------------------
# Évaluation sur le test set
# -----------------------------------------------------------------------------

def eval_baseline_on_test(X_test, y_test):
    # charge le modèle baseline sauvegardé et l'évalue sur le test set

    print('--- Évaluation Baseline ---')

    # chargement du modèle et du vectorizer sauvegardés par baseline.py
    model_path = os.path.join(config.MODELS_DIR, 'baseline_model.pkl')
    vec_path   = os.path.join(config.MODELS_DIR, 'baseline_vectorizer.pkl')

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vec_path, 'rb') as f:
        vectorizer = pickle.load(f)

    # vectorisation du test set avec le vectorizer entraîné
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred       = model.predict(X_test_tfidf)

    metrics = compute_and_print_metrics(y_test, y_pred, model_name='Baseline')
    return y_pred, metrics


def eval_bilstm_on_test(test_loader, word2idx):
    # charge le meilleur BiLSTM sauvegardé et l'évalue sur le test set

    print('--- Évaluation BiLSTM ---')

    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')

    # on reconstruit l'architecture exacte avant de charger les poids
    # PyTorch sauvegarde les poids, pas l'architecture — il faut la redéfinir
    model = BiLSTM(vocab_size=len(word2idx))
    weights_path = os.path.join(config.MODELS_DIR, 'bilstm_best.pt')
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            labels    = batch['label'].to(device)

            logits = model(input_ids)
            preds  = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    metrics = compute_and_print_metrics(all_labels, all_preds, model_name='BiLSTM')
    return np.array(all_preds), metrics


def eval_roberta_on_test(test_loader):
    # charge le meilleur RoBERTa sauvegardé et l'évalue sur le test set

    print('--- Évaluation RoBERTa ---')

    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')

    # rechargement de l'architecture + poids fine-tunés
    model = RobertaForSequenceClassification.from_pretrained(
        config.ROBERTA_MODEL_NAME,
        num_labels=2
    )
    weights_path = os.path.join(config.MODELS_DIR, 'roberta_best.pt')
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    metrics = compute_and_print_metrics(all_labels, all_preds, model_name='RoBERTa')
    return np.array(all_preds), metrics


# -----------------------------------------------------------------------------
# Calcul des métriques
# -----------------------------------------------------------------------------

def compute_and_print_metrics(y_true, y_pred, model_name):
    # calcule et affiche toutes les métriques pour un modèle donné
    # retourne un dictionnaire pour pouvoir les comparer ensuite

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    print(f'\nRésultats {model_name} sur le test set :')
    print(f'  Accuracy  : {acc:.4f}')
    print(f'  Precision : {prec:.4f}')
    print(f'  Recall    : {rec:.4f}')
    print(f'  F1 Score  : {f1:.4f}')
    print()
    print(classification_report(y_true, y_pred, target_names=['Fake', 'Real']))

    return {
        'model'    : model_name,
        'accuracy' : acc,
        'precision': prec,
        'recall'   : rec,
        'f1'       : f1
    }


# -----------------------------------------------------------------------------
# Visualisations
# -----------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, model_name):
    # affiche et sauvegarde la matrice de confusion d'un modèle
    # une bonne matrice a des valeurs élevées sur la diagonale

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Fake', 'Real'],
        yticklabels=['Fake', 'Real']
    )
    plt.title(f'Matrice de Confusion — {model_name}')
    plt.ylabel('Label réel')
    plt.xlabel('Label prédit')
    plt.tight_layout()

    filename  = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
    save_path = os.path.join(config.RESULTS_DIR, filename)
    plt.savefig(save_path)
    plt.show()
    print(f'Matrice sauvegardée : {save_path}')


def plot_model_comparison(all_metrics):
    # graphique en barres qui compare les 4 métriques pour les 3 modèles
    # c'est la figure principale à montrer dans le rapport

    model_names = [m['model']     for m in all_metrics]
    accuracy    = [m['accuracy']  for m in all_metrics]
    precision   = [m['precision'] for m in all_metrics]
    recall      = [m['recall']    for m in all_metrics]
    f1          = [m['f1']        for m in all_metrics]

    x      = np.arange(len(model_names))
    width  = 0.2   # largeur de chaque barre

    fig, ax = plt.subplots(figsize=(11, 6))

    # une barre par métrique, décalée horizontalement
    ax.bar(x - 1.5 * width, accuracy,  width, label='Accuracy',  color='#3498db')
    ax.bar(x - 0.5 * width, precision, width, label='Precision', color='#2ecc71')
    ax.bar(x + 0.5 * width, recall,    width, label='Recall',    color='#e67e22')
    ax.bar(x + 1.5 * width, f1,        width, label='F1 Score',  color='#9b59b6')

    ax.set_title('Comparaison des modèles sur le test set', fontsize=14, pad=15)
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend()

    # on affiche les valeurs au-dessus des barres pour faciliter la lecture
    for bars in ax.containers:
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(config.RESULTS_DIR, 'model_comparison.png')
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f'Graphique sauvegardé : {save_path}')


def print_summary_table(all_metrics):
    # affiche un tableau récapitulatif propre dans le terminal

    print('\n' + '=' * 60)
    print('RÉSUMÉ FINAL — COMPARAISON DES MODÈLES')
    print('=' * 60)
    print(f'{"Modèle":<12} {"Accuracy":>10} {"Precision":>10} {"Recall":>10} {"F1":>10}')
    print('-' * 60)

    for m in all_metrics:
        print(
            f'{m["model"]:<12} '
            f'{m["accuracy"]:>10.4f} '
            f'{m["precision"]:>10.4f} '
            f'{m["recall"]:>10.4f} '
            f'{m["f1"]:>10.4f}'
        )

    print('=' * 60)

    # on identifie le meilleur modèle selon le F1
    best = max(all_metrics, key=lambda x: x['f1'])
    print(f'\nMeilleur modèle (F1) : {best["model"]} → {best["f1"]:.4f}')


# -----------------------------------------------------------------------------
# Pipeline complète
# -----------------------------------------------------------------------------

def run_evaluation():
    print('=== Évaluation finale des trois modèles ===\n')

    # chargement des données
    splits, word2idx = load_processed_data()
    X_train = splits['X_train']
    X_val   = splits['X_val']
    X_test  = splits['X_test']
    y_train = splits['y_train']
    y_val   = splits['y_val']
    y_test  = splits['y_test']

    # création des DataLoaders pour BiLSTM et RoBERTa
    _, _, test_loader_bilstm = get_bilstm_loaders(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        word2idx
    )

    _, _, test_loader_roberta, _ = get_roberta_loaders(
        X_train, X_val, X_test,
        y_train, y_val, y_test
    )

    # évaluation des trois modèles
    y_pred_base,    metrics_base    = eval_baseline_on_test(X_test, y_test)
    y_pred_bilstm,  metrics_bilstm  = eval_bilstm_on_test(test_loader_bilstm, word2idx)
    y_pred_roberta, metrics_roberta = eval_roberta_on_test(test_loader_roberta)

    all_metrics = [metrics_base, metrics_bilstm, metrics_roberta]

    # matrices de confusion pour les trois modèles
    plot_confusion_matrix(y_test, y_pred_base,    'Baseline')
    plot_confusion_matrix(y_test, y_pred_bilstm,  'BiLSTM')
    plot_confusion_matrix(y_test, y_pred_roberta, 'RoBERTa')

    # graphique de comparaison
    plot_model_comparison(all_metrics)

    # tableau récapitulatif
    print_summary_table(all_metrics)

    return all_metrics


if __name__ == '__main__':
    run_evaluation()
