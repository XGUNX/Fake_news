# =============================================================================
# run_training.py
# -----------------------------------------------------------------------------
# Point d'entrée unique pour entraîner tous les modèles dans l'ordre.
# C'est le seul fichier à lancer pour reproduire tous les résultats du projet.
#
# Ordre d'exécution :
#   1. Preprocessing  → nettoie et sauvegarde les données
#   2. Baseline       → TF-IDF + Régression Logistique
#   3. BiLSTM         → entraînement du réseau récurrent
#   4. RoBERTa        → fine-tuning du transformer
#   5. Évaluation     → compare les trois modèles et génère les graphiques
#
# Usage :
#   python run_training.py              # entraîne tout
#   python run_training.py --skip-preprocessing  # si les données sont déjà traitées
# =============================================================================

import os
import sys
import time
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

from src.preprocess   import run_preprocessing, load_processed_data
from src.baseline     import run_baseline
from src.bilstm       import BiLSTM
from src.dataset      import get_bilstm_loaders, get_roberta_loaders
from src.train        import train_bilstm, train_roberta
from src.evaluate     import run_evaluation


def print_section(title):
    # affiche un séparateur visuel pour chaque étape dans le terminal
    print('\n' + '=' * 60)
    print(f'  {title}')
    print('=' * 60)


def main(skip_preprocessing=False):

    start_total = time.time()

    # -------------------------------------------------------------------------
    # Étape 1 — Preprocessing
    # -------------------------------------------------------------------------
    print_section('ÉTAPE 1 — PREPROCESSING')

    if skip_preprocessing:
        # si on a déjà traité les données on les recharge directement
        # ça économise du temps quand on relance juste pour retester les modèles
        print('Preprocessing ignoré — rechargement des données existantes...')
        splits, word2idx = load_processed_data()
    else:
        # première exécution : on nettoie, on split, on construit le vocab
        X_train, X_val, X_test, y_train, y_val, y_test, word2idx = run_preprocessing()
        splits = {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test
        }

    X_train = splits['X_train']
    X_val   = splits['X_val']
    X_test  = splits['X_test']
    y_train = splits['y_train']
    y_val   = splits['y_val']
    y_test  = splits['y_test']

    # -------------------------------------------------------------------------
    # Étape 2 — Baseline
    # -------------------------------------------------------------------------
    print_section('ÉTAPE 2 — BASELINE (TF-IDF + Régression Logistique)')

    # le baseline est rapide — quelques secondes max
    baseline_metrics = run_baseline()

    # -------------------------------------------------------------------------
    # Étape 3 — BiLSTM
    # -------------------------------------------------------------------------
    print_section('ÉTAPE 3 — BILSTM')

    # création des DataLoaders pour le BiLSTM
    train_loader_bilstm, val_loader_bilstm, _ = get_bilstm_loaders(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        word2idx
    )

    # instanciation du modèle avec la taille du vocabulaire construit au preprocessing
    model_bilstm = BiLSTM(vocab_size=len(word2idx))

    # lancement de l'entraînement — sauvegarde automatique du meilleur modèle
    bilstm_weights = train_bilstm(model_bilstm, train_loader_bilstm, val_loader_bilstm)
    print(f'BiLSTM entraîné — poids : {bilstm_weights}')

    # -------------------------------------------------------------------------
    # Étape 4 — RoBERTa
    # -------------------------------------------------------------------------
    print_section('ÉTAPE 4 — ROBERTA (Fine-tuning)')

    # création des DataLoaders pour RoBERTa
    # le tokenizer est chargé à l'intérieur de get_roberta_loaders
    train_loader_rob, val_loader_rob, _, tokenizer = get_roberta_loaders(
        X_train, X_val, X_test,
        y_train, y_val, y_test
    )

    # fine-tuning de RoBERTa — c'est l'étape la plus longue
    # sur CPU : ~20-30 min | sur GPU : ~2-5 min
    _, roberta_weights = train_roberta(train_loader_rob, val_loader_rob)
    print(f'RoBERTa fine-tuné — poids : {roberta_weights}')

    # -------------------------------------------------------------------------
    # Étape 5 — Évaluation finale
    # -------------------------------------------------------------------------
    print_section('ÉTAPE 5 — ÉVALUATION FINALE')

    # compare les trois modèles sur le test set et génère les graphiques
    all_metrics = run_evaluation()

    # -------------------------------------------------------------------------
    # Résumé du temps total
    # -------------------------------------------------------------------------
    total_time = time.time() - start_total
    minutes    = int(total_time // 60)
    secondes   = int(total_time % 60)

    print_section('ENTRAÎNEMENT TERMINÉ')
    print(f'Temps total : {minutes}min {secondes}s')
    print(f'\nFichiers générés :')
    print(f'  models/baseline_model.pkl')
    print(f'  models/bilstm_best.pt')
    print(f'  models/roberta_best.pt')
    print(f'  results/confusion_matrix_baseline.png')
    print(f'  results/confusion_matrix_bilstm.png')
    print(f'  results/confusion_matrix_roberta.png')
    print(f'  results/model_comparison.png')
    print(f'\nPour lancer l\'app : streamlit run app.py')


# -----------------------------------------------------------------------------
# Gestion des arguments en ligne de commande
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entraîne tous les modèles du projet.')

    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Ignore le preprocessing si les données sont déjà traitées'
    )

    args = parser.parse_args()
    main(skip_preprocessing=args.skip_preprocessing)
