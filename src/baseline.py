# =============================================================================
# baseline.py
# -----------------------------------------------------------------------------
# Modèle de référence : TF-IDF + Régression Logistique
# C'est le modèle le plus simple qu'on peut faire sur ce problème.
# Il ne fait pas de deep learning — pas de réseau de neurones, pas de PyTorch.
# Son rôle c'est de nous donner un score minimal à dépasser avec BiLSTM et RoBERTa.
# Si nos modèles deep learning font moins bien que ça, c'est qu'il y a un problème.
# =============================================================================

import os
import sys
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
from src.preprocess import load_processed_data


# -----------------------------------------------------------------------------
# Vectorisation TF-IDF
# -----------------------------------------------------------------------------

def get_tfidf_features(X_train, X_val, X_test):
    # TF-IDF transforme chaque texte en un vecteur de nombres
    # TF = fréquence du mot dans le document
    # IDF = à quel point le mot est rare dans tout le corpus
    # un mot fréquent partout (comme "the") aura un score faible
    # un mot rare mais présent dans un document aura un score élevé

    print('Vectorisation TF-IDF...')

    # on instancie le vectoriseur avec les paramètres de config
    vectorizer = TfidfVectorizer(
        max_features=config.BASELINE_MAX_FEATURES,   # on garde les N mots les plus importants
        ngram_range=config.BASELINE_NGRAM_RANGE,     # unigrammes et bigrammes
        sublinear_tf=True                             # applique log(tf) pour réduire l'effet des mots très fréquents
    )

    # IMPORTANT : on fit uniquement sur le train, jamais sur val ou test
    # si on fittait sur tout le dataset on aurait du data leakage
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # pour val et test on applique juste le transform (pas de fit)
    X_val_tfidf  = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f'Shape des features train : {X_train_tfidf.shape}')
    # shape attendue : (nb_articles_train, max_features) → ex: (337, 10000)

    return vectorizer, X_train_tfidf, X_val_tfidf, X_test_tfidf


# -----------------------------------------------------------------------------
# Entraînement
# -----------------------------------------------------------------------------

def train_baseline(X_train_tfidf, y_train):
    # la régression logistique est un classifieur linéaire simple
    # elle apprend quels mots sont associés à fake (0) et quels mots à real (1)
    # c'est rapide à entraîner — quelques secondes max

    print('Entraînement de la régression logistique...')

    model = LogisticRegression(
        C=config.BASELINE_C,               # contrôle la régularisation (évite le surapprentissage)
        max_iter=config.BASELINE_MAX_ITER, # nombre max d'itérations pour la convergence
        random_state=config.RANDOM_SEED,
        solver='lbfgs'                     # algorithme d'optimisation — bon pour les petits datasets
    )

    model.fit(X_train_tfidf, y_train)
    print('Entraînement terminé')

    return model


# -----------------------------------------------------------------------------
# Évaluation
# -----------------------------------------------------------------------------

def evaluate(model, X_tfidf, y_true, split_name='Test'):
    # calcule et affiche les métriques sur un split donné

    y_pred = model.predict(X_tfidf)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)

    print(f'\n--- Résultats sur {split_name} ---')
    print(f'  Accuracy  : {acc:.4f}')
    print(f'  Precision : {prec:.4f}')
    print(f'  Recall    : {rec:.4f}')
    print(f'  F1 Score  : {f1:.4f}')

    # rapport détaillé par classe
    print(f'\nRapport complet :')
    print(classification_report(y_true, y_pred, target_names=['Fake', 'Real']))

    # on retourne les métriques dans un dict pour les comparer plus tard avec BiLSTM et RoBERTa
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


# -----------------------------------------------------------------------------
# Matrice de confusion
# -----------------------------------------------------------------------------

def plot_confusion_matrix(model, X_tfidf, y_true):
    # la matrice de confusion montre combien d'articles sont bien/mal classifiés
    # ligne = label réel, colonne = label prédit
    # idéalement on veut des grands nombres sur la diagonale (vrais positifs / vrais négatifs)

    y_pred = model.predict(X_tfidf)
    cm     = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,          # affiche les nombres dans les cases
        fmt='d',             # format entier
        cmap='Blues',
        xticklabels=['Fake', 'Real'],
        yticklabels=['Fake', 'Real']
    )

    plt.title('Matrice de Confusion — Baseline')
    plt.ylabel('Label réel')
    plt.xlabel('Label prédit')
    plt.tight_layout()

    # sauvegarde dans results/
    save_path = os.path.join(config.RESULTS_DIR, 'confusion_matrix_baseline.png')
    plt.savefig(save_path)
    plt.show()
    print(f'Matrice sauvegardée : {save_path}')


# -----------------------------------------------------------------------------
# Sauvegarde du modèle
# -----------------------------------------------------------------------------

def save_baseline(model, vectorizer):
    # on sauvegarde à la fois le modèle et le vectorizer
    # les deux sont nécessaires pour faire une prédiction sur un nouveau texte

    model_path = os.path.join(config.MODELS_DIR, 'baseline_model.pkl')
    vec_path   = os.path.join(config.MODELS_DIR, 'baseline_vectorizer.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    with open(vec_path, 'wb') as f:
        pickle.dump(vectorizer, f)

    print(f'Modèle sauvegardé     : {model_path}')
    print(f'Vectorizer sauvegardé : {vec_path}')


def load_baseline():
    # recharge le modèle et le vectorizer depuis les fichiers sauvegardés
    model_path = os.path.join(config.MODELS_DIR, 'baseline_model.pkl')
    vec_path   = os.path.join(config.MODELS_DIR, 'baseline_vectorizer.pkl')

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(vec_path, 'rb') as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


# -----------------------------------------------------------------------------
# Prédiction sur un texte brut (utilisé par l'app Streamlit)
# -----------------------------------------------------------------------------

def predict_baseline(text, model, vectorizer):
    # prend un texte brut et retourne le label prédit + la probabilité
    # on vectorise d'abord avec le même vectorizer utilisé à l'entraînement

    X = vectorizer.transform([text])

    label       = model.predict(X)[0]
    # predict_proba retourne [[prob_fake, prob_real]]
    confidence  = model.predict_proba(X)[0][label]

    return label, confidence


# -----------------------------------------------------------------------------
# Pipeline complète
# -----------------------------------------------------------------------------

def run_baseline():
    print('=== Baseline : TF-IDF + Régression Logistique ===\n')

    # chargement des données déjà traitées par preprocess.py
    splits, _ = load_processed_data()
    X_train, X_val, X_test = splits['X_train'], splits['X_val'], splits['X_test']
    y_train, y_val, y_test = splits['y_train'], splits['y_val'], splits['y_test']

    # vectorisation
    vectorizer, X_train_tfidf, X_val_tfidf, X_test_tfidf = get_tfidf_features(
        X_train, X_val, X_test
    )

    # entraînement
    model = train_baseline(X_train_tfidf, y_train)

    # évaluation sur val puis test
    # on regarde val d'abord pour voir si le modèle généralise bien
    # on ne touche au test qu'une seule fois à la fin
    val_metrics  = evaluate(model, X_val_tfidf,  y_val,  split_name='Validation')
    test_metrics = evaluate(model, X_test_tfidf, y_test, split_name='Test')

    # matrice de confusion sur le test
    plot_confusion_matrix(model, X_test_tfidf, y_test)

    # sauvegarde
    save_baseline(model, vectorizer)

    print('\n=== Baseline terminée ===')
    print(f'Score F1 à battre : {test_metrics["f1"]:.4f}')

    return test_metrics


# lancement direct : python src/baseline.py
if __name__ == '__main__':
    run_baseline()
