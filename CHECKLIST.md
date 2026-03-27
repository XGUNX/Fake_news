# Projet — Détection de Fake News sur les Réseaux Sociaux
**Dataset :** FakeNewsNet (Kaggle — mdepak)  
**Modèles :** TF-IDF + LR / BiLSTM / RoBERTa  
**App :** Streamlit  

---

## Les fichiers du dataset

Le dataset FakeNewsNet contient des articles collectés depuis deux sources de fact-checking :
**BuzzFeed News** et **PolitiFact**. Chaque source a ses articles séparés en deux fichiers.

| Fichier | Contenu |
|---|---|
| `BuzzFeed_fake_news_content.csv` | Articles **faux** vérifiés par BuzzFeed News |
| `BuzzFeed_real_news_content.csv` | Articles **vrais** vérifiés par BuzzFeed News |
| `PolitiFact_fake_news_content.csv` | Articles **faux** vérifiés par PolitiFact |
| `PolitiFact_real_news_content.csv` | Articles **vrais** vérifiés par PolitiFact |
| `*.txt` (plusieurs fichiers) | IDs de tweets liés aux articles — **on ne les utilise pas** (nécessite l'API Twitter) |

> **Important :** les labels (fake/real) ne sont **pas** dans les colonnes des CSV.  
> On les déduit directement du nom du fichier : `_fake_` → 0, `_real_` → 1.  
> Les colonnes principales utiles pour nous sont `title` et `text`.

---

## Checklist du projet

### ✅ Phase 1 — Mise en place

- [x] Définir la structure du projet (dossiers, fichiers)
- [x] Écrire `config.py` — tous les chemins, hyperparamètres et constantes au même endroit
- [x] Écrire `requirements.txt` — toutes les dépendances du projet
- [x] Installer les dépendances : `pip install -r requirements.txt`
- [x] Télécharger le dataset depuis Kaggle et le placer dans `data/raw/`

---

### ✅ Phase 2 — Exploration des données

- [x] Écrire `notebooks/01_exploration.ipynb`
  - [x] Charger les 4 CSV et fusionner en un seul dataframe
  - [x] Vérifier les colonnes, types, et premières lignes
  - [x] Analyser les valeurs manquantes
  - [x] Vérifier l'équilibre des classes → **résultat : 211 fake / 211 real, parfaitement équilibré**
  - [x] Analyser la distribution par source (BuzzFeed vs PolitiFact)
  - [x] Analyser la longueur des textes → **résultat : ~400 mots fake, ~500 mots real**
  - [x] Générer des nuages de mots (fake vs real)
  - [x] Analyser les n-grammes les plus fréquents
  - [x] Lire des exemples d'articles réels
  - [x] Vérifier les doublons → **résultat : aucun doublon**
  - [x] Remplir le tableau de conclusions (section 11)

---

### ✅ Phase 3 — Expérimentations

- [x] Écrire `notebooks/02_experiments.ipynb`
  - [x] Tester le chargement et la fusion des fichiers
  - [x] Tester la fonction de nettoyage du texte (`clean_text`)
  - [x] Tester la combinaison titre + corps avec le template `[SEP]`
  - [x] Tester le split train / val / test stratifié (80 / 10 / 10)
  - [x] Tester la construction du vocabulaire pour le BiLSTM
  - [x] Tester la conversion texte → indices pour le BiLSTM
  - [x] Tester la tokenisation RoBERTa et vérifier les shapes

---

### ⬜ Phase 4 — Preprocessing

- [ ] Écrire `src/preprocess.py` en s'appuyant sur ce qu'on a testé dans `02_experiments.ipynb`
  - [ ] Fonction de chargement et fusion des 4 CSV
  - [ ] Fonction de nettoyage du texte
  - [ ] Fonction de combinaison titre + corps
  - [ ] Fonction de split train / val / test
  - [ ] Fonction de construction du vocabulaire BiLSTM
  - [ ] Sauvegarder les données nettoyées dans `data/processed/`

---

### ⬜ Phase 5 — Modèle Baseline

- [ ] Écrire `src/baseline.py`
  - [ ] Vectorisation TF-IDF
  - [ ] Entraînement Logistic Regression
  - [ ] Évaluation : accuracy, precision, recall, F1
  - [ ] Sauvegarder le modèle dans `models/`
- [ ] Obtenir un premier score de référence à battre

---

### ⬜ Phase 6 — Modèle BiLSTM

- [ ] Écrire `src/dataset.py` — classe PyTorch Dataset pour charger les données
- [ ] Écrire `src/bilstm.py` — architecture du modèle BiLSTM
- [ ] Écrire `src/train.py` — boucle d'entraînement générique
- [ ] Entraîner le BiLSTM et sauvegarder les poids dans `models/`
- [ ] Comparer les résultats avec le baseline

---

### ⬜ Phase 7 — Modèle RoBERTa

- [ ] Écrire `src/roberta_model.py` — wrapper RoBERTa pour la classification
- [ ] Adapter `src/train.py` pour RoBERTa si nécessaire
- [ ] Fine-tuner RoBERTa et sauvegarder les poids dans `models/`
- [ ] Comparer les résultats avec les deux modèles précédents

---

### ⬜ Phase 8 — Évaluation

- [ ] Écrire `src/evaluate.py`
  - [ ] Calculer accuracy, precision, recall, F1 pour chaque modèle
  - [ ] Générer les matrices de confusion
  - [ ] Générer un graphique de comparaison des 3 modèles
  - [ ] Sauvegarder tous les résultats dans `results/`

---

### ⬜ Phase 9 — Application Streamlit

- [ ] Écrire `app.py`
  - [ ] Interface pour saisir un texte
  - [ ] Chargement du meilleur modèle sauvegardé
  - [ ] Affichage de la prédiction (Fake / Real) avec un score de confiance
  - [ ] Lancer l'app : `streamlit run app.py`

---

### ⬜ Phase 10 — Finalisation

- [ ] Écrire `run_training.py` — script unique pour tout entraîner d'un coup
- [ ] Vérifier que tout tourne de bout en bout sans erreur
- [ ] Nettoyer le code et les notebooks
- [ ] Vérifier que tous les fichiers ont des commentaires en français

---


