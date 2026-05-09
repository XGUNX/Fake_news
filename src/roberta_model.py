# =============================================================================
# roberta_model.py
# -----------------------------------------------------------------------------
# Ce fichier gère tout ce qui est spécifique à RoBERTa :
#   - le chargement du modèle pré-entraîné
#   - la sauvegarde et le rechargement des poids fine-tunés
#   - la prédiction sur un texte brut (utilisé par l'app Streamlit)
#
# Pourquoi RoBERTa et pas BERT ?
# RoBERTa (Robustly Optimized BERT) est entraîné plus longtemps,
# sur plus de données et sans la tâche NSP (Next Sentence Prediction).
# Sur des tâches de classification de texte court il surpasse BERT
# de façon assez constante, surtout sur du texte bruité comme les réseaux sociaux.
#
# Ce qu'on fait ici c'est du fine-tuning :
# on prend RoBERTa déjà entraîné sur des milliards de mots et on l'adapte
# à notre tâche spécifique (fake vs real) avec nos 422 articles.
# La boucle d'entraînement elle-même est dans train.py.
# =============================================================================

import os
import sys

import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
from src.preprocess import clean_text


# -----------------------------------------------------------------------------
# Chargement du modèle
# -----------------------------------------------------------------------------

def load_roberta_model(weights_path=None):
    # charge RoBERTa avec une tête de classification binaire (fake / real)
    # si weights_path est fourni on charge nos poids fine-tunés
    # sinon on charge les poids originaux pré-entraînés de HuggingFace

    print(f'Chargement de {config.ROBERTA_MODEL_NAME}...')

    # num_labels=2 : RoBERTa ajoute automatiquement une couche dense (2 neurones) en sortie
    # c'est ce qu'on appelle la "classification head"
    model = RobertaForSequenceClassification.from_pretrained(
        config.ROBERTA_MODEL_NAME,
        num_labels=2
    )

    if weights_path and os.path.exists(weights_path):
        # on charge seulement les poids sauvegardés pendant l'entraînement
        # map_location='cpu' : on charge d'abord sur CPU même si le GPU est disponible
        # ça évite les problèmes si le modèle a été entraîné sur un GPU différent
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f'Poids fine-tunés chargés depuis : {weights_path}')
    else:
        print('Poids pré-entraînés originaux chargés (pas encore fine-tuné)')

    return model


def load_roberta_tokenizer():
    # charge le tokenizer RoBERTa
    # le tokenizer est séparé du modèle — on peut l'utiliser sans charger le modèle
    tokenizer = RobertaTokenizer.from_pretrained(config.ROBERTA_MODEL_NAME)
    return tokenizer


# -----------------------------------------------------------------------------
# Prédiction sur un texte brut
# -----------------------------------------------------------------------------

def predict_roberta(text, model, tokenizer):
    # prend un texte brut, le nettoie, le tokenise et retourne la prédiction
    # cette fonction est appelée par l'app Streamlit

    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    model  = model.to(device)
    model.eval()  # mode évaluation : désactive le dropout

    # nettoyage du texte — même transformation qu'à l'entraînement
    # si on ne nettoie pas ici, le modèle voit un texte différent de ce qu'il a appris
    text_clean = clean_text(text)

    # tokenisation : même paramètres qu'à l'entraînement
    encoded = tokenizer(
        text_clean,
        max_length=config.ROBERTA_MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids      = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    # pas besoin de calculer les gradients pour une prédiction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # outputs.logits : scores bruts de forme (1, 2)
    # softmax : convertit les scores en probabilités qui somment à 1
    probs = torch.softmax(outputs.logits, dim=1)[0]  # [prob_fake, prob_real]

    label      = torch.argmax(probs).item()   # 0 = fake, 1 = real
    confidence = probs[label].item()           # probabilité de la classe prédite

    return label, confidence, probs.cpu().numpy()


# -----------------------------------------------------------------------------
# Sauvegarde manuelle (optionnelle — train.py sauvegarde déjà automatiquement)
# -----------------------------------------------------------------------------

def save_roberta(model, filename='roberta_best.pt'):
    # sauvegarde les poids du modèle dans le dossier models/
    save_path = os.path.join(config.MODELS_DIR, filename)
    torch.save(model.state_dict(), save_path)
    print(f'Modèle sauvegardé : {save_path}')


# -----------------------------------------------------------------------------
# Test rapide
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    print('=== Test de roberta_model.py ===\n')

    # on charge le modèle sans poids fine-tunés pour tester
    model     = load_roberta_model()
    tokenizer = load_roberta_tokenizer()

    # test de prédiction sur deux exemples fictifs
    exemples = [
        'The president signed a new law that will help millions of Americans.',
        'SHOCKING: Secret government plan to control your mind revealed!!!'
    ]

    print('\nTest de prédiction :')
    for texte in exemples:
        label, confidence, probs = predict_roberta(texte, model, tokenizer)
        label_name = config.LABEL_MAP[label]
        print(f'\n  Texte      : {texte[:70]}...')
        print(f'  Prédiction : {label_name}')
        print(f'  Confiance  : {confidence:.2%}')
        print(f'  Proba fake : {probs[0]:.2%} | Proba real : {probs[1]:.2%}')

    print('\nTest OK')
