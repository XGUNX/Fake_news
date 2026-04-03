# =============================================================================
# app.py
# -----------------------------------------------------------------------------
# Application Streamlit pour démontrer le projet en temps réel.
# L'utilisateur entre un texte et le modèle prédit si c'est fake ou real.
#
# Pour lancer l'app :
#   streamlit run app.py
#
# Ce que fait l'app :
#   1. L'utilisateur choisit un modèle (Baseline, BiLSTM, RoBERTa)
#   2. Il entre un texte dans la zone de saisie
#   3. L'app nettoie le texte, le passe dans le modèle et affiche la prédiction
#   4. On affiche aussi le score de confiance pour que ce soit plus parlant
# =============================================================================

import os
import sys
import pickle

import torch
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.preprocess import clean_text
from src.bilstm import BiLSTM
from src.roberta_model import load_roberta_model, load_roberta_tokenizer, predict_roberta
from src.baseline import predict_baseline


# -----------------------------------------------------------------------------
# Configuration de la page Streamlit
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title='Fake News Detector',
    page_icon='🔍',
    layout='centered'
)


# -----------------------------------------------------------------------------
# Chargement des modèles
# -----------------------------------------------------------------------------
# On utilise @st.cache_resource pour ne charger les modèles qu'une seule fois
# Sans ça le modèle se rechargerait à chaque interaction de l'utilisateur
# ce qui rendrait l'app très lente

@st.cache_resource
def load_baseline():
    # charge le modèle baseline (TF-IDF + LR) et son vectorizer
    model_path = os.path.join(config.MODELS_DIR, 'baseline_model.pkl')
    vec_path   = os.path.join(config.MODELS_DIR, 'baseline_vectorizer.pkl')

    if not os.path.exists(model_path):
        return None, None

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vec_path, 'rb') as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


@st.cache_resource
def load_bilstm():
    # charge le BiLSTM et le vocabulaire nécessaire pour convertir les textes
    weights_path = os.path.join(config.MODELS_DIR, 'bilstm_best.pt')
    vocab_path   = os.path.join(config.DATA_PROCESSED_DIR, 'word2idx.pkl')

    if not os.path.exists(weights_path) or not os.path.exists(vocab_path):
        return None, None

    with open(vocab_path, 'rb') as f:
        word2idx = pickle.load(f)

    model = BiLSTM(vocab_size=len(word2idx))
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()

    return model, word2idx


@st.cache_resource
def load_roberta():
    # charge RoBERTa fine-tuné et son tokenizer
    weights_path = os.path.join(config.MODELS_DIR, 'roberta_best.pt')

    if not os.path.exists(weights_path):
        return None, None

    model     = load_roberta_model(weights_path)
    tokenizer = load_roberta_tokenizer()

    return model, tokenizer


# -----------------------------------------------------------------------------
# Prédiction BiLSTM
# -----------------------------------------------------------------------------

def predict_with_bilstm(text, model, word2idx):
    # nettoie le texte, le convertit en indices et fait la prédiction avec le BiLSTM
    from src.preprocess import text_to_indices

    text_clean = clean_text(text)
    indices    = text_to_indices(text_clean, word2idx)

    # on crée un batch de taille 1 (un seul texte)
    input_tensor = torch.LongTensor([indices])

    with torch.no_grad():
        predictions, probs = model.predict(input_tensor)

    label      = predictions[0].item()
    confidence = probs[0][label].item()

    return label, confidence, probs[0].numpy()


# -----------------------------------------------------------------------------
# Interface Streamlit
# -----------------------------------------------------------------------------

def main():

    # --- En-tête ---
    st.title('🔍 Fake News Detector')
    st.markdown('Entrez un texte pour savoir si c\'est une **fake news** ou une **vraie information**.')
    st.markdown('---')

    # --- Sidebar : choix du modèle ---
    st.sidebar.title('Paramètres')
    st.sidebar.markdown('### Modèle à utiliser')

    model_choice = st.sidebar.radio(
        label='',
        options=['Baseline (TF-IDF + LR)', 'BiLSTM', 'RoBERTa'],
        index=2   # RoBERTa sélectionné par défaut car c'est le meilleur
    )

    # petite description de chaque modèle dans la sidebar
    descriptions = {
        'Baseline (TF-IDF + LR)': '📊 Modèle simple et rapide. Pas de deep learning. Score de référence.',
        'BiLSTM'                : '🧠 Réseau de neurones récurrent. Lit le texte dans les deux sens.',
        'RoBERTa'               : '🚀 Transformer pré-entraîné. Le plus précis des trois.'
    }
    st.sidebar.info(descriptions[model_choice])

    # --- Zone de saisie ---
    st.markdown('### Entrez votre texte ici')

    # quelques exemples pour guider l'utilisateur
    exemple = st.selectbox(
        'Ou choisissez un exemple :',
        options=[
            '',
            'The president signed a bipartisan bill to fund infrastructure projects across the country.',
            'BREAKING: Scientists discover that vaccines contain microchips to track the population!!!',
            'Congress passed new legislation aimed at reducing carbon emissions by 2035.',
            'SHOCKING truth about the moon landing that NASA doesn\'t want you to know!'
        ]
    )

    # la zone de texte se remplit avec l'exemple sélectionné
    user_text = st.text_area(
        label='Texte à analyser :',
        value=exemple,
        height=180,
        placeholder='Collez ici un titre ou un article de presse...'
    )

    # --- Bouton de prédiction ---
    if st.button('🔍 Analyser', use_container_width=True):

        # vérification que l'utilisateur a entré quelque chose
        if not user_text.strip():
            st.warning('Veuillez entrer un texte avant d\'analyser.')
            return

        # on affiche un spinner pendant que le modèle tourne
        with st.spinner('Analyse en cours...'):

            label, confidence, probs = None, None, None

            # --- Baseline ---
            if model_choice == 'Baseline (TF-IDF + LR)':
                model, vectorizer = load_baseline()

                if model is None:
                    st.error('Le modèle Baseline n\'a pas encore été entraîné. Lancez `python src/baseline.py` d\'abord.')
                    return

                text_clean        = clean_text(user_text)
                label, confidence = predict_baseline(text_clean, model, vectorizer)
                probs             = None   # LR retourne predict_proba mais on simplifie ici

            # --- BiLSTM ---
            elif model_choice == 'BiLSTM':
                model, word2idx = load_bilstm()

                if model is None:
                    st.error('Le modèle BiLSTM n\'a pas encore été entraîné. Lancez `run_training.py` d\'abord.')
                    return

                label, confidence, probs = predict_with_bilstm(user_text, model, word2idx)

            # --- RoBERTa ---
            elif model_choice == 'RoBERTa':
                model, tokenizer = load_roberta()

                if model is None:
                    st.error('Le modèle RoBERTa n\'a pas encore été fine-tuné. Lancez `run_training.py` d\'abord.')
                    return

                label, confidence, probs = predict_roberta(user_text, model, tokenizer)

        # --- Affichage du résultat ---
        st.markdown('---')
        st.markdown('### Résultat')

        # couleur et emoji selon la prédiction
        if label == 0:
            # fake news → rouge
            st.error(f'🔴 **FAKE NEWS** — Confiance : {confidence:.1%}')
        else:
            # real news → vert
            st.success(f'🟢 **VRAIE INFORMATION** — Confiance : {confidence:.1%}')

        # barre de progression pour visualiser la confiance
        st.progress(confidence)

        # détail des probabilités si disponibles (BiLSTM et RoBERTa)
        if probs is not None:
            st.markdown('#### Détail des probabilités')
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label='🔴 Fake News',
                    value=f'{probs[0]:.1%}'
                )
            with col2:
                st.metric(
                    label='🟢 Real News',
                    value=f'{probs[1]:.1%}'
                )

        # texte nettoyé — utile pour comprendre ce que le modèle a vraiment reçu
        with st.expander('Voir le texte après nettoyage'):
            st.code(clean_text(user_text))

    # --- Footer ---
    st.markdown('---')
    st.caption('Projet Deep Learning — FakeNewsNet Dataset | BiLSTM & RoBERTa')


if __name__ == '__main__':
    main()
