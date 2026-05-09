# run_roberta_only.py
# lance uniquement le fine-tuning RoBERTa
# à utiliser quand baseline et BiLSTM sont déjà entraînés

import os
import sys
import numpy as np
import torch

# ton i5 13H a 12 threads — on les utilise tous
torch.set_num_threads(12)
torch.set_num_interop_threads(12)
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE']  = '1'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

from src.preprocess   import load_processed_data
from src.dataset      import get_roberta_loaders
from src.train        import train_roberta
from src.evaluate     import run_evaluation

print('Rechargement des données traitées...')
splits, word2idx = load_processed_data()

X_train = splits['X_train']
X_val   = splits['X_val']
X_test  = splits['X_test']
y_train = splits['y_train']
y_val   = splits['y_val']
y_test  = splits['y_test']

# on réduit le train set à 5000 articles — CPU ne peut pas gérer 36 000
print('Réduction du dataset pour RoBERTa...')
np.random.seed(config.RANDOM_SEED)
idx         = np.random.choice(len(X_train), size=10000, replace=False)
X_train_rob = X_train[idx]
y_train_rob = y_train[idx]
print(f'Train réduit : {len(X_train_rob)} articles')

# création des DataLoaders
train_loader, val_loader, _, tokenizer = get_roberta_loaders(
    X_train_rob, X_val, X_test,
    y_train_rob, y_val, y_test
)

# fine-tuning
_, weights = train_roberta(train_loader, val_loader)
print(f'RoBERTa terminé — poids : {weights}')

# évaluation finale des 3 modèles
print('\nLancement de l\'évaluation finale...')
run_evaluation()