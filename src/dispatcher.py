'''
author: Anmol Durgapal || @slothfulwave612

Python module for dispatching different modules.
'''

## import necessary packages/modules
from sklearn import linear_model
from sklearn import ensemble

MODELS = {
    'log_regg': linear_model.LogisticRegression(n_jobs=-1),
    'random_forest': ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2)
}