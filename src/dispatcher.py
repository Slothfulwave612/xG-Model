'''
author: Anmol Durgapal || @slothfulwave612

Python module for dispatching different modules.
'''

## import necessary packages/modules
from sklearn import linear_model
from sklearn import ensemble
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

## all the models here are tuned -- see explore_experiment.ipynb 
MODELS = {
    "log_regg": linear_model.LogisticRegression(
        C=0.3593813663804626,
        penalty="l2",
        solver="saga",
        n_jobs=-1,
        max_iter=200,
        verbose=1
    ),

    "random_forest": ensemble.RandomForestClassifier(
        criterion="entropy",
        max_depth=5,
        n_estimators=100,
        min_samples_split=2,
        n_jobs=-1,
        verbose=1
    ),

    "xg_boost": XGBClassifier(
        min_child_weight=5,
        max_depth=4,
        learning_rate=0.05,
        gamma=0.,
        colsample_bytree=0.7,
        n_jobs=-1,
        verbosity=1
    )
}
