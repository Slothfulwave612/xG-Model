'''
author: Anmol Durgapal || @slothfulwave612

Python module for training models.
'''

## import necessary packages/modules
import os
import pandas as pd
import joblib
from sklearn import metrics

from . import dispatcher

## get values from script file
TRAINING_DATA = os.environ.get('TRAINING_DATA')
FOLD = int(os.environ.get('FOLD'))
MODEL = os.environ.get('MODEL')

## for a given fold-number we have defined fold-number for train-data
## e.g. if 0 is given as a fold number, then rows with folds 1, 2, 3 and 4 will be treated as train-data
## and rows with folds 0 is treated as validation-data
FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

if __name__ == '__main__':
    ## read in the datsets
    df = pd.read_pickle(TRAINING_DATA)

    ## fetch train and valid data   
    train_df = df[df['kfold'].isin(FOLD_MAPPING.get(FOLD))].reset_index(drop=True)
    valid_df = df[df['kfold'] == FOLD].reset_index(drop=True)

    ## fetch target values for train and validation data
    y_train = train_df['target'].values
    y_valid = valid_df['target'].values

    ## drop columns
    train_df.drop(['target', 'kfold', 'shot_statsbomb_xg', 'player_name'], axis=1, inplace=True)
    valid_df.drop(['target', 'kfold', 'shot_statsbomb_xg', 'player_name'], axis=1, inplace=True)

    ## train the model
    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, y_train)
    preds = clf.predict_proba(valid_df)[:, 1]

    ## print auc_roc_score
    print(metrics.roc_auc_score(y_valid, preds))

    ## save the model
    joblib.dump(clf, f'models/simple_models/{MODEL}_{FOLD}.pkl')