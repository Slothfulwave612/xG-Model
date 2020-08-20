'''
author: Anmol Durgapal || @slothfulwave612

Python module for training models.
'''

## import necessary packages/modules
import os
import pandas as pd
import joblib
from sklearn import preprocessing
from sklearn import metrics

from . import dispatcher

## get values from script file
TRAINING_DATA = os.environ.get('TRAINING_DATA')
TEST_DATA = os.environ.get('TEST_DATA')
SAVE_PATH = os.environ.get('SAVE_PATH')
MODEL = os.environ.get('MODEL')

if __name__ == '__main__':
    ## read in the datsets
    train_df = pd.read_pickle(TRAINING_DATA)
    test_df = pd.read_pickle(TEST_DATA)

    ## initialize the min-max-scaler
    scaler_1 = preprocessing.MinMaxScaler()   ## for train
    scaler_2 = preprocessing.MinMaxScaler()   ## for test

    ## scale the values from train and test dataframe
    x_train = scaler_1.fit_transform(train_df.drop(['shot_statsbomb_xg', 'player_name', 'target'], axis=1))
    x_test = scaler_2.fit_transform(test_df.drop(['shot_statsbomb_xg', 'player_name', 'target'], axis=1))

    ## fetch target values for train and test dataframe
    y_train = train_df['target'].values
    y_test = test_df['target'].values
   
    ## train the model
    clf = dispatcher.MODELS[MODEL]
    clf.fit(x_train, y_train)

    ## predict values for train and test set
    preds_train = clf.predict_proba(x_train)[:, 1]
    preds_test = clf.predict_proba(x_test)[:, 1]

    ## add the predicted values
    train_df['pred_' + MODEL] = preds_train
    test_df['pred_' + MODEL] = preds_test

    ## print auc_roc_score
    print(f'ROC-AUC Score on train-dataset: {metrics.roc_auc_score(y_train, preds_train)}')
    print(f'ROC-AUC Score on test-dataset: {metrics.roc_auc_score(y_test, preds_test)}')

    ## check for directory
    if os.path.isdir(SAVE_PATH) == False:
        ## make directory
        os.mkdir(SAVE_PATH)

    ## save the dataset and model
    train_df.to_pickle(SAVE_PATH + '/train_preds_' + MODEL + '.pkl')
    test_df.to_pickle(SAVE_PATH + '/test_preds_' + MODEL + '.pkl')
    joblib.dump(clf, f'models/simple_models/{MODEL}.pkl')