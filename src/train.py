'''
author: Anmol Durgapal || @slothfulwave612

Python module for training models.
'''

## import necessary packages/modules
import os
import numpy as np
import pandas as pd
import joblib
from sklearn import preprocessing
from sklearn import metrics

from . import dispatcher, scaling

## get values from script file
TYPE = os.environ.get("TYPE")
TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
SAVE_PATH = os.environ.get("SAVE_PATH")
MODEL = os.environ.get("MODEL")
SCALE = os.environ.get("SCALE_TYPE")

if __name__ == '__main__':
    ## read in the datsets
    train_df = pd.read_pickle(TRAINING_DATA)
    test_df = pd.read_pickle(TEST_DATA)

    ## read in the datsets --> where final predicted will be appended
    final_train = pd.read_pickle(f"input/{TYPE}_dataset/train_test_data/train_df.pkl")
    final_test = pd.read_pickle(f"input/{TYPE}_dataset/train_test_data/test_df.pkl")

    ## drop unnecessary columns
    x_train = train_df.drop(["target"], axis=1)
    x_test = test_df.drop(["target"], axis=1)

    ## scale the values from train and test dataframe 
    if MODEL == "log_regg":
        ## scaling columns
        if TYPE == "advance":
            cols = [
                "angle", "distance", "player_in_between", "goal_keeper_angle"
            ]
        else:
            cols = [
                "angle", "distance"
            ]

        ## for train data
        scale_1 = scaling.Scale(
            df = x_train,
            scale_type = SCALE,
            cols = cols
        )

        ## for test data
        scale_2 = scaling.Scale(
            df = x_test,
            scale_type = SCALE,
            cols = cols
        )

        x_train = scale_1.fit_transform()
        x_test = scale_2.fit_transform()

    ## fetch target values for train and test dataframe
    y_train = train_df['target'].values
    y_test = test_df['target'].values

    ## train the model
    MODELS = dispatcher.get_models(TYPE)
    clf = MODELS[MODEL]
    clf.fit(x_train, y_train)

    ## predict values for train and test set
    preds_train = clf.predict_proba(x_train)[:, 1]
    preds_test = clf.predict_proba(x_test)[:, 1]

    ## add angle and distance values
    final_train["angle"] = train_df["angle"]
    final_train["distance"] = train_df["distance"]
    final_test["angle"] = test_df["angle"]
    final_test["distance"] = test_df["distance"]

    ## add the predicted values
    final_train['pred_' + MODEL] = preds_train
    final_test['pred_' + MODEL] = preds_test

    ## calculate auc-roc score
    roc_train = metrics.roc_auc_score(y_train, preds_train)
    roc_test = metrics.roc_auc_score(y_test, preds_test)

    ## print scores
    print("*** For Train Data ***")
    print(f"ROC-AUC Score: {roc_train}")
    print()
    print(f"*** For Test Data ***")
    print(f"ROC-AUC Score: {roc_test}")

    ## check for directory
    if os.path.isdir(SAVE_PATH) == False:
        ## make directory
        os.mkdir(SAVE_PATH)

    ## save the dataset
    final_train.to_pickle(SAVE_PATH + '/train_preds_' + MODEL + '.pkl')
    final_test.to_pickle(SAVE_PATH + '/test_preds_' + MODEL + '.pkl')

    if os.path.isdir(f"models/{TYPE}_models") == False:
        ## make directory
        os.mkdir(f"models/{TYPE}_models")

    ## save the model
    joblib.dump(clf, f'models/{TYPE}_models/{MODEL}.pkl')