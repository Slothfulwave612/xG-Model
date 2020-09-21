'''
author: Anmol Durgapal || @slothfulwave612

Python module for preparing the dataset before training and testing.
'''

## import necessary packages/modules
import sys
import os
import pandas as pd

from . import utils_io

CAT_TYPE = str(sys.argv[1])

def make_dataframe(path, path_save, name, drop_cols, filename):
    '''
    Function to making the required dataframe before training and testing.

    Arguments:
        path -- str, where the dataframe is stored.
        path_save -- str, where the new dataframe will be stored.
        name -- str, name of the dataframe with added extension.
        drop_cols -- list of columns to be dropped.
        filename -- str, name of the file + .pkl extension.
    '''
    ## read in the data
    df = pd.read_pickle(path+'/'+name)

    ## create distance and angle columns
    df['distance'] = df.apply(lambda x: utils_io.distance_bw_coordinates(x['x'], x['y']), axis=1)
    df['angle'] = df.apply(lambda x: utils_io.post_angle(x['x'], x['y']), axis=1)

    ## drop unnecessary columns
    df.drop(drop_cols, axis=1, inplace=True)

    ## save the dataframe
    df.to_pickle(path_save+'/'+filename)

if __name__ == "__main__":
    ## path to the data
    path = "input/basic_dataset/train_test_data_encoded"
    
    ## path where new dataset will be saved
    path_save = 'input/basic_dataset/train_test_data_final'

    if CAT_TYPE == "ohe":
        ## file names
        name_1 = 'train_ohe.pkl'
        filename_1 = 'train_ohe_final.pkl'
        name_2 = 'test_ohe.pkl'
        filename_2 = 'test_ohe_final.pkl'

        ## drop columns
        drop_cols = ['x', 'y', "shot_type_name", "body_part", "x0_Free Kick", "x1_Other"]

    elif CAT_TYPE == "label":
        ## file names
        name_1 = "train_label.pkl"
        filename_1 = "train_label_final.pkl"
        name_2 = "test_label.pkl"
        filename_2 = "test_label_final.pkl"

        ## drop columns
        drop_cols = ['x', 'y']

    ## check for directory
    if os.path.isdir(path_save) == False:
        ## make directory
        os.mkdir(path_save)

    ## make final train and test dataframes
    make_dataframe(path, path_save, name_1, drop_cols, filename_1)
    make_dataframe(path, path_save, name_2, drop_cols, filename_2)