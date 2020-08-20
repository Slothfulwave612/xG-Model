'''
author: Anmol Durgapal || @slothfulwave612

Python module for preparing the dataset before training and testing.
'''

## import necessary packages/modules
import os
import pandas as pd

from . import utils_io

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

    ## add x and y coordinate columns
    df['x'] = df['location'].apply(utils_io.coordinates_x)
    df['y'] = df['location'].apply(utils_io.coordinates_y)

    ## drop unnecessary columns
    df.drop(drop_cols, axis=1, inplace=True)

    ## create distance and angle columns
    df['distance'] = df.apply(lambda x: utils_io.distance_bw_coordinates(x['x'], x['y']), axis=1)
    df['angle'] = df.apply(lambda x: utils_io.post_angle(x['x'], x['y']), axis=1)

    ## save the dataframe
    df.to_pickle(path_save+'/'+filename)

if __name__ == '__main__':
    ## path and file name
    path = 'input/simple_dataset/train_test_data_encoded'
    path_save = 'input/simple_dataset/train_test_data_final'
    name_1 = 'train_ohe.pkl'
    filename_1 = 'train_ohe_final.pkl'
    name_2 = 'test_ohe.pkl'
    filename_2 = 'test_ohe_final.pkl'

    ## drop columns
    drop_cols = ['location', 'shot_type_name', 'body_part', 'x0_Kick Off', 'x1_Other']

    ## make final train and test dataframes
    make_dataframe(path, path_save, name_1, drop_cols, filename_1)
    make_dataframe(path, path_save, name_2, drop_cols, filename_2)