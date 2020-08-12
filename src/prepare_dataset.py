'''
author: Anmol Durgapal || @slothfulwave612

Python module for preparing the dataset before training and testing.
'''

## import necessary packages/modules
import pandas as pd

from . import utils_io

def make_dataframe(path, name, drop_cols, filename):
    '''
    Function to making the required dataframe before training and testing.

    Arguments:
        path -- str, where the dataframe is stored.
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
    df.to_pickle(path+'/'+filename)

if __name__ == '__main__':
    ## path and file name
    path = 'input/simple_dataset'
    name_1 = 'train_trans.pkl'
    filename_1 = 'train_final.pkl'
    name_2 = 'test_trans.pkl'
    filename_2 = 'test_final.pkl'

    ## drop columns
    drop_cols = ['location', 'shot_type_name', 'body_part', 'x0_Kick Off', 'x1_Other']

    ## make final train and test dataframes
    make_dataframe(path, name_1, drop_cols, filename_1)
    make_dataframe(path, name_2, drop_cols, filename_2)