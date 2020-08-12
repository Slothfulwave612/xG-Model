'''
author: Anmol Durgapal || @slothfulwave612

Python module for making train and test data.

The train dataset will include shot information from:
                1. All La Liga Matches
                2. All UEFA Champions League Final Matches
                3. All FAWSL Matches
                4. All Women World Cup(2019) Matches
                5. All NWSL(2018) Matches

The test dataset will include shot information from :
                1. All Mens World Cup(2018) Matches
                2. All Premier League Matches
'''

## import necessary modules/packages
import os
import pandas as pd

from . import utils_io

def make_train_test(path):
    '''
    Function for making and saving train and test data.

    Argument:
        path -- str, path where the shot data is stored.
    '''
    ## load in all the datasets
    ucl_data = pd.read_pickle(path+'/Champions_League_shots.pkl')
    fawsl_data = pd.read_pickle(path+'/FA_Women\'s_Super_League_shots.pkl')
    menwc_data = pd.read_pickle(path+'/FIFA_World_Cup_shots.pkl')
    ll_data = pd.read_pickle(path+'/La_Liga_shots.pkl')
    nwsl_data = pd.read_pickle(path+'/NWSL_shots.pkl')
    pl_data = pd.read_pickle(path+'/Premier_League_shots.pkl')
    wwc_data = pd.read_pickle(path+'/Women\'s_World_Cup_shots.pkl')

    ## make train dataframe
    train_df = pd.concat(
        [
            ll_data, 
            ucl_data, 
            fawsl_data, 
            wwc_data, 
            nwsl_data
        ]
    )

    ## make test dataframe
    test_df = pd.concat(
        [
            menwc_data,
            pl_data
        ]
    )

    ## save train dataframe
    train_df.to_pickle(path+'/train_df.pkl')

    ## save test dataframe
    test_df.to_pickle(path+'/test_df.pkl')

if __name__ == '__main__':
    ## path where the shot dataset is present
    path_data='input/simple_dataset'

    ## make split
    make_train_test(path=path_data)