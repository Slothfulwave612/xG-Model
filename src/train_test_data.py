'''
author: Anmol Durgapal || @slothfulwave612

Python module for making train and test data.

The train dataset will include shot information from:
                1. All La Liga Matches
                2. All UEFA Champions League Final Matches
                3. All Mens World Cup(2018) Matches
                4. All Premier League Matches
                5. All NWSL(2018) Matches

The test dataset will include shot information from :
                1. All FAWSL Matches
                2. All Women World Cup(2019) Matches
'''

## import necessary modules/packages
import sys
import os
import pandas as pd

from . import utils_io as uio

TYPE = str(sys.argv[1])

if __name__ == "__main__":
    ## path where the shot dataset is present
    path_data = f"input/{TYPE}_dataset/all_competitions"

    ## path where the new dataset is saved
    path_save = f"input/{TYPE}_dataset/train_test_data"

    ## make split
    uio.make_train_test(path=path_data, path_save=path_save)