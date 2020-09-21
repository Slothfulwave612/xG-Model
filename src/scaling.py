"""
__author__: Anmol_Durgapal(@slothfulwave612)

Python module for performing scaling to the data.
"""

import sys
import os
import pandas as pd
from sklearn import preprocessing

class Scale:
    """
    class containing method to perform scaling.
    """

    def __init__(self, df, scale_type, cols):
        """
        Function to initialize class object.

        Args:
            df (pandas.DataFrame): the dataframe to be used.
            scale_type (str): standardization(std) or normalization(nrm).
            cols (list): columns in which scaling will be performed.
        """        
        self.df = df
        self.scale_type = scale_type
        self.cols = cols
    
    def __standardization(self):
        """
        Function to perform standardization.

        Returns:
            pandas.DataFrame: dataframe with scaled features.
        """        
        ## init object of StandardScaler class
        scaler = preprocessing.StandardScaler()

        ## fit the scaler to the data
        scaler.fit(self.df.loc[:, self.cols])

        ## transform the data
        self.df.loc[:, self.cols] = scaler.transform(self.df.loc[:, self.cols])

        return self.df
    
    def __normalization(self):
        """
        Function to perform normalization.

        Returns:
            pandas.DataFrame: dataframe with scaled features.
        """        
        ## init object of MinMaxScaler class
        scaler = preprocessing.MinMaxScaler()

        ## fit the scaler to the data
        scaler.fit(self.df.loc[:, self.cols])

        ## transform the data
        self.df.loc[:, self.cols] = scaler.transform(self.df.loc[:, self.cols])

        return self.df

    def fit_transform(self):
        """
        Function to perform scaling.

        Returns:
            pandas.DataFrame: dataframe with scaled features.
        """
        if self.scale_type == "std":
            return self.__standardization()
        
        elif self.scale_type == "nrm":
            return self.__normalization()
        else:
            return "Scaling type not understood"