'''
author: Anmol Durgapal || @slothfulwave612

Python module for encoding categorical variable.
'''

## import necessary package/module
import sys
import os
import pandas as pd
from sklearn import preprocessing

## fetch model
CAT_TYPE = str(sys.argv[1])

class CategoricalFeatures:
    '''
    class for encoding categorical variable.
    '''

    def __init__(self, df, categorical_features, encoding_type, handle_na=False):
        '''
        Function to initialize object.
        
        Arguments:
            self -- represents to the object.
            df -- pandas dataframe.
            categorical_features -- list of categorical column names.
            encoding_type -- str, label-encoding or one-hot-encoding.
            handle_na -- bool, True: to handle NaN values.
                               False: otherwise.
        '''
        self.dataframe = df
        self.cat_feats = categorical_features
        self.enc_type = encoding_type
        self.handle_na = handle_na
        self.label_encoders = dict()
        self.ohe = None

        if self.handle_na:
            for cols in self.cat_feats:
                self.dataframe.loc[: cols] = self.dataframe.loc[: cols].astype(str).fillna('-999999')
        
        self.output_df = self.dataframe.copy(deep=True)

    def _label_encoding(self):
        '''
        Function to perform labelencoding.

        Argument:
            self -- represents to the object.

        Returns:
            self.output_df -- pandas dataframe.
        '''
        for cols in self.cat_feats:
            ## create label-encoder object
            lbl = preprocessing.LabelEncoder()

            ## fit
            lbl.fit(self.output_df[cols].values)

            ## transform
            self.output_df.loc[:, cols] = lbl.transform(self.output_df[cols].values)

            ## save label_encoders
            self.label_encoders[cols] = lbl

        return self.output_df
    
    def _one_hot(self):
        '''
        Function to perform one-hot-encoding.

        Argument:
            self -- represents to the object.
        
        Returns:
            self.df -- pandas dataframe.
        '''
        ## create one-hot-encoding class
        ohe = preprocessing.OneHotEncoder()

        ## fit
        ohe.fit(self.dataframe[self.cat_feats].values)

        ## save ohe
        self.ohe = ohe

        ## transform
        temp = ohe.transform(self.dataframe[self.cat_feats].values)

        ## make dataframe
        temp_df = pd.DataFrame(temp.toarray().astype(int))
        temp_df.columns = ohe.get_feature_names()

        return temp_df

    def fit_tranform(self):
        '''
        Function to exceute the process of categorical encoding.

        Argument:
            self -- represent the object of the class.
        
        Returns:
        pandas dataframe
        '''
        if self.enc_type == 'label':
            return self._label_encoding()
        
        elif self.enc_type == 'ohe':
            return self._one_hot()
        
        else:
            raise Exception('Encoding type not understood')

    def transform(self, dataframe):
        '''
        Function to transform test dataset.

        Arguments:
            self -- represents the object of the class.
            dataframe -- pandas dataframe.

        Returns:
            dataframe -- pandas dataframe.
        '''
        if self.handle_na == True:
            ## fill nan values
            for cols in self.cat_feats:
                dataframe.loc[:, cols] = dataframe.loc[:, cols].astype(str).fillna('-999999')

        if self.enc_type == 'label':
            for c, lbl in self.label_encoders.items():
                dataframe.loc[:, c] = lbl.transform(dataframe[c].values)
            
            return dataframe
        
        elif self.enc_type == 'ohe':
            ohe = self.ohe.transform(dataframe[self.cat_feats]. values)

            ## make dataframe
            temp_df = pd.DataFrame(ohe.toarray().astype(int))
            temp_df.columns = self.ohe.get_feature_names()

            return temp_df
        
        else:
            raise Exception("Encoding Type not understood")

if __name__ == "__main__":
    ## path where data is stored
    path = 'input/basic_dataset/train_test_data'

    ## path where dataset will be stored
    path_save = 'input/basic_dataset/train_test_data_encoded'

    ## read in the data
    train_df = pd.read_pickle(path+'/train_df.pkl')
    test_df = pd.read_pickle(path+'/test_df.pkl')

    ## length of train-dataframe
    len_train = len(train_df)

    ## concatenate the two dataframes
    full_df = pd.concat([train_df, test_df]).reset_index(drop=True)

    ## categorical columns
    cols = ['shot_type_name', 'body_part']

    ## create object for Categorical Features
    if CAT_TYPE == "ohe":
        ## one hot encoding
        cat_feats = CategoricalFeatures(full_df, categorical_features=cols, encoding_type='ohe')
    
        ## encoding on data
        trans_df = cat_feats.fit_tranform()

        ## concate trans_df and full_df
        full_df = pd.concat([full_df, trans_df], axis=1)

    elif CAT_TYPE == "label":
        ## label encoding
        cat_feats = CategoricalFeatures(full_df, categorical_features=cols, encoding_type='label')

        full_df = cat_feats.fit_tranform()

    ## train and test df
    train_df = full_df.loc[:len_train-1, :].reset_index(drop=True)
    test_df = full_df.loc[len_train:, :].reset_index(drop=True)

    ## check for directory
    if os.path.isdir(path_save) == False:
        ## make directory
        os.mkdir(path_save)

    ## save the dataframes
    if CAT_TYPE == "ohe":
        train_df.to_pickle(path_save+'/train_ohe.pkl')
        test_df.to_pickle(path_save+'/test_ohe.pkl')
    elif CAT_TYPE == "label":
        train_df.to_pickle(path_save+'/train_label.pkl')
        test_df.to_pickle(path_save+'/test_label.pkl')