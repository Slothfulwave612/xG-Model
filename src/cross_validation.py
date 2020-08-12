'''
author: Anmol Durgapal || @slothfulwave612

Python module for cross-validation.
'''

## import necessary packages/modules
import pandas as pd
from sklearn import model_selection

class CrossValidation:
    '''
    a cross validation class.
    --> Binary Classification Problem.
    '''

    def __init__(self, df, target_cols, shuffle=True, num_folds=5):
        '''
        Function for init class object.

        Arguments:
            self -- reference to the object.
            df -- pandas dataframe.
            target_cols -- list of target column/s.
            shuffle -- bool, True: for shuffling the dataframe.
                             False: otherwise. Default: True
            num_folds -- int, number of folds required. Default: 5.
        '''
        self.dataframe = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.num_folds = num_folds
        self.shuffle = shuffle

        if self.shuffle == True:
            ## shuffle the dataframe
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)
        
        ## make a new column that'll hold the fold number
        self.dataframe['kfold'] = -1
    
    def make_folds(self):
        '''
        Function to make folds.

        Arguments:
            self -- reference to the object.
        
        Returns:
            self.dataframe -- pandas dataframe.
        '''
        if self.num_targets != 1:
            ## check error
            raise Exception('Invalid number of targets')

        ## our target column
        target = self.target_cols[0]    

        ## unique values in the target columns
        unique_value = self.dataframe[target].nunique()

        if unique_value == 1:
            ## check error
            raise Exception('Only one unique value found!')

        elif unique_value > 1:
            ## make StratifiedKFold object
            kf = model_selection.StratifiedKFold(n_splits=self.num_folds, shuffle=False)

            ## assing fold numbers
            for fold, (_, valid_idx) in enumerate(kf.split(X=self.dataframe, y=self.dataframe[target].values)):
                self.dataframe.loc[valid_idx, 'kfold'] = fold
            
        return self.dataframe

if __name__ == '__main__':
    ## path where train data is saved
    path = 'input/simple_dataset'

    ## load the train-dataframe
    df = pd.read_pickle(path+'/train_df.pkl')

    ## make CrossValidation class object
    cv = CrossValidation(df, target_cols=['target'], shuffle=True, num_folds=5)

    ## make folds
    new_df = cv.make_folds()

    ## kfold column to int
    new_df['kfold'].astype(int)

    ## save the dataframe
    new_df.to_pickle(path+'/train_folds.pkl')