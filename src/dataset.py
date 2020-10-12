'''
author: Anmol Durgapal || @slothfulwave612

Python module for creating datasets using Statsbomb open-data.

1. Simple Dataset.
'''

## import necessary packages/modules
import os
import sys

from . import utils_io as uio

TYPE = str(sys.argv[1])

if __name__ == "__main__":
    ## path where competition.json file is saved
    path_comp = "input/Statsbomb/data/competitions.json"

    ## path where event files are saved
    path_season = "input/Statsbomb/data/events"

    ## path where match files are saved
    path_match = "input/Statsbomb/data/matches"

    ## get competition data
    comp_df = uio.get_competition(path_comp)

    ## init an empty dictionary
    comp_info = dict()

    ## add comp_name and respective ids
    for _, data in comp_df.iterrows():
        if comp_info.get(data['competition_name']) == None:
            comp_info[data['competition_name']] = data['competition_id']

    if TYPE == "basic":
        ## check for the directory
        if os.path.isdir('input/basic_dataset') == False:
            ## make required directories
            os.mkdir('input/basic_dataset')
            os.mkdir('input/basic_dataset/all_competitions')
            os.mkdir('input/basic_dataset/train_test_data')
            os.mkdir('input/basic_dataset/train_test_data_encoded')
            os.mkdir('input/basic_dataset/train_test_data_final')
            os.mkdir('input/basic_dataset/train_test_data_result')

        if os.path.isdir('input/basic_dataset/all_competitions') == False:
            ## make required directories
            os.mkdir('input/basic_dataset/all_competitions')

        ## path where the dataset will be saved
        path_save='input/basic_dataset/all_competitions'

        for comp_name, comp_id in comp_info.items():
            ## fetch season ids
            season_ids = comp_df.loc[comp_df['competition_id'] == comp_id, 'season_id'].to_list()

            ## save shot-dataframe
            uio.simple_dataset(
                comp_name=comp_name,
                comp_id=comp_id,
                season_ids=season_ids,
                path_season=path_season,
                path_match=path_match,
                path_save=path_save,
                filename=comp_name.replace(' ', '_') + '_shots.pkl'
            )
    
    elif TYPE == "intermediate":
        ## check for the directory
        if os.path.isdir('input/intermediate_dataset') == False:
            ## make required directories
            os.mkdir('input/intermediate_dataset')
            os.mkdir('input/intermediate_dataset/all_competitions')
            os.mkdir('input/intermediate_dataset/train_test_data')
            os.mkdir('input/intermediate_dataset/train_test_data_encoded')
            os.mkdir('input/intermediate_dataset/train_test_data_final')
            os.mkdir('input/intermediate_dataset/train_test_data_result')

        if os.path.isdir('input/intermediate_dataset/all_competitions') == False:
            ## make required directories
            os.mkdir('input/intermediate_dataset/all_competitions')
        
        ## path where the dataset will be saved
        path_save='input/intermediate_dataset/all_competitions'

        for comp_name, comp_id in comp_info.items():
            ## fetch season ids
            season_ids = comp_df.loc[comp_df['competition_id'] == comp_id, 'season_id'].to_list()

            ## get event-dataframe
            shot_df = uio.multiple_season_event_df(comp_name, comp_id, season_ids, path_match, path_season, shot="intermediate")

            ## filename for saving
            filename = comp_name.replace(' ', '_') + "_shots.pkl"

            ## save the dataset
            shot_df.to_pickle(f'{path_save}/{filename}')