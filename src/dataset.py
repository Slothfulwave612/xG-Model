'''
author: Anmol Durgapal || @slothfulwave612

Python module for creating datasets using Statsbomb open-data.

1. Simple Dataset.
'''

## import necessary packages/modules
import os

from . import utils_io

def simple_dataset(comp_name, comp_id, season_ids, path_season, path_match, path_save, filename):
    '''
    Function to make a dataset for our simple-xG-model.

    The dataset will have:
        1. x and y location,
        2. Statsbomb-xG,
        3. Player Name,
        4. Shot Type Name,
        5. Body Part
        6. Goal or No-Goal.

    Arguments:
        path_season -- str, path to the directory where event files are saved.
        path_match -- str, path to the directory where match data file is stored for each competitions.
        path_save -- str, path to the directory where the shot dataframe will be saved.
    '''
    
    ## get event-dataframe
    event_df = utils_io.multiple_season_event_df(comp_name, comp_id, season_ids, path_match, path_season)

    ## col-list
    col_list = ['location', 'shot_statsbomb_xg', 'player_name', 'shot_outcome_name', 'shot_body_part_name', 'shot_type_name']

    ## shot-dataframe from event-dataframe
    shot_df = event_df.loc[:, col_list]

    ## create body part column
    shot_df['body_part'] = shot_df['shot_body_part_name'].apply(utils_io.body_part)

    ## create target column - 2 classes - goal and no goal
    shot_df['target'] = shot_df['shot_outcome_name'].apply(utils_io.goal)

    ## drop shot_outcome_name and shot_body_part_name column
    shot_df.drop(['shot_outcome_name', 'shot_body_part_name'], axis=1, inplace=True)

    shot_df.to_pickle(f'{path_save}/{filename}')

if __name__ == '__main__':
    ## path where competition.json file is saved
    path_comp='input/Statsbomb/data/competitions.json'

    ## path where event files are saved
    path_season='input/Statsbomb/data/events'

    ## path where match files are saved
    path_match='input/Statsbomb/data/matches' 

    ## path where the dataset will be saved
    path_save='input/simple_dataset'

    ## get competition data
    comp_df = utils_io.get_competition(path_comp)

    ## init an empty dictionary
    comp_info = dict()

    ## add comp_name and respective ids
    for _, data in comp_df.iterrows():
        if comp_info.get(data['competition_name']) == None:
            comp_info[data['competition_name']] = data['competition_id']

    for comp_name, comp_id in comp_info.items():
        ## fetch season ids
        season_ids = comp_df.loc[comp_df['competition_id'] == comp_id, 'season_id'].to_list()

        ## save shot-dataframe
        simple_dataset(
            comp_name=comp_name,
            comp_id=comp_id,
            season_ids=season_ids,
            path_season=path_season,
            path_match=path_match,
            path_save=path_save,
            filename=comp_name.replace(' ', '_') + '_shots.pkl'
        )
