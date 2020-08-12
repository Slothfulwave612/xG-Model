'''
author: Anmol Durgapal || @slothfulwave612

Python module for i/o operations on the dataset.
'''

## import necessary packages/modules
import numpy as np
import pandas as pd
from pandas import json_normalize
import json
import math
import multiprocessing
from tqdm.auto import tqdm, trange
from joblib import Parallel, delayed

def get_competition(path):
    '''
    Function for getting data about all the competitions.

    Argument:
        path -- str, path to competition.json file.

    Returns:
        comp_df -- pandas dataframe, all competition data.
    '''
    ## load the json file
    comp_data = json.load(open(path))

    ## make pandas dataframe
    comp_df = pd.DataFrame(comp_data)

    return comp_df

def flatten_json(sub_str):
    '''
    Function to take out values from nested dictionary present in 
    the json file, so to make a representable dataframe.
    
    ---> This piece of code was found on stackoverflow <--
    
    Argument:
        sub_str -- substructure defined in the json file.
    
    Returns:
        flattened out information.
    '''
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(sub_str)
    
    return out

def get_matches(comp_id, season_id, path):
    '''
    Function for getting match-data for a given competition
    
    Arguments:
        comp_id -- int, the competition id.
        season_id -- int, the season id.
        path -- str, path to .json file containing match data.
    
    Returns:
        match_df -- pandas dataframe, containing all the matches 
    '''    
    ## loading up the data from json file
    match_data = json.load(open(path, encoding='utf8'))
    
    ## flattening the json file
    match_flatten = [flatten_json(x) for x in match_data]
    
    ## creating a dataframe
    match_df = pd.DataFrame(match_flatten)
    
    match_df_cols = list(match_df.columns)
    
    ## renaming the dataframe
    for i in range(len(match_df_cols)):
        if match_df_cols[i].count('away_team') == 2:
            ## for away_team columns
            match_df_cols[i] = match_df_cols[i][len('away_team_'):]
        
        elif match_df_cols[i].count('_0') == 1:
            ## for _0 columns
            match_df_cols[i] = match_df_cols[i].replace('_0', '')
        
        elif match_df_cols[i].count('competition') == 2:
            ## for competition columns
            match_df_cols[i] = match_df_cols[i][len('competition_'):]
        
        elif match_df_cols[i].count('home_team') == 2:
            ## for away_team columns
            match_df_cols[i] = match_df_cols[i][len('home_team_'):]
        
        elif match_df_cols[i].count('season') == 2:
            ## for away_team columns
            match_df_cols[i] = match_df_cols[i][len('season_'):]

    match_df.columns = match_df_cols 
        
    return match_df

def make_event_df(match_id, path):
    '''
    Function for making event dataframe.
    
    Argument:
        match_id -- int, the required match id for which event data will be constructed.
        path -- str, path to .json file containing event data.
    
    Returns:
        df -- pandas dataframe, the event dataframe for the particular match.
    '''
    ## read in the json file
    event_json = json.load(open(path, encoding='utf-8'))
    
    ## normalize the json data
    df = json_normalize(event_json, sep='_')
    
    return df

def full_season_events(match_df, match_ids, path, comp_name=None, leave=True):
    '''
    Function to make event dataframe for a full season.
    
    Arguments:
        match_df -- pandas dataframe, containing match-data.
        match_id -- list, list of match id.
        path -- str, path to directory where .json file is listed.
                e.g. '../input/Statsbomb/data/events'
        comp_name -- str, competition name + season name, default: None.
        leave -- keeps all traces of the progressbar upon termination of iteration.
    
    Returns:
        event_df -- pandas dataframe, containing event data for the whole season.
    '''
    ## init an empty dataframe
    event_df = pd.DataFrame()

    if comp_name == None:
        t = match_ids
    else:
        t = tqdm(match_ids, desc=f'Grabbing data for {comp_name}', position=0, leave=leave)

    for match_id in t:
        ## .json file
        temp_path = path + f'/{match_id}.json'

        temp_df = make_event_df(match_id, temp_path)
        event_df = pd.concat([event_df, temp_df])
        
    return event_df.loc[event_df['type_name'] == 'Shot']     

def multiple_season_event_df(comp_name, comp_id, season_ids, path_match, path_season):
    '''
    Function for making event dataframe having multile seasons 
    for the same competition.
    
    Arguments:
        comp_name -- str, competition name + season 
        comp_id -- int, competition id.
        season_ids -- list, list containing season ids.
        path_match -- str, path to .json file containing match data.
        path_season -- str, path to directory where .json file is listed.
                       e.g. '../input/Statsbomb/data/events'
    
    Returns:
        event_df -- pandas dataframe, containing event of multiple seasons.
    '''
    ## init an empty dataframe
    event_df = pd.DataFrame()
    
    ## making the event-dataframe
    for season_id in tqdm(season_ids, desc=f'Grabbing data for {comp_name}', leave=True):
        
        ## add season id to path-match
        team_path_match = path_match + f'/{comp_id}/{season_id}.json'

        ## make a match dataframe for a particular season
        match_df = get_matches(comp_id, season_id, team_path_match)

        ## list all the match ids
        match_ids = list(match_df['match_id'].unique())
        
        comp_name_ = match_df['competition_name'].unique()[0] + '-' + match_df['season_name'].unique()[0]

        ## create the event dataframe for the whole season
        temp_df = full_season_events(match_df, match_ids, path_season, comp_name=comp_name_, leave=False)

        ## concat the dataframes
        event_df = pd.concat([event_df, temp_df])
    
    ## make final dataframe
    event_df = event_df.reset_index(drop=True)
    
    return event_df    

def goal(value):
    '''
    Function to output 1: if goal or 0: otherwise.

    Arguments:
        value -- str, shot-outcome-name.

    Returns:
        0 or 1 -- 0 means no goal 1 means goal.
    '''
    if value == 'Goal':
        return 1
    else:
        return 0

def body_part(value):
    '''
    Function to output: Head -- if it is a header,
                        Foot -- if it is right/left foot,
                        Other -- if any other body part
    '''
    if value == 'Left Foot' or value == 'Right Foot':
        return 'Foot'
    else:
        return value


def coordinates_x(value):
    '''
    Return x coordinate
    '''
    return value[0]

def coordinates_y(value):
    '''
    Return 80 - x coordinate
    '''
    return 80 - value[1]    

def distance_bw_coordinates(x1, y1, x2=120.0, y2=40.0):
    '''
    Function for calculating the distance between shot location 
    and the goal post.
    
    Arguments:
    x1, y1 -- float, the x and y coordinate for shot location.
    x2, y2 -- float, the x and y coordinate for the goal post location.(default for Statsbomb defined goal-post)
    '''
    diff_sqr_x = (x2 - x1)**2
    diff_sqr_y = (y2 - y1)**2
    
    distance = math.sqrt(diff_sqr_x + diff_sqr_y)   ## euclidean distnace
    
    return distance

def post_angle(x, y, g1_x=120, g1_y=36, g2_x=120, g2_y=44):
    '''
    Function to calculate the post angle.
    
    Arguments:
    x -- float, x coordinate from where the shot was taken.
    y -- float, y coordinate from where the shot was taken.
    g1 and g2 are the coordinates of the two woodwork, default values
    specifying the woodwork coordinate for Statsbomb data.
    
    Returns:
    angle -- float, the angle in degrees.
    '''
    if x == 120:
        return 0
    
    ## calculating the three sides of the triangle.
    A_dis = distance_bw_coordinates(x, y, g1_x, g1_y)
    B_dis = distance_bw_coordinates(x, y, g2_x, g2_y)
    C_dis = distance_bw_coordinates(g1_x, g1_y, g2_x, g2_y)
    
    ## using cosine law
    value = ((A_dis**2) + (B_dis**2) - (C_dis**2)) / (2 * A_dis * B_dis)
    
    angle = np.degrees(np.arccos(value))
    
    return angle    

# def make_event_df(x):
#     '''
#     Function for making event dataframe.
    
#     Argument:
#         match_id -- int, the required match id for which event data will be constructed.
#         path -- str, path to .json file containing event data.
    
#     Returns:
#         df -- pandas dataframe, the event dataframe for the particular match.
#     '''
#     path = x[1]
#     ## read in the json file
#     event_json = json.load(open(path, encoding='utf-8'))
    
#     ## normalize the json data
#     df = json_normalize(event_json, sep='_')
    
#     return df


# def full_season_events(comp_name, match_df, match_ids, path):
#     event_df = pd.DataFrame()

#     with multiprocessing.Pool() as p:
#         # Generate job tuples
#         jobs = [(match_id, path + f"/{match_id}.json") for match_id in match_ids]
#         # Run & get results from multiprocessing generator

#         for temp_df in tqdm(
#             p.imap_unordered(make_event_df, jobs, chunksize=10),
#             total=len(jobs),
#             desc=f"Making Event Data For {comp_name}",
#             leave=False
#         ):
#             event_df = pd.concat([event_df, temp_df], sort=True)

#     return event_df    

# def multiple_season_event_df(comp_name, comp_id, season_ids, path_match, path_season):
#     '''
#     Function for making event dataframe having multile seasons 
#     for the same competition.
    
#     Arguments:
#         comp_name -- str, competition name + season 
#         comp_id -- int, competition id.
#         season_ids -- list, list containing season ids.
#         path_match -- str, path to .json file containing match data.
#         path_season -- str, path to directory where .json file is listed.
#                        e.g. '../input/Statsbomb/data/events'
    
#     Returns:
#         event_df -- pandas dataframe, containing event of multiple seasons.
#     '''
#     ## init an empty dataframe
#     event_df = pd.DataFrame()
    
#     ## making the event-dataframe
#     for season_id in tqdm(season_ids, desc=f'Grabbing data for {comp_name}', leave=True):
        
#         ## add season id to path-match
#         team_path_match = path_match + f'/{season_id}.json'

#         ## make a match dataframe for a particular season
#         match_df = get_matches(comp_id, season_id, team_path_match)
    
#         ## list all the match ids
#         match_ids = list(match_df['match_id'].unique())
        
#         # comp_name_ = match_df['competition_name'].unique()[0] + '-' + match_df['season_name'].unique()[0]

#         ## create the event dataframe for the whole season
#         temp_df = full_season_events(comp_name, match_df, match_ids, path_season)

#         ## concat the dataframes
#         event_df = pd.concat([event_df, temp_df], sort=True)
    
#     ## make final dataframe
#     event_df = event_df.reset_index(drop=True)
    
#     return event_df           