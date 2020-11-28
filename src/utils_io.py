'''
author: Anmol Durgapal || @slothfulwave612

Python module for i/o operations on the dataset.
'''

## import necessary packages/modules
import os
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import json
import math
import multiprocessing
from tqdm.auto import tqdm, trange
import statsmodels.api as sm

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

def full_season_events(match_df, match_ids, path, comp_name=None, leave=True, shot="basic"):
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
        event_df = pd.concat([event_df, temp_df], sort=False)
    
    if shot == "basic":
        return event_df.loc[event_df['type_name'] == 'Shot']     
    elif shot == "intermediate":
        return intermediate_dataset(event_df)
    elif shot == "advance":
        return intermediate_dataset(event_df, adv=True)

def multiple_season_event_df(comp_name, comp_id, season_ids, path_match, path_season, shot):
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
        temp_df = full_season_events(match_df, match_ids, path_season, comp_name=comp_name_, leave=False, shot=shot)

        ## add competition 
        temp_df["comp_name"] = comp_name_

        ## concat the dataframes
        event_df = pd.concat([event_df, temp_df], sort=False)
    
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
    if value == "Left Foot" or value == "Right Foot":
        return "Foot"
    else:
        return value

def change_dims(old_value, old_min, old_max, new_min, new_max):
    '''
    Function for changing the coordinates to our pitch dimensions.

    Arguments:
        old_value, old_min, old_max, new_min, new_max -- float values.

    Returns:
        new_value -- float value(the coordinate value either x or y).
    '''
    ## calculate the value
    new_value = ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min

    return new_value

def coordinates_x(value):
    '''
    Return x coordinate
    '''
    value_x = change_dims(value[0], 0, 120, 0, 104)
    return value_x

def coordinates_y(value):
    '''
    Return 80 - x coordinate
    '''
    value_y = change_dims(80- value[1], 0, 80, 0, 68)
    return value_y

def distance_bw_coordinates(x1, y1, x2=104.0, y2=34.0):
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

def post_angle(x, y, g1_x=104, g1_y=30.34, g2_x=104, g2_y=37.66):
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
    if x == 104 and (30.34 <= y <= 37.66):
        return 180

    if x == 104 and (y > 37.66 or y < 30.34):
        return 0
    
    ## calculating the three sides of the triangle.
    A_dis = distance_bw_coordinates(x, y, g1_x, g1_y)
    B_dis = distance_bw_coordinates(x, y, g2_x, g2_y)
    C_dis = distance_bw_coordinates(g1_x, g1_y, g2_x, g2_y)
    
    ## using cosine law
    value = ((A_dis**2) + (B_dis**2) - (C_dis**2)) / (2 * A_dis * B_dis)

    angle = np.degrees(np.arccos(value))
    
    return angle 

def create_result_df(df, length, col):    
    '''
    Function to create a result dataframe(statsbomb_xg vs predicted_xg).

    Arguments:
        df -- pandas dataframe.
        length -- int, length of the dataframe.
        col -- str, column name for predicted xG value.

    Returns:
        result -- pandas dataframe containing statsbomb_xg and predicted_xg as columns.
    '''
    ## fetch all the player names
    players = df.loc[df['target'] == 1, 'player_name'].value_counts()[:length].index

    ## init a dictionary
    result_dict = {
        'player_name': [],
        'shots': [],
        'goals': [],
        'statsbomb_xg': [],
        'predicted_xg': []
    }

    ## calculate required values
    for player in players:

        ## total number of shots taken by a player
        shots = len(df.loc[(df['player_name'] == player)])
        
        ## total number of goals scored by a player
        goals = len(df.loc[
            (df['player_name'] == player) &
            (df['target'] == 1)
        ])
        
        ## aggregated statsbomb-xG-value for a player
        stats_xg = df.loc[
            (df['player_name'] == player),
            'shot_statsbomb_xg'
        ].sum()
        
        ## aggregated predicted-xG-value for a player
        pred_xg = df.loc[
            (df['player_name'] == player),
            col
        ].sum()
        
        ## append result to result_dict
        result_dict['player_name'].append(player)
        result_dict['shots'].append(shots)
        result_dict['goals'].append(goals)
        result_dict['statsbomb_xg'].append(stats_xg)
        result_dict['predicted_xg'].append(pred_xg)
        
    ## create pandas dataframe
    result = pd.DataFrame(result_dict).sort_values(by='goals', ascending=False).reset_index(drop=True)

    return result

def get_indices(width, height, xpartition, ypartition, xinput, yinput):
    """
    Function to get the indices for grid.

    Args:
        width (float): width of the pitch.
        height (float): height of the pitch.
        xpartition (int): number of rows in a grid
        ypartition (int): number of colimns in a grid.
        xinput (float): x-coordinate location.
        yinput (float): y-coordinate location.

    Returns:
        tuple: containing indices for the grid.
    """    
    ## calculate number of partitions in x and y
    x_step = width / xpartition
    y_step = height / ypartition

    ## calculate x and y values
    x = math.ceil((xinput if xinput > 0 else 0.5) / x_step) # handle border cases as well
    y = math.ceil((yinput if yinput > 0 else 0.5) / y_step)  # handle border cases as well

    return (
        ypartition - y, x - 1
    )

def get_stats(x_val, y_val):
    """
    Function to train model using statsmodel api.

    Args:
        x_val (pandas.DataFrame): containing features.
        y_val (numpy.ndarray): containing targets.

    Returns:
        statsmodels.iolib.summary.Summary: summary about our model
    """    
    ## train logistic model
    log_reg = sm.Logit(y_val, x_val).fit(maxiter=1000)

    return log_reg

def make_df(df, cols, rows=25):
    """
    Function to make the required dataframe.

    Args:
        df (pandas.DataFrame))
        cols (list): the required columns.
        rows (int, optional): First rows. Defaults to 25.
    """    
    ## fetch columns
    df = df[cols]

    ## a new dataframe
    new_df = df.groupby(by="player_name").sum().reset_index().sort_values("target", ascending=False).reset_index(drop=True)

    ## rename target column
    new_df = new_df.rename({"target": "goals_scored"}, axis=1)

    ## fetch first few rows
    first_few = new_df.head(rows)

    return first_few


def area(x1, y1, x2, y2, x3, y3): 
    """
    Funtion to calculate area of triangle.

    Args:
        float: coordinates for triangle vertices.

    Returns:
        float: area of the triangle.
    """    
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1)  
                + x3 * (y1 - y2)) / 2.0) 

def is_inside(player_coord_x, player_coord_y, shot_location_x, shot_location_y, pole_1_x=104.0, pole_1_y=30.34, pole_2_x=104.0, pole_2_y=37.66):
    """
    Function to return whether player is between the player taking shot and goal.

    Args:
        player_coord_x (float): player-coordinate-x.
        player_coord_y (float): player-coordinate-y.
        shot_location_x (float): shot-coordinate-x.
        shot_location_y (float): shot-coordinate-y.
        pole_1_x (float, optional): goal-post(1) coordinate x. Defaults to 104.0.
        pole_1_y (float, optional): goal-post(1) coordinate y. Defaults to 30.34.
        pole_2_x (float, optional): goal-post(2) coordinate x. Defaults to 104.0.
        pole_2_y (float, optional): goal-post(2) coordinate x. Defaults to 37.66.
    
    Returns:
        bool: True if present else False.
    """    
    # calculate area of triangle ABC 
    A = area(shot_location_x, shot_location_y, pole_1_x, pole_1_y, pole_2_x, pole_2_y) 
  
    # calculate area of triangle PBC  
    A1 = area(player_coord_x, player_coord_y, pole_1_x, pole_1_y, pole_2_x, pole_2_y) 
      
    # calculate area of triangle PAC  
    A2 = area(player_coord_x, player_coord_y, shot_location_x, shot_location_y, pole_2_x, pole_2_y) 
      
    # calculate area of triangle PAB  
    A3 = area(player_coord_x, player_coord_y, shot_location_x, shot_location_y, pole_1_x, pole_1_y) 
      
    # check if sum of A1, A2 and A3  
    # is same as A 
    if round(A,2) == round(A1 + A2 + A3, 2): 
        return True
    else: 
        return False

def freeze_frame_vars(freeze_frame, shot_location_x, shot_location_y):
    """
    Function for making freeze frame variables.

    Args:
        freeze_frame (list): containing tracking information.
        shot_location_x (float): shot coordinate location x.
        shot_location_y (float): shot coordinate location y.

    Returns:
        float values: 1. number of teammates between goal and shot-location.
                      2. number of opponents(excluding goalkeeper) between goal and shot-location.
                      3. goalkeeper covering angle.
                      4. distance between goalkeeper and the goal.
                      5. distance between goalkeeper and the shot-location.
    """    
    ## init two variable to 0
    count_teammate, count_opponent, goal_keeper_angle, dis_goal_keeper, dis_shot_keeper = 0, 0, 0, 0, 0

    ## traverse the freeze frame
    for frame in freeze_frame:
        ## fetch coodinate location of the players
        x_coord = coordinates_x(frame["location"])
        y_coord = coordinates_y(frame["location"])

        ## fetch player's position
        position = frame["position"]["name"]

        if position != "Goalkeeper":
            if frame["teammate"] == True and is_inside(x_coord, y_coord, shot_location_x, shot_location_y):
                count_teammate += 1
            
            elif frame["teammate"] == False and is_inside(x_coord, y_coord, shot_location_x, shot_location_y):
                count_opponent += 1
        else:
            ## compute goalkeeper covering angle
            goal_keeper_angle = post_angle(x_coord, y_coord)

            ## compute distance between goalkeeper and goal
            dis_goal_keeper = distance_bw_coordinates(x_coord, y_coord)

            ## compute distance between goalkeeper and shot-location
            dis_shot_keeper = distance_bw_coordinates(x_coord, y_coord, shot_location_x, shot_location_y)
    
    return count_teammate, count_opponent, goal_keeper_angle, dis_goal_keeper, dis_shot_keeper


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
    event_df = multiple_season_event_df(comp_name, comp_id, season_ids, path_match, path_season, shot="basic")

    ## col-list
    col_list = ['location', 'shot_statsbomb_xg', 'player_name', "comp_name", 'shot_outcome_name', 'shot_body_part_name', 'shot_type_name']

    ## shot-dataframe from event-dataframe
    shot_df = event_df.loc[:, col_list]

    ## create body part column
    shot_df['body_part'] = shot_df['shot_body_part_name'].apply(body_part)

    ## create target column - 2 classes - goal and no goal
    shot_df['target'] = shot_df['shot_outcome_name'].apply(goal)

    ## drop shot_outcome_name and shot_body_part_name column
    shot_df.drop(['shot_outcome_name', 'shot_body_part_name'], axis=1, inplace=True)

    ## filter out shots from penalties, corners and Kick Off
    shot_df = shot_df.loc[ 
        (shot_df["shot_type_name"] != "Penalty") &
        (shot_df["shot_type_name"] != "Corner")  &
        (shot_df["shot_type_name"] != "Kick Off")
    ]

    ## add x and y coordinate columns
    shot_df['x'] = shot_df['location'].apply(coordinates_x)
    shot_df['y'] = shot_df['location'].apply(coordinates_y)

    ## drop location column
    shot_df.drop('location', inplace=True, axis=1)

    ## save the dataset
    shot_df.to_pickle(f'{path_save}/{filename}')

def intermediate_dataset(df, adv=False):
    """
    Function for making dataframe for intermediate model(containing shots info).
    
    Args:
        df (pandas.DataFrame): required dataframe.
        adv (bool, optional): for preparing advanced dataset.
    
    Returns:
        pandas.DataFrame: dataframe for intermediate model
    """
    ## init an empty dictionary
    if adv == True:
        main_dict = {
            'x' : [], 'y': [],
            "shot_type_name": [], "shot_body_part_name": [],
            "player_name": [], "shot_statsbomb_xg": [], 
            "pass_type": [], "open_goal": [],
            "under_pressure": [], "deflected": [], "player_in_between": [],
            "goal_keeper_angle": [], "target": []
        }
    else:
        main_dict = {
            'x' : [], 'y': [],
            "shot_type_name": [], "shot_body_part_name": [],
            "player_name": [], "shot_statsbomb_xg": [], 
            "pass_type": [], "open_goal": [],
            "under_pressure": [], "deflected": [],  "target": []
        }
    
    ## fetch shots from the dataframe
    shot_df = df.loc[
        df["type_name"] == "Shot"
    ].copy()
    
    
    ## fetch key-pass and assists from the dataframe 
    try:
        pass_df = df.loc[
            (df["pass_shot_assist"] == True) |
            (df["pass_goal_assist"] == True)
        ].copy().set_index("id")
    except KeyError:
        pass_df = df.loc[
            (df["pass_shot_assist"] == True)
        ].copy().set_index("id")
    
    for _, data in shot_df.iterrows():
        ## ignore shots from penalties, corners and Kick Off
        if (data["shot_type_name"] == "Penalty") or\
           (data["shot_type_name"] == "Corner") or\
           (data["shot_type_name"] == "Kick Off"):
            continue
            
        ## fetch shot location
        location = data["location"]

        ## get x and y coordinates
        x = coordinates_x(location)
        y = coordinates_y(location)

        if adv == True:
            ## fetch freeze frame
            freeze_frame = data["shot_freeze_frame"]
            
            ## calculate freeze-frame-variables
            count_teammate, count_opponent, goal_keeper_angle, dis_goal_keeper, dis_shot_keeper = freeze_frame_vars(
                freeze_frame, x, y
            )

            ## append info to main-dict for advanced features
            main_dict["player_in_between"].append(count_teammate + count_opponent)
            main_dict["goal_keeper_angle"].append(goal_keeper_angle) 

        ## fetch shot_type_name
        shot_type_name = data["shot_type_name"]
        
        ## fetch shot_outcome_name
        if data["shot_outcome_name"] == "Goal":
            target = 1
        else:
            target = 0
            
        ## fetch shot_body_part_name
        if data["shot_body_part_name"] == "Right Foot":
            body_part = "Foot"
        elif data["shot_body_part_name"] == "Left Foot":
            body_part = "Foot"
        else:
            body_part = data["shot_body_part_name"]
            
        ## fetch player name
        player_name = data["player_name"]
        
        ## fetch statsbomb xG
        stats_xg = data["shot_statsbomb_xg"]
        
        try:
            ## fetch open_goal
            if pd.isna(data["shot_open_goal"]):
                open_goal = 0
            else:
                open_goal = 1
        except Exception:
            open_goal = 0
        
        ## fetch under-pressure
        if pd.isna(data["under_pressure"]):
            pressure = 0
        elif data["under_pressure"] == True:
            pressure = 1
            
        ## fetch deflected
        try:
            if pd.isna(data["shot_deflected"]):
                deflected = 0
            elif data["shot_deflected"] == True:
                deflected = 1
        except Exception:
            deflected = 0
        
        ## is-assisted by a pass or not
        if pd.isna(data["shot_key_pass_id"]):
            pass_type = "Not Assisted"
        else:
            ## fetch key pass id
            key_pass_id = data["shot_key_pass_id"]
            
            ## fetch data-row of the key pass
            temp_data = pass_df.loc[key_pass_id]

            ## init pass_type
            pass_type = ""
            
            ## fetch through balls
            try:
                if temp_data["pass_technique_name"] == "Through Ball":
                    pass_type = "Through Ball"
            except Exception:
                pass

            ## fetch cutbacks
            try:
                if temp_data["pass_cut_back"] == True:
                    pass_type = "Cut Back"
            except Exception:
                pass
            
            ## fetch cross
            try:
                if temp_data["pass_cross"] == True:
                    pass_type = "Cross"
            except Exception:
                pass
            
            if pass_type == "":
                # fetch pass_type_name
                if temp_data["pass_type_name"] == "Corner":
                    pass_type = "From Corner"
                elif temp_data["pass_type_name"] == "Free Kick":
                    pass_type = "From Free Kick"
                else:
                    pass_type = "Other"
            
    
        ## append to dict
        main_dict['x'].append(x)
        main_dict['y'].append(y)
        main_dict["shot_type_name"].append(shot_type_name)
        main_dict["shot_body_part_name"].append(body_part)
        main_dict["player_name"].append(player_name)
        main_dict["shot_statsbomb_xg"].append(stats_xg)
        main_dict["pass_type"].append(pass_type)
        main_dict["open_goal"].append(open_goal)
        main_dict["under_pressure"].append(pressure)
        main_dict["deflected"].append(deflected)
        main_dict["target"].append(target)

    return pd.DataFrame(main_dict)

def make_train_test(path, path_save):
    '''
    Function for making and saving train and test data.

    Argument:
        path -- str, path where the shot data is stored.
        path_save -- str, path where the data will be stored.
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
            menwc_data,  
            pl_data,
            nwsl_data
        ]
    )

    ## make test dataframe
    test_df = pd.concat(
        [
            fawsl_data,
            wwc_data
        ]
    )

    ## randomly shuffle both the datasets
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    test_df = test_df.sample(frac=1).reset_index(drop=True)

    ## check for directory
    if os.path.isdir(path_save) == False:
        ## make directory
        os.mkdir(path_save)

    ## save train dataframe
    train_df.to_pickle(path_save+'/train_df.pkl')

    ## save test dataframe
    test_df.to_pickle(path_save+'/test_df.pkl')