# xG Model 

## Content

* [Directory Tree](#directory-tree)
* [Overview](#overview)
* [What is xG?](#what-is-xg)
* [How xG is used?](#how-xg-is-used)
* [xG Model](#xg-model)
  * [Basic Model](#basic-model)
  * [Intermediate Model](#intermediate-model)
  * [Advance Model](#advance-model)
* [Steps to run the script](#steps-to-run-the-script)
* [How to plot?](#how-to-plot)
* [Reviewing the basic model](#reviewing-the-basic-model)
* [Reviewing the intermediate model](#reviewing-the-intermediate-model)
* [Reviewing the advance model](#reviewing-the-advance-model)
 
## Directory Tree

```
.
├── input
│   ├── advance_dataset
│   │   ├── all_competitions
│   │   ├── train_test_data
│   │   ├── train_test_data_encoded
│   │   ├── train_test_data_final
│   │   └── train_test_data_result
│   ├── basic_dataset
│   │   ├── all_competitions
│   │   ├── train_test_data
│   │   ├── train_test_data_encoded
│   │   ├── train_test_data_final
│   │   └── train_test_data_result
│   ├── intermediate_dataset
│   │   ├── all_competitions
│   │   ├── train_test_data
│   │   ├── train_test_data_encoded
│   │   ├── train_test_data_final
│   │   └── train_test_data_result
│   └── Statsbomb
├── models
│   ├── advance_models
│   ├── basic_models
│   └── intermediate_models
├── notebook
├── plots
│   ├── advance_model
│   ├── basic_model
│   ├── intermediate_model
│   └── real_values
└── src
```

* Directories from above tree:

  1. **input**: contains dataset used in to make the xG model.
  2. **models**: contains the trained model.
  3. **notebook**: contains jupyter notebook used for experimentations and plotting visualizations.
  4. **plots**: contains all the generated plots.
  5. **src**: contains all the python files used for making xG model.
  
* **NOTE**: Before executing the commands to run the scripts, beware that some of the Statsbomb's data files **are empty** (or was during the project development) make sure to check them out. These files are *data/matches/16/76.json*, *data/matches/16/42.json* and *data/matches/37/90.json* also remove lines from *competition.json* for (*comp-id 16* *match-id 76 and 42*) and (*comp-id 37* *match-id 90*).

## Overview

* Expected goals metric is one of the widely used metrics in football analytics field. It allows us to evaluate team and player performance.

* In a low-scoring game such as football, final match score does not provide a clear picture of the performance. This is why more and more sports analytics turn to the advanced models like xG, which is a statistical measure of the quality of chances created and conceded.

* Here, in this project I have tried to build an xG Model from scratch using machine learning algorithms.

## What is xG?

* In layman's term, xG (or expected goals) is the **probability** that a shot will result in a goal based on the characteristics of that shot.

* Some of these characteristics/variables include:

  * **Location of shooter:** How far was it from the goal and at what angle on the pitch?
  
  * **Body part:** Was it a header or off the shooter's foot?
  
  * **Type of pass:** Was it from a through ball, cross, set piece, etc?
  
* Every shot is compared to thousands of shots with similar characteristics to determine the probability that this shot will result in a goal. That probability is the expected goal total. 

* An xG of `0` is a *certain miss*, while an xG of `1` is a *certain goal*. 

* An xG of `0.5` would indicate that if identical shots were attempted `10` times, `5` would be expected to result in a goal.  

* Check out [this](https://www.youtube.com/watch?v=w7zPZsLGK18) youtube video by Opta to know more.

* **Note:** xG does not take into account the quality of player(s) involved in a particular play. It is an estimate of how the average player or team would perform in a similar situation.

## How xG is used?

* xG has many uses. Some examples are:

* Comparing xG to actual goals scored can indicate a player's shooting ability or luck. A player who consistently scores more goals than their total xG probably has an above average shooting/finishing ability.

* A team's xG differential (xG minus xG allowed) can indicate how a team should be performing. A negative goal differential but a positive xG differential might indicate a team has experienced poor luck or has below average finishing ability.

* xG can be used to assess a team's abilities in various situations, such as open play, from a free kick, corner kick, etc. For example, a team that has allowed more goals from free kicks than their xGA from free kicks is probably below average at defending these set pieces.

* A team's xGA (xG allowed) can indicate a team's ability to prevent scoring chances. A team that limits their opponent's shots and more importantly, limits their ability to take high probability shots will have a lower xGA.

## xG Model

* In this GitHub repository you will find my take on making xG model from scratch.

* I have used [Statsbomb's](https://twitter.com/StatsBomb) [open-data](https://github.com/statsbomb/open-data) for training and testing my models.

* **Why Statsbomb?**

  * *Statsbomb* is a leading football analytics company and provides data that is superior in quality.
  
  * Their event data accurately provides player locations for shots and measures pressure events on the pitch.
  
  * Their event data also shows location of all players on the pitch, including the Goal Keeper in any shot. (I have used this to make an advance xG model, that incorporates the positional information of players while a shot is being taken)
  
  * To know more about Statsbomb, visit [this](https://statsbomb.com/).
  
    ![statsbomb](https://user-images.githubusercontent.com/33928040/100495379-282c7600-3171-11eb-8a07-5c1127caafdb.png)

* I thought of creating three xG models:
  
  1. [Basic Model](#simple-model)
  2. [Intermediate Model](#intermediate-model)
  3. [Advance Model](#advanced-model)
  
* **NOTE**: I have only included shots from **Open Play** and **Free Kick** for training and testing the models.

* The very first step is to add Statsbomb's open-data to *input* directory. Refers [this](https://github.com/Slothfulwave612/xG-Model/tree/master/input/Statsbomb) to see how to add the Statsbomb's open-data. Also create *models* directory before running the scripts.

### Basic Model

* As the name says, I have used basic attributes for this model, attributes that can be available easily from any data provider.

* Following are the attributes used in making this model: 
  
  * **shot_type_name:** Was the shot from *open play* or *free kick*?
  
  * **body_part:** Was the shot taken by *foot*, *head* or *any other body part*?
  
  * **distance:** The distance of shot-location from goal.
  
  * **angle:** At what angle on the pitch was the shot taken from?
  
* **distance** and **angle** were extracted from **location** attribute using the required formulae.

### Intermediate Model

* This model has all the attributes of *basic model* and also the following additional attributes:
 
  * **pass_type**: Whether the pass received from *cross* or *through ball* or *corner* or *free kick* or *cut back* or *other*(a simple pass) or *not assisted*.
  
  * **open_goal**: Whether we have open-goal situation or not.
  
  * **under_pressure**: Whether the shot was taken under-pressure.
  
  * **deflected**: Whether the shot was deflected or not.
  
### Advance Model

* The advance model has all the attributes of *intermediate model* and also the following additional attributes:
 
  * **player_in_between**: Number of players between the shot-location and goal.
  
  * **goal_keeper_angle**: The goalkeeper covering angle.
  
* Statsbomb's open-data contains event-data for many competitions like all Messi La Liga matches since his debut till 2019-20 season, world cup 2018 matches, Arsenal invincible season data, FAWSL data for two seasons etc.

* I have created two datasets using the open-data - **train-data** and **test-data**.
  
  * *train-data* is used for training our model.
  
  * *test-data* for the evaluation process.
  
* Our *train-data* contains shot-data from the following competitions:
  
  * **La Liga** (in which Messi played since his debut till 2019-20 season).
  * **All UCL Finals** (since 2000 to 2019)
  * **World Cup** (2018, Men)
  * **Arsenal Invincible Season** (Arsenal's Premier League matches)
  * **National Women's Soccer League** (2018 season)
  
* Our *test-data* contains shot-data from the following competitions:  
  
  * **FA Women's Super League** (2018-19 and 2019-20 season)
  * **World Cup** (2019, Women)
  
* The machine learning algorithms implemented for making the models are: **Logistic Regression**, **Random Forest** and **xG Boost**(eXtreme Gradient Boosting).

## Steps to run the script

* To execute all the scripts in one go I have created **run.sh** file, on executing it will create the required datasets, train and test the model and save the required model and corresponding categorical features.

* The syntax to execute **run.sh** script is as follows:

```console
sh run.sh "model_type" "algo_name" "scale_type" "data" "encode"
```

* Let's take a quick look at the syntax:
 
 1. **sh run.sh**: command to execute a shell script, all the arguments after this command are command-line-arguments.
 2. **"model_type"**: refers to the type of model used, either "basic" or "intermediate" or "advance".
 3. **"algo_name"**: refers to which algorithm to implement, either "log_regg" or "random_forest" or "xg_boost".
 4. **"scale_type"**: refers to scaling type needed by the algorithm, either "std" for StandardScaler or "nrm" for Normalization or "none" if scaling is not required.
 5. **"data"**: refers to whether to make dataset or not, either "True" to make the dataset or "False" to not make the dataset.
 6. **"encode"**: refers to whether to encode the string values or not, either "True" or "False".
 
* **Note**: Before executing the below command make sure you have created a *models* directory where the trained model will be saved.
 
* Now to excute the script for *basic model* (using logistic regression) run the following command (in your terminal):
 
```console
sh run.sh "basic" "log_regg" "std" "True" "True"
```

* Let's run through the command once again:

 * The first two arguments are simple to understand, we are creating a "basic" model and using *logistic regression* as our algorithm.
 
 * Since algorithms like logistic regression requires scaling their attributes before training so we have used "std" as our scaling approach. One can also use "nrm" but the results will be quite same.
 
 * The last two arguments are "True" as we need to make the datasets for our basic model (which is why data is "True") and we also need to encode our string features that we have in our data (which is why encode is "True"). We will be using *One-hot encoding* in case of logistic-regression.
 
* After executing the command you will see the following output on your screen:
![1](https://user-images.githubusercontent.com/33928040/100496546-fae4c580-317a-11eb-9fc9-66a3275b7287.png)
![2](https://user-images.githubusercontent.com/33928040/100496547-fcae8900-317a-11eb-9ce6-8bfc18b86d11.png)

* After executing you will find inside *input* directory a *basic_dataset* directory has been created containing all required datasets, and also inside *models* directory you will find *basic_models* directory where the required model and its corresponsing categorical features has been saved.

* Similarly for *random_forest* use the following command:

```console
sh run.sh "basic" "random_forest" "none" "False" "True"

// "none" because random-forest do not require any scaling
// "False" because data has already been made in logistic-regression process (present inside input/basic_dataset/all_competitions), if not made pass "True"
// "True" because we need to have LabelEncoding in case of random-forest
```

* For *xg_boost* use the following command:

```console
sh run.sh "basic" "xg_boost" "none" "False" "False"

// "none" because xg-boost do not require any scaling
// "False" because data has already been made in logistic-regression process (present inside input/basic_dataset/all_competitions), if not made pass "True"
// "False" because LabelEncoded dataset has already been present (inside input/basic_dataset/train_test_data_final) as we have ran random-forest, if not pass "True"
```

* You can run also for *intermediate* or *advance* model by providing "intermediate" or "advance" respectively instead of "basic", but make sure to pass the right command line arguments after it.

## How to plot?

* In order to review our basic model(or any model) we will create plots and an aggregated xG table to see how our model is performing.

* So first we have to load the data. Go to *notebook/explore_experiment.ipynb* navigate to *load the data* (after importing the modules and files) cell and from there load the required dataset (you will load the datasets present inside *input/basic_dataset/train_test_data_result*)
  
  * Six datasets are present inside *train_test_data_result* having train and test data for all three algorithms used.
  * Place the right path and excute the cell.
  
* Then we have to plot Statsbomb xG values. These are the values from Statsbomb xG model and we will be comparing these values to our values to see how our model performs.
  
  * Execute the cell by providing right path (make sure *plots* directory is present) where you are saving the plot.
  * After running the cell you will find the plots have been saved to the path you provided.
 
* Then we will plot values predicted on training dataset. Execute the cell in *training plots* by providing the right path.

* After that we will plot values predicted on test dataset. Execute the cell in *test plots* by providing the right path.

* And finally we will plot the aggregated xG values. Execute the cells in *aggregated xG values* by providing the right path. 

* One can run similarly plot for other models as well (i.e. for *intermediate* and *advance* model) but make sure to run or have the models for *intermediate* or *advance* before plotting for the same.
  
## Reviewing the basic model

* To review the basic model for all three machine learning algorithms used I made a table for top 25 players with the most number of goals for both train and test dataset.

* Let's see how the table looks like for the train dataset([link](https://github.com/Slothfulwave612/xG-Model/blob/master/plots/basic_model/train_simple.png)):

  ![train_basic_table](https://github.com/Slothfulwave612/xG-Model/blob/master/plots/basic_model/train_simple.png)
  
* Table for test dataset([link](https://github.com/Slothfulwave612/xG-Model/blob/master/plots/basic_model/test_simple.png))  :
  
  ![test_basic_table](https://github.com/Slothfulwave612/xG-Model/blob/master/plots/basic_model/test_simple.png)
  
* To see all the plots visit [this](https://github.com/Slothfulwave612/xG-Model/tree/master/plots/basic_model).

## Reviewing the intermediate model

* Table for train dataset([link](https://github.com/Slothfulwave612/xG-Model/blob/master/plots/intermediate_model/train_simple.png)):

  ![train_basic_table](https://github.com/Slothfulwave612/xG-Model/blob/master/plots/intermediate_model/train_simple.png)
  
* Table for test dataset([link](https://github.com/Slothfulwave612/xG-Model/blob/master/plots/intermediate_model/test_simple.png))  :
  
  ![test_basic_table](https://github.com/Slothfulwave612/xG-Model/blob/master/plots/intermediate_model/test_simple.png)
  
* To see all the plots visit [this](https://github.com/Slothfulwave612/xG-Model/tree/master/plots/intermediate_model).

## Reviewing the advance model

* Table for train dataset([link](https://github.com/Slothfulwave612/xG-Model/blob/master/plots/advance_model/train_simple.png)):

  ![train_basic_table](https://github.com/Slothfulwave612/xG-Model/blob/master/plots/advance_model/train_simple.png)
  
* Table for test dataset([link](https://github.com/Slothfulwave612/xG-Model/blob/master/plots/advance_model/test_simple.png))  :
  
  ![test_basic_table](https://github.com/Slothfulwave612/xG-Model/blob/master/plots/advance_model/test_simple.png)
  
* To see all the plots visit [this](https://github.com/Slothfulwave612/xG-Model/tree/master/plots/advance_model1).
