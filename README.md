# xG Model 

* **<---Project Under Development--->**

## Content

* [Overview](#overview)

* [What is xG](#what-is-xg)

* [How xG is used?](#how-xg-is-used)

* [xG Model](#xg-model)
  * [Simple Dataset](#simple-dataset)
    * [Steps to Run the script](#steps-to-run-the-script)
    * [Evaluating The Results](#evaluating-the-results)
  

## Overview

* Expected goals metric is one of the widely used metrics in football analytics field. It allows us to evaluate team and player performance.

* In a low-scoring game such as football, final match score does not provide a clear picture of performance. This is why more and more sports analytics turn to the advanced models like xG, which is a statistical measure of the quality of chances created and conceded.

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

* Check out [this](https://www.youtube.com/watch?v=w7zPZsLGK18) to know more.

* **Note:** xG does not take into account the quality of player(s) involved in a particular play. It is an estimate of how the average player or team would perform in a similar situation.

## How xG is used?

* xG has many uses. Some examples are:

* Comparing xG to actual goals scored can indicate a player's shooting ability or luck. A player who consistently scores more goals than their total xG probably has an above average shooting/finishing ability.

* A team's xG differential (xG minus xG allowed) can indicate how a team should be performing. A negative goal differential but a positive xG differential might indicate a team has experienced poor luck or has below average finishing ability.

* xG can be used to assess a team's abilities in various situations, such as open play, from a free kick, corner kick, etc. For example, a team that has allowed more goals from free kicks than their xGA from free kicks is probably below average at defending these set pieces.

* A team's xGA (xG allowed) can indicate a team's ability to prevent scoring chances. A team that limits their opponent's shots and more importantly, limits their ability to take high probability shots will have a lower xGA.

## xG Model

* So here I have created my own xG model from scratch.

* I have used [Statsbomb's](https://twitter.com/StatsBomb) [open-data](https://github.com/statsbomb/open-data) for training and testing my models.

* I thought of creating xG model on three types of datasets:
  1. [Simple Dataset](#simple-dataset)
  2. Intermediate Dataset(#intermediate-dataset)
  3. Advanced Dataset(#advanced-dataset)
  
* Now I'll be explaining all the datasets one by one with the model and their performance in each one of them.

### Simple Dataset

* As the name says, I have used simple attributes for the dataset.

* These are the following attributes used:
  
  * **location:** The location on the pitch from where the shot was taken.
  
  * **shot_type_name:** Was the shot from *open play*, *free kick*, *penalty*, *corner* or *kick off*?
  
  * **body_part:** Was the shot taken by *foot*, *head* or *any other body part*?
  
  * **angle:** At what angle on the pitch was the shot taken from?
  
* Statsbomb open-data contains event-data for many competitions like all Messi La Liga matches since his debut till 2018-19 season, world cup 2018 matches, Arsenal invincible season data, FAWSL data for two seasons etc.

* I have created two datasets using the open-data - **train-data** and **test-data**.
  
  * *train-data* is used for training our model.
  
  * *test-data* for the evaluation process.
  
* Our *train-data* contains shot-data from the competitions:
  
  * **La Liga** (in which Messi played since his debut till 2018-19 season).
  * **All UCL Finals** (since 2000 to 2019)
  * **World Cup** (2018, Men)
  * **Arsenal Invincible Season** (Only premier league matches)
  * **National Women's Soccer League** (2018 season)
  
* Our *test-data* contains shot-data from the following competitions:  
  
  * **FA Women's Super League** (2018-19 and 2019-20 season)
  * **World Cup** (2019, Women)
  
* I have used only **logistic regression** and **random forests** for creating our models. (random-forests code will be added soon)
  
* Now let's see a step-by-step approach to run the scripts to create the dataset, train the model and test it on the test-data.

#### Steps to Run the script

* **Step 0:** We will start by choosing an evaluation metrics. For that we plotted a countplot for no-goals vs goals-scored.

* Here is the result:
  
  ![target_count](https://user-images.githubusercontent.com/33928040/90810906-595f0400-e341-11ea-95eb-9aa2c16ddc90.jpg)
  
* Since, we can see that the data is sweked, so we are going to use *AUC-ROC-Curve* as our evaluation metric.  

* **Step 01:** At first, we create separate shot-data files for each and every competitions.

* For this, you can run `python -m src.dataset` from your terminal, the command will exract all the shot data from the Statsbomb data and save the files in `all_competitions` directory which will be located at `input/simple_dataset`.

* The command will yield the following result in your terminal:
  
  ![dataset_simple](https://user-images.githubusercontent.com/33928040/90808054-126f0f80-e33d-11ea-9260-fa5bce759427.png)
  
* And the directory where the files are saved will look like this:
  
  ![all_comp](https://user-images.githubusercontent.com/33928040/90808131-2c105700-e33d-11ea-9617-a276950bdb01.png)
  
* **Step 02:** Since now we have all the shot-data for each and every competitons present with us, we can now create train and test sets.

* For this, you can run `python -m src.train_test_data` from your terminal, the command will create and save train and test data in `train_test_data` directory, hich will be located at `input/simple_dataset`.

* The newly created directory will look like this:
  
  ![train_test_data](https://user-images.githubusercontent.com/33928040/90808496-b8bb1500-e33d-11ea-86c1-06d5b5dc75bd.png)
  
* **Step 03:** Now is the time for encoding the categorical attributes in our datasets.

* If you are using a **logistic regression**, you have to run `python -m src.categorical`, this will do the required *ohe-hot encoding* to the categorical columns and the new dataset with encoded columns will be saved in `train_test_data_encoded` directory, located at `input/simple_dataset`.

* The newly created directory will look like this:
  
  ![ohe_simple](https://user-images.githubusercontent.com/33928040/90808828-3da62e80-e33e-11ea-9129-eb730ec52d33.png)
  
* **Step 04:** Now we will prepare our datasets for training and testing, we will drop the columns which are not required, and the newly created datasets will be saved in `train_test_data_final` directory.

* For this run `python -m src.prepare_dataset`.

* The newly created directory will look like this:
  
  ![final_ohe](https://user-images.githubusercontent.com/33928040/90809115-a1c8f280-e33e-11ea-9997-ccc507449e84.png)
  
* **Step 05:** Now is the time for training our model and using test-data for evaluations.

* You can run `sh run.sh log_regg` if you want to train **logistic regression model**.

* The script will fetch the datasets created using *Step 04*, will perform scaling to both the datasets, then will train the model using the train-dataset and make predictions on test-dataset. The script will output ROC scores for both the datasets and will save the datasets with prediction at `train_test_data_result`.

* Here the AUC-ROC score is:
  
  ![roc_log_regg](https://user-images.githubusercontent.com/33928040/90809758-99bd8280-e33f-11ea-9dc8-cc5e75120f50.png)
  
* And the newly created directory will look like this:
  
  ![preds_log_regg](https://user-images.githubusercontent.com/33928040/90809865-be195f00-e33f-11ea-837b-d426b43f1e10.png)
  
#### Evaluating The Results

* From the evaluation scores, *AUC-ROC-Score*, we can see that these scores are quite descent but still it do not tell us about how are model is performing.

* So, for that we can plot some visualizations and compare the result with the Statsbomb-xG. 

* After plotting and comparing the results look quite interesting.

* Here is the scatterplot for shots in train-dataset using Statsbomb-xG:
  
  ![train_data_plot](https://user-images.githubusercontent.com/33928040/90812930-7c3ee780-e344-11ea-94bd-1f82d86d0b6a.jpg)
  
* Here is the scatterplot for shots in train-dataset using xG predicted by out logistic-model:
  
  ![simple_train_pred_logregg_plot](https://user-images.githubusercontent.com/33928040/90814985-97f7bd00-e347-11ea-8079-812dbf9a3985.jpg)
  
* Our model is performing quite well here.  
  
* Since we have taken only limited parameter(distance and angle, ..) into consideration, we can see that the shots that were near to the goal have a high probability of getting scored and the shots that were far have a lower probability.

* But still we can see some errors here.

* Notice the one shot which is at the goal line. Statsbomb xG is predicting a high value and it should be high but our model is having a low probability for the same.

* Another thing to notice is for some shots that are someway far from goal, Statsbomb xG value is more than our predicted xG value, the reason is we are only taking angle and distance into consideration not the opponent player's position or other features. 

* We can do the same for test-data as well. The first visualization is for the Statsbomb xG value and second for the xG value predicted by our model.

  ![test_data_plot](https://user-images.githubusercontent.com/33928040/90815010-9f1ecb00-e347-11ea-8cea-d6d03eccf58d.jpg)
  
  ![simple_test_pred_logregg_plot](https://user-images.githubusercontent.com/33928040/90814995-9b8b4400-e347-11ea-8372-65c292d571ee.jpg)
  
* Now, we can do another thing to evaluate our model's performance.  

* What we can do here is, create a dataframe containing *player name*, *total shots taken*, *total goals*, *Statsbomb xG* and *predicted xG* and then we can compare how the model is performing in evaluating in total xG. 

* Here is the result, first for the train-dataset and second for the test-dataset:

  ![train_table_logregg](https://user-images.githubusercontent.com/33928040/90815016-a219bb80-e347-11ea-8b00-3cf453a53bf0.png)
  
  ![test_table_logregg](https://user-images.githubusercontent.com/33928040/90815013-a1812500-e347-11ea-938a-560c31cb22cc.png)

* From here we can conclude that on aggregate the predicted xG is very close to aggregate Statsbomb xG.

* The project is still under development and will be updated soon.
