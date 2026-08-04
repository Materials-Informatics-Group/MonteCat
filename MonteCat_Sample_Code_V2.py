## ====================================================================================================================================================================================
## Author: < Fernando García Escobar and Keisuke Takahashi>. Hokkaido University 2023-2026
## Licence: GNU General Public License (GPL) 3.0
## Description: < This script follows the proposed MonteCat algorithm that constructs a Regression Model from a big pool of engineered Descriptors (Features) through an adaptation of
## the Metropolis-Hastings algorithm. The number of iterations and the Temperature modulating the Acceptance Probability are determined by the user. This updated version enables 
## handling feature families created by engineering 'base features' to avoid feature redundancy throughout the feature search. Result log files are updated periodically for individual 
## runs with a specific random seed, and global result files (across all tested seeds) are output at the end, with the sampled feature subsets and score metrics for all runs. >
## ====================================================================================================================================================================================
## Library imports & configurations ===================================================================================================================================================

import os
import pandas as pd
import numpy as np
from statistics import mean
import random
import re

from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

## ====================================================================================================================================================================================
# General Script Parameters (Editable by the user) ====================================================================================================================================

model_to_test = 'Linear'   # Regression model to use (Choose between 'Linear', 'SVR' or 'RF')

C_value = 10               # Hyperparameter value for the SVR model rbf kernel
gamma_value = 0.01         # Hyperparameter value for the SVR model rbf kernel
iterations = 100           # Number of steps/iterations of the algorithm
kB = 0.00008617333262      # Boltzmann constant in eV/K units
temperature = 1            # Temperature parameter used to tune the Acceptance Probability curve behavior
#seed_value = 0            # For reproducibility
#random.seed(seed_value)   # For reproducibility

model_dictionary = {'SVR': SVR(kernel = 'rbf', C = C_value, gamma = gamma_value),
                    'Linear': LinearRegression(),
                    'RF': RandomForestRegressor(n_estimators = 100, random_state = 0)
                    }

model_tested = model_dictionary[model_to_test]

# Output Filename (can be freely changed by the user).

output_filename = 'SampleOutput'

## ====================================================================================================================================================================================
# Data Management =====================================================================================================================================================================

current_directory = os.getcwd() + '/'

df = pd.read_csv(current_directory + 'Dataset_V2.csv')

# Loading auxiliary base features' list to generate the reference feature dictionary

base_feature_list = pd.read_csv(current_directory + f'Base_Features_V2.csv')['Descriptor'].to_list()

zero_order_suffixes = ['', '_min', '_max', '_avg', '_var', '_hmean', '_gmean', '_wtmean', '_wtvar', '_wthmean', '_wtgmean']

# Descriptor Columns

features_bank = df.iloc[:,:-1].columns.tolist()
features_in_model = []

# Output Filename (can be freely changed by the user)

output_filename_stem = f'SampleOutput_{model_to_test}_T{temperature}'

# Update current seed output log file every x runs

output_index_constant = 10

global_result_filename = f'Global_{output_filename_stem}_Summary'
global_features_filename = f'Global_{output_filename_stem}_Feature_List'

global_log_data = []
global_feature_subsets = {}

## ====================================================================================================================================================================================
# Custom functions ====================================================================================================================================================================

"""
'create_feature_reference_dictionary' creates a dictionary with the base Descriptor names (features/properties) as the keys, and all derived analogues (zeroth and first order features) 
stored as lists to facilitate handling feature family addition, removals and swaps from the feature bank and model during execution.
"""

def create_feature_reference_dictionary(base_feature_list, dataset_features, zero_order_suffixes):
    suffix_pattern = '|'.join([re.escape(s) for s in zero_order_suffixes])
    reference_dictionary = {}
    family_list = []
    for base_feature in base_feature_list:
        pattern = rf'\b{re.escape(base_feature)}({suffix_pattern})\b'
        family_members = [x for x in dataset_features if re.search(pattern, x)]
        if family_members:
            reference_dictionary[base_feature] = family_members
    return reference_dictionary

"""
'train_model' is the basic function to train a regression model and return the test data's r2 score, mean absolute error and root mean squared error averaged across 10 random data 
splits.
"""

def train_model(feature_array, target, model):
    results = {
               'Score_List': [],
               'RMSE_List': [], 
               'MAE_List': []
              }
    for j in range(10):
        X_train, X_test, y_train, y_test = train_test_split(feature_array, target, test_size = 0.2, random_state = j)
        model.fit(X_train, y_train)
        results['Score_List'].append(model.score(X_test, y_test))
        train_y = []
        for i, j in zip(X_train, y_train):
            train_y.append(model.predict([i])[0])
        test_y = []
        for i, j in zip(X_test, y_test):
            test_y.append(model.predict([i])[0])
        results['RMSE_List'].append(root_mean_squared_error(y_test, test_y))
        results['MAE_List'].append(mean_absolute_error(y_test, test_y))
        training_score = mean(results['Score_List']) if len(results['Score_List']) > 0 else 0
        training_mae = mean(results['RMSE_List']) if len(results['Score_List']) > 0 else np.inf
        training_rmse = mean(results['MAE_List']) if len(results['Score_List']) > 0 else np.inf
    results.update({'Score': training_score, 'RMSE': training_mae, 'MAE': training_rmse})
    return results

"""
'greedy_addition' is the basic building block in a greedy Sequential Feature Addition process. It tests all features not added to a regression model and adds the one that improves 
the score metric the most. In MonteCatV2, however, entire descriptor families stemming from the same base feature are removed from the remaining available Descriptors in the bank, 
drastically reducing the number of features to test.
"""

def greedy_addition(df, feature_dict, model):
    best_score = -np.inf
    best_mae = np.inf
    best_rmse = np.inf
    best_family = None
    best_feature = None
    for family_to_test, feature_family in feature_dict.items():
        for feature_to_use in feature_family:
            train_results = train_model(np.array(df[[feature_to_use]]), df.iloc[:, -1], model)
            if train_results['Score'] > best_score:
                best_score = train_results['Score']
                best_mae = train_results['MAE']
                best_rmse = train_results['RMSE']
                best_family = family_to_test
                best_feature = feature_to_use
    result = {
              'Base': best_family, 
              'Feature': best_feature, 
              'Score': best_score, 
              'MAE': best_mae,
              'RMSE': best_rmse,
              'Type': 'Addition'
              }
    return result

"""
'random_addition' is used in making random addition proposals, where one random feature from the bank is added andthe model's Score is calculated.
"""

def random_addition(df, model_state, available_families, feature_dict, model):
    if not available_families: return None
    selected_family = random.choice(list(available_families))
    added_feature = random.choice(feature_dict[selected_family]) 
    proposal_state = model_state.copy()
    proposal_state[selected_family] = added_feature
    features_to_use = list(proposal_state.values())
    train_results = train_model(df[features_to_use].values, df.iloc[:, -1], model)
    result = {
              'State': proposal_state, 
              'Base': selected_family, 
              'Feature': added_feature, 
              'Score': train_results['Score'], 
              'MAE': train_results['MAE'],
              'RMSE': train_results['RMSE'],
              'Type': 'Addition'
              }
    return result

"""
'random_removal' is used in making random removal proposals, where one random feature from the model is withdrawn and the model's Score is calculated.
"""

def random_removal(df, model_state, model):
    if len(model_state) <= 1: return None
    selected_family = random.choice(list(model_state.keys()))
    proposal_state = model_state.copy()
    removed_feature = proposal_state.pop(selected_family)
    features_to_use = list(proposal_state.values())
    train_results = train_model(df[features_to_use].values, df.iloc[:, -1], model)
    result = {
              'State': proposal_state, 
              'Base': selected_family, 
              'Feature': removed_feature, 
              'Score': train_results['Score'], 
              'MAE': train_results['MAE'],
              'RMSE': train_results['RMSE'], 
              'Type': 'Removal'
              }
    return result

"""
'random_swap_same' is used in making random feature swaps, where a random feature in the model is selected, swapped with another feature from the same family and the model's Score 
is calculated.
"""

def random_swap_same(df, model_state, feature_dict, model):
    if not model_state: return None
    selected_family = random.choice(list(model_state.keys()))
    current_analogue = model_state[selected_family]
    available_analogues = [x for x in feature_dict[selected_family] if x != current_analogue]
    if not available_analogues: return None
    proposed_analogue = random.choice(available_analogues)
    proposal_state = model_state.copy()
    proposal_state[selected_family] = proposed_analogue
    features_to_use = list(proposal_state.values())
    train_results = train_model(df[features_to_use].values, df.iloc[:, -1], model)
    result = {
              'State': proposal_state, 
              'Base': selected_family, 
              'Feature': proposed_analogue, 
              'Swap_Out': current_analogue, 
              'Score': train_results['Score'], 
              'MAE': train_results['MAE'],
              'RMSE': train_results['RMSE'],
              'Type': 'Swap_Same'
              }
    return result

"""
'accept_proposal' is called when the tentative proposal increases the Score, or does not decrease it. Since this is a direct acceptance of the outcome, the Acceptance Probability 
Value and the Test value are not calculated. The function then updates the dictionary where each iteration's results are stored.
"""
def accept_proposal(proposal_to_adopt, model_state, available_families):
    move_type = proposal_to_adopt['Type']
    base_feature_proposal = proposal_to_adopt['Base']
    if move_type == 'Removal':
        del model_state[base_feature_proposal]
        available_families.add(base_feature_proposal)
    else:
        model_state[base_feature_proposal] = proposal_to_adopt['Feature']
        if move_type == 'Addition':
            available_families.remove(base_feature_proposal)
    return model_state, available_families

## ====================================================================================================================================================================================
# Main script =========================================================================================================================================================================

feature_reference_dict = create_feature_reference_dictionary(base_feature_list, df.iloc[:,: -1].columns.to_list(), zero_order_suffixes)

for seed_value in range(0, 50):

    output_filename = f'{output_filename_stem}_{seed_value}'

    random.seed(seed_value)

    counter = 1

    # Local State Variables

    model_state = {}
    available_families = set(feature_reference_dict.keys())
    current_score = 0.0
    log_data = []

    # Global State Variables

    best_global_score = -np.inf
    best_global_mae = np.inf
    best_global_rmse = np.inf
    best_global_state = {}
    best_global_features = np.inf

    # First iteration, where the first addition is always Greedy

    greedy_addition_result = greedy_addition(df, feature_reference_dict, model_tested)

    model_state[greedy_addition_result['Base']] = greedy_addition_result['Feature']
    available_families.remove(greedy_addition_result['Base'])
    current_score = greedy_addition_result['Score']
    current_mae = greedy_addition_result['MAE']
    current_rmse = greedy_addition_result['RMSE']
    event = greedy_addition_result['Type']
    event_type = f'Direct_{event}'

    # First ever output generation (with column headers); all further are updates

    round_entry = {
                'Iteration': counter,
                'D_Number': len(model_state),
                'Score': round(current_score, 5),
                'MAE': round(current_mae, 5),
                'RMSE': round(current_rmse, 5),
                'Base': greedy_addition_result['Base'],
                'Descriptor': greedy_addition_result['Feature'],
                'Event': event_type,
            }

    log_data.append(round_entry)

    counter += 1

    for i in range(iterations - 1):

        addition_result = random_addition(df, model_state, available_families, feature_reference_dict, model_tested)
        removal_result = random_removal(df, model_state, model_tested)
        same_swap_result = random_swap_same(df, model_state, feature_reference_dict, model_tested)

        valid_proposals = [x for x in [addition_result, removal_result, same_swap_result] if x is not None]

        best_proposal = max(valid_proposals, key = lambda x: x['Score'])
        proposal_score = best_proposal['Score']

        accept_condition = False
        event = best_proposal['Type']

        if proposal_score > current_score:
                accept_condition = True
                event_type = f'Direct_{event}'
        else:
            acceptance_probability = np.exp((proposal_score - current_score) / (kB*temperature))
            test_p = np.random.uniform()
            if acceptance_probability > test_p:
                accept_condition = True
                event_type = f'Conditional_{event}'
            else:
                accept_condition = False
                event_type = f'{event}_Rejection'

        if accept_condition:
            model_state, available_families = accept_proposal(best_proposal, model_state, available_families)
            current_score = proposal_score
            current_mae = best_proposal['MAE']
            current_rmse = best_proposal['RMSE']

        best_score_condition = current_score > best_global_score
        feature_number_condition = (current_score == best_global_score and len(model_state) < best_global_features)

        if best_score_condition or feature_number_condition:
            best_global_score = current_score
            best_global_mae = current_mae
            best_global_rmse = current_rmse
            best_global_state = model_state.copy()
            best_global_features = int(len(model_state))

        round_entry = {
                'Iteration': counter,
                'D_Number': len(model_state),
                'Score': round(current_score, 5),
                'MAE': round(current_mae, 5),
                'RMSE': round(current_rmse, 5),
                'Base': best_proposal['Base'],
                'Descriptor': best_proposal['Feature'],
                'Event': event_type,
            }
        log_data.append(round_entry)

# Periodic result output every 'output_index_constant' iterations

        if (counter % output_index_constant == 0):
            pd.DataFrame(log_data).to_csv(current_directory + f'{output_filename}.csv', index = False)

        counter += 1

    pd.DataFrame(log_data).to_csv(current_directory + f'{output_filename}.csv', index = False)

    run_entry = {
                'Model': model_to_test,
                'Temperature': int(temperature),
                'Output': int(seed_value),
                'Score': round(best_global_score, 5),
                'Features': best_global_features
            }

    global_log_data.append(run_entry)
    summary_df = pd.DataFrame(global_log_data)
    summary_df.to_csv(current_directory + f'{global_result_filename}.csv', index = False)

    feature_list = sorted(list(best_global_state.values()), key = lambda x: df.iloc[:,:-1].columns.to_list().index(x))

    global_feature_subsets[f'{output_filename}'] = feature_list
    feature_list_df = pd.DataFrame({x: pd.Series(y) for x, y in global_feature_subsets.items()})
    feature_list_df.to_csv(current_directory + f'{global_features_filename}.csv', index = False)

## ====================================================================================================================================================================================
## ====================================================================================================================================================================================
