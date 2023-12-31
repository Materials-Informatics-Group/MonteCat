## ====================================================================================================================================================================================
## Authors: < Fernando García Escobar and Keisuke Takahashi>. Hokkaido University 2023
## Licence: GNU General Public License (GPL) 3.0
## Articles: Fernando García Escobar, Toshiaki Taniike, Keisuke Takahashi, 
## "MonteCat - A Basin-hopping-inspired Catalyst Descriptor Search algorithm for Regression Models",Journal of Chemical Information and Modeling 2024 (Accepted)
## Description: < This script follows the proposed MonteCat algorithm that constructs a Regression Model from a big pool of engineered Descriptors (Features) through an adaptation of
## the Metropolis-Hastings algorithm. The number of iterations and the Temperature modulating the Acceptance Probability are determined by the user. >
## ====================================================================================================================================================================================
## Library imports & Configurations ===================================================================================================================================================

import pandas as pd
import numpy as np
from statistics import mean
import random

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

## ====================================================================================================================================================================================
# General Script Parameters (Editable by the user) ====================================================================================================================================

model_to_test = 'Linear'   # Regression model to use (Choose between 'Linear' and 'SVR')

C_value = 10               # Hyperparameter value for the SVR model rbf kernel
gamma_value = 0.01         # Hyperparameter value for the SVR model rbf kernel
iterations = 1000          # Number of steps/iterations of the algorithm
kB = 0.00008617333262      # Boltzmann constant in eV/K units
temperature = 5            # Temperature parameter used to tune the Acceptance Probability curve behavior
seed_value = 0             # For reproducibility
random.seed(seed_value)    # For reproducibility

model_dictionary = {'SVR': SVR(kernel = 'rbf', C = C_value, gamma = gamma_value),
                    'Linear': LinearRegression()
                    }

model_tested = model_dictionary[model_to_test]

# Output Filename (can be freely changed by the user).

output_filename = 'SampleOutput'

## ====================================================================================================================================================================================
# Dataset loading =====================================================================================================================================================================

df = pd.read_csv('SampleData.csv')

# Descriptor Columns

descriptors_bank = df.iloc[:,:-1].columns.tolist()
descriptors_in_model = []

## ====================================================================================================================================================================================
# Functions ===========================================================================================================================================================================

"""
'train_model' is the basic function to train a regression model and return the test data's r2 mean score after 10 random data splits. In case there is an error during training, 
conditional clauses are present to return a score of 0.
"""
def train_model(descriptors, target, model):
    placeholder_scores = []
    for j in range(10):
        X_train, X_test, y_train, y_test = train_test_split(descriptors, target, test_size = 0.2, random_state = j)
        try:
            model.fit(X_train, y_train.values.ravel())
            placeholder_scores.append(model.score(X_test, y_test))
        except:
            pass
    try:
        model_score = mean(placeholder_scores)
    except:
        model_score = 0
    return model_score

"""
'greedy_addition' tests all Descriptors not part of a regression model, and adds the one that increases the Score the most.
"""
def greedy_addition(df, descriptors_in_model, descriptors_bank, model, counter, result_dict):
    tested_descriptors = []
    placeholder_scores = []
    for i in descriptors_bank:
        descriptors_in_model.append(i)
        placeholder_scores.append(train_model(np.array(df[descriptors_in_model]), df.iloc[:, -1], model))
        tested_descriptors.append(i)
        descriptors_in_model.remove(i)
    best_score = max(placeholder_scores)
    best_descriptor = tested_descriptors[placeholder_scores.index(best_score)]
    descriptors_in_model.append(best_descriptor)
    descriptors_bank.remove(best_descriptor)
    result_dict['Iteration'].append(counter)
    result_dict['D_Number'].append(len(descriptors_in_model))
    result_dict['Current_Score'].append(best_score)
    result_dict['Descriptor'].append(best_descriptor)
    result_dict['Outcome'].append('Direct_Addition')
    result_dict['Acceptance_Probability'].append(0) 
    result_dict['Test_Probability'].append(0)
    return descriptors_in_model, descriptors_bank, result_dict

"""
'random_addition' is used in making random addition proposals, where one Descriptor from the bank is added and the model's Score is calculated.
"""
def random_addition(df, descriptors_in_model, descriptors_bank, model):
    chosen_descriptor = random.choice(descriptors_bank)
    descriptors_in_model.append(chosen_descriptor)
    model_score = train_model(np.array(df[descriptors_in_model]), df.iloc[:, -1], model)
    descriptors_in_model.remove(chosen_descriptor)
    proposal_result = {'Descriptor': chosen_descriptor,
                       'Proposal_Score': model_score}
    return proposal_result

"""
'random_removal' is used in making random removal proposals, where one Descriptor from the model is withrawn and the model's Score is calculated.
"""
def random_removal(df, descriptors_in_model, descriptors_bank, model):
    chosen_descriptor = random.choice(descriptors_in_model)
    descriptors_in_model.remove(chosen_descriptor)
    model_score = train_model(np.array(df[descriptors_in_model]), df.iloc[:, -1], model)
    descriptors_in_model.append(chosen_descriptor)
    proposal_result = {'Descriptor': chosen_descriptor,
                       'Proposal_Score': model_score}
    return proposal_result

"""
'direct_accept' is called when the tentative proposal increases the Score, or does not decrease it. Since this is a direct acceptance of the outcome, the Acceptance Probability Value 
and the Test value are not calculated. The function then updates the dictionary where each iteration's results are stored.
"""
def direct_accept(action_to_perform, counter, result_dict, proposal_result, descriptors_in_model, descriptors_bank):
    if action_to_perform == 'Addition':
        descriptors_in_model.append(proposal_result['Descriptor'])
        descriptors_bank.remove(proposal_result['Descriptor'])
    else:
        descriptors_in_model.remove(proposal_result['Descriptor'])
        descriptors_bank.append(proposal_result['Descriptor'])
    result_dict['Iteration'].append(counter)
    result_dict['D_Number'].append(len(descriptors_in_model))
    result_dict['Current_Score'].append(proposal_result['Proposal_Score'])
    result_dict['Descriptor'].append(proposal_result['Descriptor'])
    result_dict['Outcome'].append(f'Direct_{action_to_perform}')
    result_dict['Acceptance_Probability'].append(0) 
    result_dict['Test_Probability'].append(0)
    return descriptors_in_model, descriptors_bank, result_dict


"""
'conditional_accept' is called when the model's Score is lower than the previous round, but the probability test was cleared by the random draw of the tested value vs the Acceptance 
Value. The function then updates the dictionary where each iteration's results are stored.
"""
def conditional_accept(action_to_perform, counter, result_dict, proposal_result, descriptors_in_model, descriptors_bank, acceptance_probability, test_probability):
    if action_to_perform == 'Addition':
        descriptors_in_model.append(proposal_result['Descriptor'])
        descriptors_bank.remove(proposal_result['Descriptor'])
    else:
        descriptors_in_model.remove(proposal_result['Descriptor'])
        descriptors_bank.append(proposal_result['Descriptor'])
    result_dict['Iteration'].append(counter)
    result_dict['D_Number'].append(len(descriptors_in_model))
    result_dict['Current_Score'].append(proposal_result['Proposal_Score'])
    result_dict['Descriptor'].append(proposal_result['Descriptor'])
    result_dict['Outcome'].append(f'Conditional_{action_to_perform}')
    result_dict['Acceptance_Probability'].append(acceptance_probability) 
    result_dict['Test_Probability'].append(test_probability)
    return descriptors_in_model, descriptors_bank, result_dict

"""
'conditional_reject' is called when the model's Score is lower than the previous round and the probability test was not passed by the random draw of the tested value against the 
Acceptance Value. The function then updates the dictionary where each iteration's results are stored.
"""
def conditional_reject(action_to_perform, counter, result_dict, proposal_result, descriptors_in_model, descriptors_bank, acceptance_probability, test_probability):
    result_dict['Iteration'].append(counter)
    result_dict['D_Number'].append(len(descriptors_in_model))
    result_dict['Current_Score'].append(result_dict['Current_Score'][-1])
    result_dict['Descriptor'].append(proposal_result['Descriptor'])
    result_dict['Outcome'].append(f'{action_to_perform}_Rejection')
    result_dict['Acceptance_Probability'].append(acceptance_probability) 
    result_dict['Test_Probability'].append(test_probability)
    return descriptors_in_model, descriptors_bank, result_dict

"""
'result_compilation' takes out the latest results stored in the results dictionary (results_package) and prepares a one-row DataFrame for real-time output update.
"""
def result_compilation(result_dict):
    df_iteration = pd.DataFrame([result_dict['Iteration'][-1]], columns = ['Iteration'])
    df_number = pd.DataFrame([result_dict['D_Number'][-1]], columns = ['D_Number'])
    df_score = pd.DataFrame([round(result_dict['Current_Score'][-1], 5)], columns = ['Current_Score'])
    df_descriptor = pd.DataFrame([result_dict['Descriptor'][-1]], columns = ['Descriptor'])
    df_event = pd.DataFrame([result_dict['Outcome'][-1]], columns = ['Outcome'])
    df_accprob = pd.DataFrame([round(result_dict['Acceptance_Probability'][-1], 5)], columns = ['Acceptance_Probability'])
    df_testprob = pd.DataFrame([round(result_dict['Test_Probability'][-1], 5)], columns = ['Test_Probability'])
    dataframes = [df_iteration, df_number, df_score, df_descriptor, df_event, df_accprob, df_testprob]
    results = pd.concat(dataframes, axis = 1)
    return results

## ====================================================================================================================================================================================
# Main script =========================================================================================================================================================================
    
counter = 1

# result_dict is continuously updated and used to generate real-time output.

result_dict = {'Iteration': [], 
               'D_Number': [], 
               'Current_Score': [], 
               'Descriptor': [], 
               'Outcome': [], 
               'Acceptance_Probability': [], 
               'Test_Probability': []
               }

# First iteration, where the first addition is always Greedy.

descriptors_in_model, descriptors_bank, result_dict = greedy_addition(df, descriptors_in_model, descriptors_bank, model_tested, counter, result_dict)

round_results = result_compilation(result_dict)

# First ever output generation (with column headers); all further are updates.

round_results.to_csv(f'{output_filename}.csv', index = False, mode = 'a', header = True)

counter += 1

# Continuous loop as long as the number of iterations is not met

for i in range(iterations):

    addition_result = random_addition(df, descriptors_in_model, descriptors_bank, model_tested)

    if len(descriptors_in_model) > 1: # Removal proposals do not occur when there are less than two Descriptors.

        removal_result = random_removal(df, descriptors_in_model, descriptors_bank, model_tested)

    else:

        removal_result = None

    if removal_result != None: # If both proposals are made, the one with the best resulting Score is selected.

        if addition_result['Proposal_Score'] > removal_result['Proposal_Score']:

            action_to_perform = 'Addition'

            current_result = addition_result

            round_score = current_result['Proposal_Score']

        else:

            action_to_perform = 'Removal'

            current_result = removal_result

            round_score = current_result['Proposal_Score']

    else: # If there's no Removal proposal, the algorithm proceeds to Addition by default.

        action_to_perform = 'Addition'

        current_result = addition_result

        round_score = current_result['Proposal_Score']

    previous_round_score = result_dict['Current_Score'][-1]

    direct_accept_condition = round_score > previous_round_score

    if direct_accept_condition:   # If the Score increases, the proposal is automatically accepted.

        descriptors_in_model, descriptors_bank, result_dict = direct_accept(action_to_perform, counter, result_dict, current_result, descriptors_in_model, descriptors_bank)

        round_results = result_compilation(result_dict)

        round_results.to_csv(f'{output_filename}.csv', index = False, mode = 'a', header = False)

    else:    # If the proposal's score's lower than last round, the Acceptance Probability is computed and compared to a  random value (0 - 1) to determine if the score's added or 
             # not. Bigger differences are more likely to be rejected.

        acceptance_probability = np.exp((round_score - previous_round_score) / (kB*temperature))

        test_probability = np.random.uniform()

        probability_conditional = acceptance_probability > test_probability

        if probability_conditional:  # If the Acceptance Probability is greater than the random value, the outcome is accepted.

            descriptors_in_model, descriptors_bank, result_dict = conditional_accept(action_to_perform, counter, result_dict, current_result, descriptors_in_model, descriptors_bank, 
                acceptance_probability, test_probability)

            round_results = result_compilation(result_dict)

            round_results.to_csv(f'{output_filename}.csv', index = False, mode = 'a', header = False)

        else:

            descriptors_in_model, descriptors_bank, result_dict = conditional_reject(action_to_perform, counter, result_dict, current_result, descriptors_in_model, 
                descriptors_bank, acceptance_probability, test_probability)

            round_results = result_compilation(result_dict)

            #round_results.to_csv(f'{output_filename}.csv', index = False, mode = 'a', header = False)

    counter += 1

descriptors_to_extract = sorted(descriptors_in_model, key = lambda x: df.iloc[:,:-1].columns.tolist().index(x))

descriptors_to_extract.append(df.iloc[:,-1].name)

df[descriptors_to_extract].to_csv('ExtractedData.csv', index = False)

## ====================================================================================================================================================================================
## ====================================================================================================================================================================================