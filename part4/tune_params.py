#################################### Wave Overtopping Neural Network ################################# 
# #########################################################################################
# imports


from sklearn.model_selection import train_test_split, ShuffleSplit

import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Back, Style

import os
import sys
# directory reach
directory = os.path.join(os.path.dirname(__file__), os.path.abspath(".."))
# setting path
sys.path.append(directory)

from helper.utils import load_data, non_dimensionalize_data, create_paramGrid, create_model, GridSearchCV_asses

# ###################################################################################
# Put static inputs

db_selection = "EU_NN"  #can be SWB or EU_NN
integrated = True
doGridCV = True
CV_splitNumber = 20
scoring = ["r2", "neg_root_mean_squared_error"]
refit = "r2"
return_train_score = True
scale_samples = "scale"
part = "part-4"
deneme = False
max_iter = 10000
# Parameters input
random_state = np.random.RandomState(41)

""" 
For the cv processed GridSearch takes an cv splitter. 
Put an integer in order to execute Kfold with some folds
Put ShuffleSplit or some other splitter class in order to split any other way.
Consider to design param_grid according to splitter
"""
cv = ShuffleSplit(n_splits=CV_splitNumber, test_size=0.2, random_state=random_state)

##########################################################################
# Create param grids

alphas = np.around(np.append(np.logspace(-2, 3, 9), np.array([0])),2)
activations = ["relu", "tanh"]
hidden_layers =  [(25,), (40,), (50,), (75,),(100,1), (30, 20), (50, 40), (85, 50)]


param_grid = create_paramGrid(deneme=deneme, alphas=alphas, hidden_layers=hidden_layers, activations=activations)

########################################################################################
# Load data and non-dimensionalized data

try:
    data = load_data(db_selection=db_selection, integrated=integrated)
    data = non_dimensionalize_data(data, integrated=integrated)
except Exception as error:
    print(Fore.RED)
    print(f"Error in loading data : {error}")
    print(Fore.RESET)

try:
    data_SWB = load_data(db_selection="SWB", integrated=integrated)
    data_SWB = non_dimensionalize_data(data_SWB, integrated=integrated)
except Exception as error:
    print(Fore.RED)
    print(f"Error in loading data : {error}")
    print(Fore.RESET)


#take a portion of swb data
data_SWB, data_SWB_to_add = train_test_split(data_SWB,
    train_size = 1, 
    random_state=random_state)


data = data.concat([data, data_SWB], join="inner", ignore_index=True)
data=data.dropna()
print(f"Concatanated data has this shape {data.shape}...")

feauture_names = data.drop(columns=["q non dim param","db", "q"]).columns

# split into samples and targets
print(data.info())
samples = data.drop(columns=["q non dim param","db", "q"])
targets = data["q"]

# split for groups array of databases (SWB or Eurotop)
groups = data["db"]

# Split data to train, test
X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(samples, targets,groups,
    test_size = 0.2, 
    random_state=random_state)


# ########################################################################################
# CV with some paramaters (GridSearchCV)

model = create_model(random_state=random_state,scale_samples=scale_samples, max_iter=max_iter)

# ###################################################################################################
# Create cross validator

# Do GridSearchCV

results_GridSearch = GridSearchCV_asses(model=model,X_train=X_train,param_grid=param_grid, y_train= y_train,cv=cv,refit=refit, scoring=scoring, return_train_score=return_train_score)

try:
    if deneme:
        results_GridSearch.to_excel(f"tables-4/GridCV_{part}_{scale_samples}_{deneme}.xlsx", index= False, header= True, sheet_name="Search Results", float_format="%.6f")
    else:
        results_GridSearch.to_excel(f"tables-4/GridCV_{part}_{scale_samples}.xlsx", index= False, header= True, sheet_name="Search Results", float_format="%.6f")

    print("Results GridCV_table created!!")
except Exception as err:
    print(err)
        





