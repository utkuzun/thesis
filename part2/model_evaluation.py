#################################### Wave Overtopping Neural Network ################################# 
# #########################################################################################
# imports

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score
from colorama import Fore, Back, Style

import time
from sklearn.exceptions import ConvergenceWarning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

import os
import sys
  
# directory reach
directory = os.path.join(os.path.dirname(__file__), os.path.abspath(".."))
# setting path
sys.path.append(directory)

from helper.utils import load_data, non_dimensionalize_data, domain_validity_table, create_model_static, drawLearningCurve, draw_metrics, get_q_ANN_with_resamples


# ###################################################################################
# Put selected MLP parameters

params = {
    "activation" : "tanh",
    "solver" : "lbfgs",
    "hidden_layer_sizes" : (40,),
    "alpha" : 0.04
}

# classification of the database/model

scale_samples = "scale"
part = "part-2"
refit="r2"
CV_splitNumber = 3
integrated = False
db_selection = "EU_NN"
max_iter = 100


# Parameters input
random_state = np.random.RandomState(41)

cv = ShuffleSplit(n_splits=CV_splitNumber,test_size=0.2,random_state=random_state)
# cv = CV_splitNumber


########################################################################################
# Load data and non-dimensionalized data


data = load_data(db_selection=db_selection, integrated=integrated)
data = non_dimensionalize_data(data, integrated=integrated)
# print(Fore.RED)
# print(f"Error in loading data : {error}")
# print(Fore.RESET)

feauture_names = data.drop(columns=["db", "q"]).columns

# split into samples and targets
print(data.info())
print(data.shape)
samples = data.drop(columns=["db", "q"])
targets = data["q"]
# targets = targets.reshape(-1, 1)

# load data for SWB vertical database

try:
    data_test_VER = load_data(db_selection="SWB_VER", integrated=integrated)
    data_test_VER = non_dimensionalize_data(data_test_VER, integrated=integrated)
except Exception as error:
    print(Fore.RED)
    print(f"Error in loading data : {error}")
    print(Fore.RESET)


X_test_VER = data_test_VER.drop(columns=["db", "q"])
y_test_VER = data_test_VER["q"]


# split for groups array of databases (SWB or Eurotop)

groups = data["db"].values
groups = groups.ravel()

# Split data to train, test

X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(samples, targets,groups,
    test_size = 0.2, 
    random_state=random_state)


# ########################################################################################
# Create the model
start_time = time.time()

model = create_model_static(random_state=random_state,params=params,scale_samples=scale_samples, max_iter=max_iter)

# ###################################################################################
# Draw learning curve

if y_train.shape[0] <= 500:
    steps = np.arange(5, 60, 5)
else:
    steps = np.arange(25, 800, 25)


fig, ax1 = plt.subplots(figsize=(7, 7))
fig = drawLearningCurve(fig, ax1, model, X_train, y_train, cv, refit, random_state, params, steps)

fig.tight_layout()
plt.savefig(f"graphs-2/learning_curve_{part}_{scale_samples}.png")


# ###################################################################################
# Draw metrics with test data from database

estimation = get_q_ANN_with_resamples(model, X_train, y_train, X_test,y_test, random_state, 5)
estimation.to_excel(f"tables-2/results_part-2.xlsx", index= False, header= True, sheet_name="results", float_format="%.6f")

q_ANN = estimation["q_ANN"].values
q_s = estimation["q_s"].values


fig, axes = plt.subplots(ncols= 1, nrows= 3, figsize=(7, 10))
fig = draw_metrics(X_test,y_test.values, q_ANN, fig,axes, params)

fig.tight_layout()
plt.savefig(f"graphs-2/metrics_for_{part}_{scale_samples}.png")


# ###################################################################################
# Draw metrics with test data from METU vertical wall data


estimation_VER = get_q_ANN_with_resamples(model, X_train, y_train, X_test_VER,y_test_VER, random_state, 5)
estimation_VER.to_excel(f"tables-2/results_VER_part-2.xlsx", index= False, header= True, sheet_name="results", float_format="%.6f")


q_ANN_VER = estimation_VER["q_ANN"].values
q_s_VER = estimation_VER["q_s"].values

fig2, axes2 = plt.subplots(ncols= 1, nrows= 3, figsize=(7, 10))
fig2 = draw_metrics(X_test_VER,y_test_VER.values, q_ANN_VER, fig2,axes2, params)
fig2.tight_layout()
plt.savefig(f"graphs-2/metrics_for_{part}_{scale_samples}_VER.png")

# ###################################################################################
# Draw training samples vs SWB_ver samples

fig3, axfinal = plt.subplots(ncols=1, nrows=1, figsize=(7,7))

axfinal.plot(X_train.values[:, 3], y_train.values.reshape(-1, 1), "or", markersize=2, label="training_data")
axfinal.plot(X_test_VER.values[:, 3], q_ANN_VER, "ob", markersize=2, label="predictions")

axfinal.set_ylabel("$ q_ANN $")
axfinal.set_xlabel("$R_c / H_{m,0,t} $")
axfinal.set_yscale("log")
axfinal.legend(loc= "best", prop={'size': 7})

fig3.tight_layout()
plt.savefig(f"graphs-2/prediction_vs_training_{part}_{scale_samples}_VER.png")

# ####################################################################################
# domain validity for test data

dom_validity = domain_validity_table(X_train, X_test, q_s.ravel(), q_ANN.ravel())
dom_validity.to_excel(f"tables-2/dom_validty_part-2.xlsx", index= False, header= True, sheet_name="domain exceedence", float_format="%.6f")

dom_validity_VER = domain_validity_table(X_train, X_test_VER, q_s_VER, q_ANN_VER)
dom_validity_VER.to_excel(f"tables-2/dom_validty_VER_part-2.xlsx", index= False, header= True, sheet_name="domain exceedence", float_format="%.6f")

print(f"R^2 between METU vertical wall data : {r2_score(q_ANN_VER, q_s_VER)}")
print(f"R^2 between EU_NN data is : {r2_score(q_ANN, q_s)} \nModel Evaluation done in {(time.time()-start_time)/60:5.3f} mins!!")