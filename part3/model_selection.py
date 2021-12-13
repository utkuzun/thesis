#################################### Wave Overtopping Neural Network ################################# 
# #########################################################################################
# imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core import base
import os
import sys

# directory reach
directory = os.path.join(os.path.dirname(__file__), os.path.abspath(".."))
# setting path
sys.path.append(directory)

from helper.utils import heatmap, annotate_heatmap, create_heatmap_data



# ###################################################################################
# Put static inputs
part = "part-3"
CV_splitNumber = 50
refit="r2"
scale_samples = "scale"
deneme=False
activation = "tanh"
base_alpha = 0.18
base_hidden_layer = "(40,)" 

# Parameters input
prng = np.random.RandomState(41)

if deneme:
    results_GridSearch =  pd.DataFrame(pd.read_excel(f"tables-3/GridCV_{part}_{scale_samples}_{deneme}.xlsx", header=0, sheet_name="Search Results"))
else:
    results_GridSearch =  pd.DataFrame(pd.read_excel(f"tables-3/GridCV_{part}_{scale_samples}.xlsx", header=0, sheet_name="Search Results"))

# def take_hiddens_comp(base_alpha=0,activation="tanh"):

#     hidden_layer_comp_df = results_GridSearch.loc[((results_GridSearch["activation"] ==activation)), [f"mean_test_{refit}", f"std_test_{refit}", "hidden_layer_sizes","param_targetTransform__regressor__alpha",  f"mean_train_{refit}", f"std_train_{refit}"]]
#     hidden_layer_comp_df["max_test_score"] = results_GridSearch.loc[( (results_GridSearch["activation"] ==activation))][[f"split{i}_test_{refit}" for i in range(CV_splitNumber)]].values.max(axis=1)
#     hidden_layer_comp_df["max_train_score"] = results_GridSearch.loc[( (results_GridSearch["activation"] ==activation))][[f"split{i}_train_{refit}" for i in range(CV_splitNumber)]].values.max(axis=1)
#     # hidden_layer_comp_df = hidden_layer_comp_df.sort_values(by=f"mean_test_{refit}", ascending=False).drop_duplicates(subset="hidden_layer_sizes", keep="first")
#     hidden_layer_comp_df = hidden_layer_comp_df.sort_values(by=f"mean_test_{refit}", ascending=False)
#     return hidden_layer_comp_df


# def take_alphas_comp(base_hidden_layer="(40,)",activation="tanh"):


#     alpha_comp_df = results_GridSearch.loc[((results_GridSearch["hidden_layer_sizes"] == base_hidden_layer) & (results_GridSearch["activation"] == activation)), [f"mean_test_{refit}", f"std_test_{refit}", "param_targetTransform__regressor__alpha", f"mean_train_{refit}", f"std_train_{refit}"]]
#     alpha_comp_df = alpha_comp_df.rename({"param_targetTransform__regressor__alpha":"alpha"}, axis="columns")
#     alpha_comp_df["max_test_score"] = results_GridSearch.loc[((results_GridSearch["hidden_layer_sizes"] == base_hidden_layer) & (results_GridSearch["activation"] == activation))][[f"split{i}_test_{refit}" for i in range(CV_splitNumber)]].values.max(axis=1)
#     alpha_comp_df["max_train_score"] = results_GridSearch.loc[((results_GridSearch["hidden_layer_sizes"] == base_hidden_layer) & (results_GridSearch["activation"] == activation))][[f"split{i}_train_{refit}" for i in range(CV_splitNumber)]].values.max(axis=1)
#     alpha_comp_df = alpha_comp_df.sort_values(by=f"mean_test_{refit}", ascending=False)
#     # alpha_comp_df = alpha_comp_df.sort_values(by=f"mean_test_{refit}", ascending=False).drop_duplicates(subset="alpha", keep="first")
#     return alpha_comp_df

# hidden_layer_comp_df = take_hiddens_comp(activation=activation, base_alpha=base_alpha)
# alpha_comp_df = take_alphas_comp(activation=activation, base_hidden_layer=base_hidden_layer)

# hidden_layer_comp_df.to_excel(f"tables-2/metrics_eval_{part}_{scale_samples}_hidden.xlsx", index= False, header= True, sheet_name="hidden_layer", float_format="%.6f")
# alpha_comp_df.to_excel(f"tables-2/metrics_eval_{part}_{scale_samples}_alphas.xlsx", index= False, header= True, sheet_name="alpha", float_format="%.6f")


# #################################################
# Do plots

# ######################
# plot surf map

# plot for activation tanh

figgridCVtanh,axesCVtanh  = plt.subplots(figsize= (7,7), nrows=1, ncols=1)
axgridCvTanh = axesCVtanh

data_tanh, alphas, hidden_layers = create_heatmap_data(results_GridSearch, activation="tanh", refit="r2")
alphas = [f"{alpha:3.3f}" for alpha in alphas]
hidden_layers = [f"{hidden_layer}" for hidden_layer in hidden_layers]

data_tanh[data_tanh<-0.5] = None

im, cbar = heatmap(data_tanh, hidden_layers, alphas, ax=axgridCvTanh,
                   cmap="YlGn", cbarlabel="$r^2$")
texts = annotate_heatmap(im, valfmt="{x:0.2f} ")

axgridCvTanh.set_title(f"GridSearchCV results with activation tanh")
axgridCvTanh.set_ylabel("hidden layer sizes")
axgridCvTanh.set_xlabel("alpha")

figgridCVtanh.tight_layout()
plt.savefig(f"graphs-3/GridCV_matrix_results_tanh_{part}_{scale_samples}_{deneme}.png")


# plot for activation relu

figgridCVrelu,axesCVrelu = plt.subplots(figsize= (7,7), nrows=1, ncols=1)
axgridCvRelu = axesCVrelu

data_relu, alphas, hidden_layers = create_heatmap_data(results_GridSearch, activation="relu", refit="r2")
alphas = [f"{alpha:3.3f}" for alpha in alphas]
hidden_layers = [f"{hidden_layer}" for hidden_layer in hidden_layers]

data_relu[data_relu<-0.5] = None

im, cbar = heatmap(data_relu, hidden_layers, alphas, ax=axgridCvRelu,
                   cmap="YlGn", cbarlabel="$r^2$")
texts = annotate_heatmap(im, valfmt="{x:0.2f} ")

axgridCvRelu.set_title(f"GridSearchCV results with activation relu")
axgridCvRelu.set_ylabel("hidden layer sizes")
axgridCvRelu.set_xlabel("alpha")

figgridCVrelu.tight_layout()
plt.savefig(f"graphs-3/GridCV_matrix_results_relu_{part}_{scale_samples}_{deneme}.png")




# ###############
# # hidden layers

# fig, (ax1,ax2) = plt.subplots(figsize= (7,7), nrows=2, ncols=1)

# ax1.set_title(f"Hidden layers vs $R^{2}$ metric with {base_alpha}")
# ax1.set_ylabel("$r^{2}$")
# ax1.set_xlabel("hidden layer sizes")

# ax1.plot(hidden_layer_comp_df["hidden_layer_sizes"], hidden_layer_comp_df[f"mean_test_{refit}"], "*r", label="CV" )
# ax1.plot(hidden_layer_comp_df["hidden_layer_sizes"], hidden_layer_comp_df[f"mean_train_{refit}"], "ob", label="train" )

# ax1.set_ylim(0,1)
# ax1.legend(loc= "best", prop={'size': 7})

# # ########################
# # Alphas
# ax2.set_title(f"Alphas vs $R^{2}$ metric with {base_hidden_layer} hidden layers")
# ax2.set_ylabel("$r^{2}$")
# ax2.set_xlabel("alphas")

# ax2.plot(alpha_comp_df["alpha"].values[:7], alpha_comp_df[f"mean_test_{refit}"].values[:7], "*r", label="CV" )
# ax2.plot(alpha_comp_df["alpha"].values[:7], alpha_comp_df[f"mean_train_{refit}"].values[:7], "ob", label="train" )

# ax2.set_ylim(0,1)
# ax2.legend(loc= "best", prop={'size': 7})
# fig.tight_layout()

# if deneme:  
#     plt.savefig(f"graphs-2/GridCV_{part}_{scale_samples}_{deneme}.png")
# else:
#     plt.savefig(f"graphs-2/GridCV_{part}_{scale_samples}.png")

