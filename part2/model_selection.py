#################################### Wave Overtopping Neural Network ################################# 
# #########################################################################################
# imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core import base


# ###################################################################################
# Put static inputs
part = "part-2"
CV_splitNumber = 20
refit="r2"
scale_samples = "scale"
deneme=False
activation = "tanh"
base_alpha = 0.18
base_hidden_layer = "(40,)" 

# Parameters input
prng = np.random.RandomState(41)

if deneme:
    results_GridSearch =  pd.DataFrame(pd.read_excel(f"tables-2/GridCV_{part}_{scale_samples}_{deneme}.xlsx", header=0, sheet_name="Search Results"))
else:
    results_GridSearch =  pd.DataFrame(pd.read_excel(f"tables-2/GridCV_{part}_{scale_samples}.xlsx", header=0, sheet_name="Search Results"))

def take_hiddens_comp(base_alpha=0,activation="tanh"):

    hidden_layer_comp_df = results_GridSearch.loc[((results_GridSearch["alpha"] == base_alpha) & (results_GridSearch["activation"] ==activation)), [f"mean_test_{refit}", f"std_test_{refit}", "hidden_layer_sizes", f"mean_train_{refit}", f"std_train_{refit}"]]
    hidden_layer_comp_df["max_test_score"] = results_GridSearch.loc[((results_GridSearch["alpha"] == base_alpha) & (results_GridSearch["activation"] ==activation))][[f"split{i}_test_{refit}" for i in range(CV_splitNumber)]].values.max(axis=1)
    hidden_layer_comp_df["max_train_score"] = results_GridSearch.loc[((results_GridSearch["alpha"] == base_alpha) & (results_GridSearch["activation"] ==activation))][[f"split{i}_train_{refit}" for i in range(CV_splitNumber)]].values.max(axis=1)
    hidden_layer_comp_df = hidden_layer_comp_df.sort_values(by=f"mean_test_{refit}", ascending=False).drop_duplicates(subset="hidden_layer_sizes", keep="first")
    print(hidden_layer_comp_df)
    return hidden_layer_comp_df


def take_alphas_comp(base_hidden_layer="(40,)",activation="tanh"):


    alpha_comp_df = results_GridSearch.loc[((results_GridSearch["hidden_layer_sizes"] == base_hidden_layer) & (results_GridSearch["activation"] == activation)), [f"mean_test_{refit}", f"std_test_{refit}", "param_targetTransform__regressor__alpha", f"mean_train_{refit}", f"std_train_{refit}"]]
    alpha_comp_df = alpha_comp_df.rename({"param_targetTransform__regressor__alpha":"alpha"}, axis="columns")
    alpha_comp_df["max_test_score"] = results_GridSearch.loc[((results_GridSearch["hidden_layer_sizes"] == base_hidden_layer) & (results_GridSearch["activation"] == activation))][[f"split{i}_test_{refit}" for i in range(CV_splitNumber)]].values.max(axis=1)
    alpha_comp_df["max_train_score"] = results_GridSearch.loc[((results_GridSearch["hidden_layer_sizes"] == base_hidden_layer) & (results_GridSearch["activation"] == activation))][[f"split{i}_train_{refit}" for i in range(CV_splitNumber)]].values.max(axis=1)
    alpha_comp_df = alpha_comp_df.sort_values(by=f"mean_test_{refit}", ascending=False).drop_duplicates(subset="alpha", keep="first")
    print(alpha_comp_df)
    return alpha_comp_df

hidden_layer_comp_df = take_hiddens_comp(activation=activation, base_alpha=base_alpha)
alpha_comp_df = take_alphas_comp(activation=activation, base_hidden_layer=base_hidden_layer)

hidden_layer_comp_df.to_excel(f"tables-2/metrics_eval_{part}_{scale_samples}_hidden.xlsx", index= False, header= True, sheet_name="hidden_layer", float_format="%.6f")
alpha_comp_df.to_excel(f"tables-2/metrics_eval_{part}_{scale_samples}_alphas.xlsx", index= False, header= True, sheet_name="alpha", float_format="%.6f")


# #################################################
# Do plots

fig, (ax1,ax2) = plt.subplots(figsize= (7,7), nrows=2, ncols=1)

###############
# hidden layers
ax1.set_title(f"Hidden layers vs $R^{2}$ metric with {base_alpha}")
ax1.set_ylabel("$r^{2}$")
ax1.set_xlabel("hidden layer sizes")

ax1.plot(hidden_layer_comp_df["hidden_layer_sizes"], hidden_layer_comp_df[f"mean_test_{refit}"], "*r", label="CV" )
ax1.plot(hidden_layer_comp_df["hidden_layer_sizes"], hidden_layer_comp_df[f"mean_train_{refit}"], "ob", label="train" )

ax1.set_ylim(0,1)
ax1.legend(loc= "best", prop={'size': 7})

# ########################
# Alphas
ax2.set_title(f"Alphas vs $R^{2}$ metric with {base_hidden_layer} hidden layers")
ax2.set_ylabel("$r^{2}$")
ax2.set_xlabel("alphas")

ax2.plot(alpha_comp_df["alpha"].values[:7], alpha_comp_df[f"mean_test_{refit}"].values[:7], "*r", label="CV" )
ax2.plot(alpha_comp_df["alpha"].values[:7], alpha_comp_df[f"mean_train_{refit}"].values[:7], "ob", label="train" )

ax2.set_ylim(0,1)
ax2.legend(loc= "best", prop={'size': 7})
fig.tight_layout()

if deneme:  
    plt.savefig(f"graphs-2/GridCV_{part}_{scale_samples}_{deneme}.png")
else:
    plt.savefig(f"graphs-2/GridCV_{part}_{scale_samples}.png")

