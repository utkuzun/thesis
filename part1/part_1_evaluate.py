#################################### Wave Overtopping Neural Network Utility functions ################################# 
# #########################################################################################
# imports
import os
import sys
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

from colorama import Fore, Back, Style

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# directory reach
directory = os.path.join(os.path.dirname(__file__), os.path.abspath(".."))
# setting path
sys.path.append(directory)


start = time.time()
# ############################################
# starter parameters

q_EU_NN_filename = os.path.join(os.path.dirname(__file__),"data-1", "utkuzun50files", "utkuzun50_268_web.txt")
data_input_filename = os.path.abspath("../database/data.xlsx")
domain_input_filename = os.path.abspath("../database/domain.csv")

# ###########################################
# Read files

data_EU = pd.read_csv(q_EU_NN_filename, delimiter="|")
data_actual = pd.read_excel(data_input_filename, sheet_name="vertical", header = 0)
data_actual_input = pd.read_excel(data_input_filename, sheet_name="vertical input part1", header = 0)

# ##########################################
# import domain validty tables
domain_data = pd.read_csv(domain_input_filename, delimiter=";", index_col=0)

# create parameters table with data_actual
data_actual_input["Lm"] = data_actual_input["Tm-1,0,t"] ** 2 * 9.81 / 2 /np.pi

data_non_dimension = pd.DataFrame(columns=domain_data.index)
data_non_dimension["Wave steepness"] = data_actual_input.apply(lambda x : x["Hm0,t"]/x["Lm"], axis=1)
data_non_dimension["Wave obliquity"] = data_actual_input.apply(lambda x : x["beta"], axis=1)
data_non_dimension["Shoaling"] = data_actual_input.apply(lambda x : x["h"]/x["Lm"], axis=1)
data_non_dimension["Toe submergence"] = data_actual_input.apply(lambda x : x["ht"]/x["Hm0,t"], axis=1)
data_non_dimension["Toe width"] = data_actual_input.apply(lambda x : x["Bt"]/x["Lm"], axis=1)
data_non_dimension["Berm submergence"] = data_actual_input.apply(lambda x : x["hb"]/x["Hm0,t"], axis=1)
data_non_dimension["Berm width"] = data_actual_input.apply(lambda x : x["B"]/x["Lm"], axis=1)
data_non_dimension["Crest submergence"] = data_actual_input.apply(lambda x : x["Ac"]/x["Hm0,t"], axis=1)
data_non_dimension["Crest submergence in presence of a crown wall"] = data_actual_input.apply(lambda x : x["Rc"]/x["Hm0,t"], axis=1)
data_non_dimension["Crest width"] = data_actual_input.apply(lambda x : x["Gc"]/x["Lm"], axis=1)
data_non_dimension["Foreshore slope"] = data_actual_input.apply(lambda x : x["mmm"], axis=1)
data_non_dimension["Roughness factor"] = data_actual_input.apply(lambda x : x["gammaf_d"], axis=1)
data_non_dimension["Down slope"] = data_actual_input.apply(lambda x : x["cot(a_d)"], axis=1)
data_non_dimension["Average slope in the run-up/down area"] = data_actual_input.apply(lambda x : x["cot(a_u)"], axis=1)
data_non_dimension["Indication of structure stability"] = data_actual_input.apply(lambda x : x["Dd"]/x["Hm0,t"], axis=1)
data_non_dimension = data_non_dimension.dropna()


# ###########################################
# take q values
q_s = data_actual["q"].values
q_ANN = data_EU["Prototype"].values

# ##########################################
# print some stats

q_data = pd.DataFrame(data = {"q_s" : q_s, "q_ANN": q_ANN})
q_data = q_data.dropna()

print(f"R^2 score calculated as : {r2_score(q_data['q_s'].values, q_data['q_ANN'].values):5.3f}")

# ##################################################
# Plot some graphs 

fig, axes = plt.subplots(ncols= 1, nrows= 3, figsize=(7, 10))
ax1, ax2, ax3 = axes.ravel()

ax1.plot(q_s.ravel(), q_ANN, "ob", label= "predictions", markersize=2)
ax1.plot(q_s.ravel(), q_s.ravel(),"-r" ,label= "True",linewidth=2)
ax1.set_ylabel("$q_{Eurotop ANN}$")
ax1.set_xlabel("$q_s$")
ax1.set_xscale("log")
ax1.set_yscale("log")

# Plot q error vs qs
ax2.plot(q_s.ravel(), (q_s.ravel() - q_ANN.ravel()) / q_s.ravel(), "ob", markersize=2)
ax2.axhline(y= 0, color="r", linestyle= "-",linewidth=2)
ax2.set_ylabel("$(q_s - q_{Eurotop ANN}) / q_s $")
ax2.set_xlabel("$q_s$")
ax2.set_xscale("log")


# Plot q error vs Rc

ax3.plot(data_actual["Rc"].values/data_actual["Hm0 toe"].values, (q_s.ravel() - q_ANN.ravel()) / q_s.ravel(), "ob", markersize=2)
ax3.axhline(y= 0, color="r", linestyle= "-", linewidth=2)
ax3.set_ylabel("$(q_s - q_{Eurotop ANN}) / q_s $")
ax3.set_xlabel("$R_c / H_{m,0,t} $")


fig.tight_layout()

# plt.show()
# ########################################
# add q_Error to domain validity and save

# create table with which exceeds domain limits with data and domain table
dom_validity = pd.DataFrame(columns=["Test ID","exc params", "E", "q_err"])
dom_validity["Test ID"] = data_actual_input["Test ID"]

dom_validity["exc params"] = data_non_dimension.apply(lambda x: [col for col in data_non_dimension.columns if ((x[col] < domain_data.loc[col, "min"]) or (x[col] > domain_data.loc[col, "max"] ))],axis=1)

dom_validity["q_s"] = q_s.ravel()
dom_validity["q_ANN"] = q_ANN.ravel()
dom_validity["q_err"] = (q_s.ravel() - q_ANN.ravel()) / q_s.ravel()
dom_validity.to_excel(f"tables-1/dom_validty_part-1.xlsx", index= False, header= True, sheet_name="part1 domain exceedence", float_format="%.6f")

plt.savefig(os.path.join(os.path.dirname(__file__), "graphs-1", "part_1_eval.png"))

# add E error domain value
# ??????????????????????????????????


print(f"Part-1 Eurotop evaluation done in {(time.time()-start)/60:5.3f} mins!!")


