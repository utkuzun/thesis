#################################### Wave Overtopping Neural Network Utility functions ################################# 
# #########################################################################################
# imports

from pandas.core.base import DataError
from pandas.core.frame import DataFrame
from scipy.sparse.construct import random
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import  mean_squared_error, r2_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import  learning_curve


from sklearn.exceptions import ConvergenceWarning

from colorama import Fore, Back, Style

import time
import scipy
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# integrated = False
# db_selection = "SWB_VER"
threshold = -6
def load_data(integrated=False, db_selection="SWB"):

    # database file names and directions
    SWB_file_name=os.path.abspath("../database/data.xlsx")
    db_ANNtool_filename = os.path.abspath("../database/ANN work oın.xlsx")

    # used param for vertical wall data
    ver_wall_params = ["mmm", "h", "Hm0 toe", "Rc", "q", "gf_d", "Ac", "Tm1,0t", "ht"]

    # import SWB data
    if db_selection == "SWB":
        try:
            # import data, assign database name and drop q=0 values. (q = zero is not computable due to logaritmic operations)
            data = pd.DataFrame(pd.read_excel(SWB_file_name, sheet_name="swb", header = 0))
            data["db"] = "SWB"
            data[data["q"]==0] = None
            data[data["q"] < 10**threshold] = None
            data = data.dropna()

        # except errors and print them
        except Exception as error:
            print(Fore.RED)
            print(f"{error} in integrated : {integrated} , db selection : {db_selection}")
            print(Fore.RESET)

    # import EU_NN data
    elif db_selection == "EU_NN":
        try:
            # import data
            data = pd.read_excel(db_ANNtool_filename, skiprows= 0, sheet_name="importdata")

            # check if data is core data and vertical wall data
            data = data.loc[(data["Label"].str.contains("F")) & (data["Core data"].str.contains("Z"))]

            # take only vertical wall params
            data = data[ver_wall_params]

        # except errors and print them
        except Exception as error:
            print(Fore.RED)
            print(f"{error} in integrated : {integrated} , db selection : {db_selection}")
            print(Fore.RESET)

        # if integrated data is asked add SWB params
        if integrated:
            try:
                data["Xr"] = np.zeros((data.shape[0], 1))
                data["Sr"] = np.zeros((data.shape[0], 1))
                data["wb"] = np.zeros((data.shape[0], 1))
                data["Cb"] = np.ones((data.shape[0], 1))
                data["delta_x"] = np.zeros((data.shape[0], 1))
                data["parapet"] = np.zeros((data.shape[0], 1))
                data["hr"] = np.zeros((data.shape[0], 1))
                data["hb"] = np.zeros((data.shape[0], 1))
                data["Wrb"] = np.zeros((data.shape[0], 1))
                data["Parapet"] = np.zeros((data.shape[0], 1))

            # except errors and print them  
            except Exception as error:
                print(Fore.RED)
                print(f"{error} in integrated : {integrated} , db selection : {db_selection}")
                print(Fore.RESET)

        # drop q=0 values, and data < 0 values, assign a database name
        data["q"] = pd.to_numeric(data["q"] ,errors='coerce')
        data[data<0] = None
        data[data["q"] == 0] = None
        data[data["q"] < 10**threshold] = None

        data = data.dropna()
        data["db"] = "EU_NN"

    # import for SWB vertical wall database
    elif db_selection == "SWB_VER":

        # import SWB ver data
        try:
            data = pd.read_excel(SWB_file_name, header= 0, sheet_name="vertical")

        except Exception as error:
            print(Fore.RED)
            print(f"{error} in integrated : {integrated} , db selection : {db_selection}")
            print(Fore.RESET)

        # options for only importing verticaş waşş data
        if not integrated:
            data = data[ver_wall_params]

        # drop q=0 values, and data < 0 values, assign a database name
        data[data["q"] == 0] = None
        data[data["q"] < 10**threshold] = None
        data = data.dropna()
        data["db"] = "SWB_VER"
    
    # print basic summary for imported data

    try:
    
        data_summary = pd.DataFrame(data= None, columns=[
            "count",
            "min-max_H",
            "min-max_Rc",
            "min-max_q",
            "min-max_h",
            "database"
            ]
        )

        summary= {
            "count" : f"{data.shape[0]}",
            "min-max_H" : f"{data['Hm0 toe'].min():5.3f} - {data['Hm0 toe'].max():5.3f}",
            "min-max_Rc": f"{data['Rc'].min():5.3f} - {data['Rc'].max():5.3f}",
            "min-max_q": f"{data['q'].min():7.5f} - {data['q'].max():7.5f}",
            "min-max_h": f"{data['h'].min():5.3f} - {data['h'].max():5.3f}",
            "database": db_selection
        }

        # add summary rows to dataframe and print some results

        data_summary= data_summary.append(summary, ignore_index=True)
        print(f"database: {db_selection} is loaded with following summary reports !!")
        print(data_summary)
        return data
    except Exception as error:
        print(Fore.RED)
        print(error)
        print(Fore.RESET)


    return data
# data = load_data(db_selection=db_selection, integrated=integrated)


def non_dimensionalize_data(data, integrated=False):

    # non dimensionalized according to ANN2 process

    try:
        data["Lm"] = data["Tm1,0t"] ** 2 * 9.81 / 2 /np.pi
        data["q non dim param"] = np.sqrt(9.81 * (data["Hm0 toe"].values) ** 3)

        # shoaling
        data["h"] /= data["Lm"]
        # crest submerge
        data["Ac"] /= data["Hm0 toe"]
        # crest submerge in presence of a crown wall
        data["Rc"] /= data["Hm0 toe"]
        # wave overtopping
        data["q"] = (data["q"].values / np.sqrt(9.81 * (data["Hm0 toe"].values) ** 3)).T
        # wave height at structure
        data["ht"] /= data["Hm0 toe"]

        # non dimensional SWB params
        if integrated:
            data["Xr"] /= data["Lm"]
            data["hb"] /= data["Hm0 toe"]
            data["wb"] /= data["Lm"]
            data["Parapet"] /= data["Lm"]
            data["parapet"] /= data["Lm"]
            data["delta_x"] /= data["Lm"]
            data["hr"] /= data["Hm0 toe"]
            data["Wrb"] /= data["Lm"]

        # wave steepness
        data["Hm0 toe"] /= data["Lm"]  
        
        # drop unneccesary params
        data = data.drop(columns= ["Lm","Tm1,0t"])

        # rename for actual representations of datas
        data = data.rename(columns= {
            "Hm0 toe" : "wave steepnes",
            "h": "wave shoaling",
            "Ac" : "crest submerge",
            "Rc" : "crest submerge with crown wall",
            "m" : "foreshore slope",
            "gf_d" : "roughness factor",
            "ht" : "depth at structure toe"})

        
    # print for errors
    except Exception as error:
        print(Fore.RED)
        print(error)
        print(Fore.RESET)

    if not integrated:
        data = data.loc[:,  ["mmm", "wave shoaling", "wave steepnes", "crest submerge with crown wall", "roughness factor", "crest submerge","depth at structure toe","q non dim param","db","q"]]
    else:
        data = data.loc[:,  ["mmm", "wave shoaling", "wave steepnes", "crest submerge with crown wall", "roughness factor", "crest submerge","depth at structure toe",
        "hb", "Xr", "wb","Parapet", "parapet", "delta_x","hr", "Wrb","q non dim param","db","q"]]

    return data


def domain_validity_table(data_train,data_test, q_s,q_ANN ):
    # create table with which exceeds domain limits with data and domain table
    dom_validity = pd.DataFrame(columns=["Test ID","exc params","q_s","q_ANN", "E", "q_err"])
    dom_validity["Test ID"] = data_test.index +2
    dom_validity.index = data_test.index
    dom_validity["q_ANN"] = q_ANN
    dom_validity["q_s"] = q_s

    dom_validity["exc params"] = data_test.apply(lambda x: [col for col in data_test.columns if ((x[col] < data_train.loc[:, col].min()) or (x[col] > data_train.loc[:, col].max()) )],axis=1)
    dom_validity["q_err"] = (q_s.ravel() - q_ANN.ravel()) / q_s.ravel()

    return dom_validity


def create_paramGrid(deneme, alphas, activations, hidden_layers):

 
    if deneme:
        param_grid = [{
            "targetTransform__regressor__alpha" : [0.001, 0.1 , 0, 10],
            "targetTransform__regressor__hidden_layer_sizes" : [(50,)],
            "targetTransform__regressor__activation" : ["tanh"],
            "targetTransform__regressor__solver" : ["lbfgs"],
            }]
    else:
        param_grid=[
            {
            "targetTransform__regressor__alpha" : alphas,
            "targetTransform__regressor__hidden_layer_sizes" :hidden_layers,
            "targetTransform__regressor__activation" : activations,
            "targetTransform__regressor__solver" : ["lbfgs"],
            },
            {
            "targetTransform__regressor__alpha" : alphas,
            "targetTransform__regressor__hidden_layer_sizes" : hidden_layers,
            "targetTransform__regressor__activation" :activations,
            "targetTransform__regressor__solver" : ["adam"],
            },
            {
            "targetTransform__regressor__alpha" :alphas,
            "targetTransform__regressor__hidden_layer_sizes" : hidden_layers,
            "targetTransform__regressor__activation" : activations,
            "targetTransform__regressor__solver" : ["sgd"],
            }
        ]   

    return param_grid 


def create_model(random_state,scale_samples=True, max_iter=1000):
    # Create the pipeline 

    # pipes for X values
    preprocessor = Pipeline([("preprocess", StandardScaler())])  # preprocessor for feauteres
    
    # pipes for y values
    logTransformer = FunctionTransformer(func=np.log10, inverse_func=scipy.special.exp10)  # log transform the targets
    targetTransform = make_pipeline(logTransformer, MinMaxScaler((0,1)))  # assign tranformer functions for the targets
    targetPipe = TransformedTargetRegressor(regressor=MLPRegressor(tol=1e-4,random_state=random_state, max_iter=max_iter), transformer=targetTransform)  # do target pipeline

    if scale_samples == "scale":
        model = Pipeline([("preprocess",preprocessor), ("targetTransform", targetPipe)])
    else:
        model = Pipeline([("targetTransform", targetPipe)])

    print(model)
    return model
# data = non_dimensionalize_data(data, integrated=integrated)

def create_model_static(random_state,params, scale_samples=True, max_iter=1000):
    # Create the pipeline 

    # pipes for X values
    preprocessor = Pipeline([("preprocess", StandardScaler())])  # preprocessor for feauteres
    
    # pipes for y values
    logTransformer = FunctionTransformer(func=np.log10, inverse_func=scipy.special.exp10)  # log transform the targets
    targetTransform = make_pipeline(logTransformer, MinMaxScaler((0,1)))  # assign tranformer functions for the targets
    targetPipe = TransformedTargetRegressor(regressor=MLPRegressor(random_state=random_state, max_iter=max_iter, **params), transformer=targetTransform)  # do target pipeline

    if scale_samples == "scale":
        model = Pipeline([("preprocess",preprocessor), ("targetTransform", targetPipe)])
    else:
        model = Pipeline([("targetTransform", targetPipe)])

    print(model)
    return model


def GridSearchCV_asses(model,X_train, y_train,param_grid,cv,refit, scoring="r2", return_train_score=True):


    start_time = time.time()
    print(f"GridSearhCV begin with {X_train.values.shape[0]} samples")
    search = GridSearchCV(model, param_grid=param_grid, cv=cv, scoring=scoring, return_train_score=return_train_score, refit=refit).fit(X_train.values, y_train.values.reshape(-1, 1))

    print(f"GridSearch done in {(time.time()-start_time)/60:5.3f} mins!!")

    # print GridSearchCV results

    results_GridSearch = pd.DataFrame(data=search.cv_results_, index=search.cv_results_["params"])
    results_GridSearch = results_GridSearch.sort_values(by=[f'rank_test_{refit}'])

    results_GridSearch["solver"] = [params['targetTransform__regressor__solver'] for params in results_GridSearch["params"]]
    results_GridSearch["activation"] = [params['targetTransform__regressor__activation'] for params in results_GridSearch["params"]]
    results_GridSearch["hidden_layer_sizes"] = [params['targetTransform__regressor__hidden_layer_sizes'] for params in results_GridSearch["params"]]
    results_GridSearch["alpha"] = [params['targetTransform__regressor__alpha'] for params in results_GridSearch["params"]]

    
    print("Best parameters found on:")
    print(f"{search.best_params_} with score {search.best_score_:5.3f}")

    return results_GridSearch

def drawLearningCurve(fig, ax1, model, X_train, y_train, cv, refit, random_state, params):


    steps, train_scores, test_scores = learning_curve(model, X_train.values, y_train.values.reshape(-1, 1),cv=cv,train_sizes=np.linspace(0.1, 1.0, 15), scoring=refit, random_state=random_state, n_jobs=-1)

    train_scores[train_scores < 0 ] = 0
    test_scores[test_scores < 0 ] = 0

    train_scores = np.mean(train_scores, axis=1)
    test_scores = np.mean(test_scores, axis=1)
    test_scores[test_scores < 0] = 0 


    ax1.plot(steps, train_scores, "b-", label="Training")
    ax1.plot(steps, test_scores, "r*", label="CV")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel(f"$r^2$ Score")
    ax1.set_title(f"Learning Curve with params : hidden layer {params['hidden_layer_sizes']} - alpha : {params['alpha']} ")

    ax1.legend(loc= "best", prop={'size': 7})

    return fig

def get_q_ANN_with_resamples(model, X_train, y_train, estimation_params,q_s, random_state, resample_num=50):


    print(f"Estimation for q in progress")

    columns = ["q_s", "q_ANN" ,"q_std"]

    for i in range(resample_num):
        columns.append(f"q_ANN_{i}")

    estimation = pd.DataFrame(data=None, columns=columns)

    for i in range(resample_num):

        print(f"Steps : {i + 1}/{resample_num}...")

        X_subset, _, y_subset, _ = train_test_split(X_train, y_train, train_size=0.8,random_state=random_state)
        model.fit(X_subset.values, y_subset.values.reshape(-1, 1))

        estimation[f"q_ANN_{i}"] = model.predict(estimation_params.values).ravel()

    estimation["q_ANN"] = estimation[[f"q_ANN_{i}" for i in range(resample_num)]].apply(np.mean, axis=1)
    estimation["q_std"] = estimation[[f"q_ANN_{i}" for i in range(resample_num)]].apply(np.std, axis=1)
    estimation["q_s"] = q_s.ravel()

    return estimation


def draw_metrics(X_test,q_s, q_ANN, fig,axes, params):

    ax1, ax2, ax3 = axes.ravel()

    # Plot qs vs qANN 

    ax1.plot(q_s.ravel(), q_ANN, "ob", label= "predictions", markersize=2)
    ax1.plot(q_s.ravel(), q_s.ravel(),"-r" ,label= "True preddictions",linewidth=2)
    ax1.set_ylabel("$q_{ANN}$")
    ax1.set_xlabel("$q_s$")

    ax1.set_xscale("log")
    ax1.set_yscale("log")


    # Plot q error vs qs
    ax2.plot(q_s.ravel(), (q_s.ravel() - q_ANN.ravel()) / q_s.ravel(), "ob", markersize=2)
    ax2.axhline(y= 0, color="r", linestyle= "-",linewidth=2)
    ax2.set_ylabel("$(q_s - q_{ANN}) / q_s $")
    ax2.set_xlabel("$q_s$")
    ax2.set_xscale("log")


    ax1.set_title(f"metrics with params : hidden layer {params['hidden_layer_sizes']} - alpha : {params['alpha']} ")

    # Plot q error vs Rc

    ax3.plot(X_test.values[:, 3], (q_s.ravel() - q_ANN.ravel()) / q_s.ravel(), "ob", markersize=2)
    ax3.axhline(y= 0, color="r", linestyle= "-", linewidth=2)
    ax3.set_ylabel("$(q_s - q_{ANN}) / q_s $")
    ax3.set_xlabel("$R_c / H_{m,0,t} $")

    return fig

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    cbar.ax.set_ylim(-0.5, np.nanmax(data))

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:3.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def create_heatmap_data(results_GridSearch, activation, refit):

    
    # data = results_GridSearch.loc[((results_GridSearch["activation"] ==activation)), [f"mean_test_{refit}", f"std_test_{refit}", "hidden_layer_sizes","param_targetTransform__regressor__alpha",  f"mean_train_{refit}", f"std_train_{refit}"]]
    data_pd = results_GridSearch.loc[((results_GridSearch["activation"] ==activation) & (results_GridSearch["solver"] =="lbfgs")), [f"mean_test_{refit}", "hidden_layer_sizes","alpha"]]
    alphas = data_pd["alpha"].unique()
    hidden_layers = data_pd["hidden_layer_sizes"].unique()

    data_np = [[data_pd.loc[((data_pd["alpha"] == alpha) & (data_pd["hidden_layer_sizes"] == hidden_layer)),f"mean_test_{refit}" ] for alpha in alphas] for hidden_layer in hidden_layers]

    return np.array(data_np), alphas, hidden_layers


def results_table(estimation, estimation_VER, resample_num, estimation_SWB=None):

    estimation_results = pd.DataFrame(data=None)
    estimation_results_summary = pd.DataFrame(data=None)

    for i in range(resample_num):
        estimation_results = pd.concat([pd.DataFrame([r2_score(estimation["q_s"].values.ravel(), estimation[f"q_ANN_{i}"].values.ravel())], columns=['r2_EU']) for i in range(resample_num)], ignore_index=True)
        estimation_results["rmse_EU"] = pd.concat([pd.DataFrame([mean_squared_error(estimation["q_s"].values.ravel(), estimation[f"q_ANN_{i}"].values.ravel(), squared=False)], columns=['rmse_EU']) for i in range(resample_num)], ignore_index=True).values
        
        estimation_results["r2_VER"] = pd.concat([pd.DataFrame([r2_score(estimation_VER["q_s"].values.ravel(), estimation_VER[f"q_ANN_{i}"].values.ravel())], columns=['r2_VER']) for i in range(resample_num)], ignore_index=True).values
        estimation_results["rmse_VER"] = pd.concat([pd.DataFrame([mean_squared_error(estimation_VER["q_s"].values.ravel(), estimation_VER[f"q_ANN_{i}"].values.ravel(), squared=False)], columns=['rmse_VER']) for i in range(resample_num)], ignore_index=True).values


        if estimation_SWB is not None:
            estimation_results["r2_SWB"] = pd.concat([pd.DataFrame([r2_score(estimation_SWB["q_s"].values.ravel(), estimation_SWB[f"q_ANN_{i}"].values.ravel())], columns=['r2_SWB']) for i in range(resample_num)], ignore_index=True).values
            estimation_results["rmse_SWB"] = pd.concat([pd.DataFrame([mean_squared_error(estimation_SWB["q_s"].values.ravel(), estimation_SWB[f"q_ANN_{i}"].values.ravel(), squared=False)], columns=['rmse_SWB']) for i in range(resample_num)], ignore_index=True).values

    summary_data = {
    "r2_EU" : f"{estimation_results['r2_EU'].mean():0.3f} +/- {estimation_results['r2_EU'].std():4.3f}",
    "rmse_EU" : f"{estimation_results['rmse_EU'].mean():0.6f} +/- {estimation_results['rmse_EU'].std():0.6f}",

    "r2_VER" : f"{estimation_results['r2_VER'].mean():0.3f} +/- {estimation_results['r2_VER'].std():4.3f}",
    "rmse_VER" : f"{estimation_results['rmse_VER'].mean():0.6f} +/- {estimation_results['rmse_VER'].std():0.6f}",

    "r2_SWB" : f"{estimation_results['r2_SWB'].mean():0.3f} +/- {estimation_results['r2_SWB'].std():4.3f}" if estimation_SWB is not None else None,
    "rmse_SWB" : f"{estimation_results['rmse_SWB'].mean():0.6f} +/- {estimation_results['rmse_SWB'].std():0.6f}" if estimation_SWB is not None else None,

    "r2_EU_act" : f'{r2_score(estimation["q_s"].values.ravel(),estimation["q_ANN"].values.ravel()):4.3f}',
    "rmse_EU_act" : f'{mean_squared_error(estimation["q_s"].values.ravel(),estimation["q_ANN"].values.ravel(), squared=False):4.3f}',

    "r2_VER_act" : f'{r2_score(estimation_VER["q_s"].values.ravel(),estimation_VER["q_ANN"].values.ravel()):4.3f}',
    "rmse_VER_act" : f'{mean_squared_error(estimation_VER["q_s"].values.ravel(),estimation_VER["q_ANN"].values.ravel(), squared=False)}:4.3f',

    "r2_SWB_act" : f'{r2_score(estimation_SWB["q_s"].values.ravel(),estimation_SWB["q_ANN"].values.ravel()):4.3f}' if estimation_SWB is not None else None,
    "rmse_SWB_act" : f'{mean_squared_error(estimation_SWB["q_s"].values.ravel(),estimation_SWB["q_ANN"].values.ravel(), squared=False):4.3f}' if estimation_SWB is not None else None,

    }

    estimation_results_summary = estimation_results_summary.append(summary_data, ignore_index=True)

    return estimation_results_summary, estimation_results



