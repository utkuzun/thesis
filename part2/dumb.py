# means = results_GridSearch["mean_test_score"]
# stds = results_GridSearch["std_test_score"]
# params = results_GridSearch["params"]

# for mean, std, param in zip(means, stds, params):

#     print(f"{param}: {mean:5.3f} +/- {std:0.03f}")
#     print(f"{param}")



# #####################################################################
# print the final models features






# #############################################################
# some test par

# MinMaxScaler test
""" 
y = MinMaxScaler((0,1)).fit_transform(y_test)
print(y)
 """

# function tranformer test
""" 
logTransformer.fit(y_train)
logy = logTransformer.transform(y_test)
back_y = logTransformer.inverse_transform(logy)

print(f"{np.sum([np.abs(back_y - y_test)])}")
 """

# test the model fit train
""" 
targetPipe = TransformedTargetRegressor(regressor=MLPRegressor(activation="tanh", hidden_layer_sizes=(40,), alpha=0.1, solver="lbfgs"), transformer=MinMaxScaler())  # do target pipeline

model = Pipeline([("preprocess",preprocessor), ("targetTransform", targetPipe)])
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(r2_score(y_test,preds))
 """

