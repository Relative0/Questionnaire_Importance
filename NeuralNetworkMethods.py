#Good to go

# Import external packages.
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import shap

# Import local packages.
from DimensionalityBinning import DimensionalBinChoice

# Create a neural network from a dictionary of inputs.
def KerasModel(Dict):
    # Create a sequential model.
    model = Sequential()
    # Add first hidden layer based on dictionary values.
    model.add(Dense(Dict['L1Neurons'], input_dim=Dict['input_dim'], activation=Dict['activation']))
    # Add second hidden layer based on dictionary values.
    model.add(Dense(Dict['L2Neurons'], input_dim=Dict['L1Neurons'], activation=Dict['activation']))
    # Add third hidden layer based on dictionary values.
    # model.add(Dense(Dict['L3Neurons'], input_dim=Dict['L2Neurons'], activation=Dict['activation']))
    # Add output layer.
    model.add(Dense(Dict['output'], activation=Dict['activation']))
    # Compile the model.
    model.compile(loss=Dict['loss'], optimizer=Dict['optimizer'], metrics=[Dict['metrics']])

    return model

# Create a neural network, train and test it using a subset of questions.
def Subset_Analysis(FullQ_Train_Test_Split, ParameterDictionary_Subset, TheBinsizeList, Questionnaire_Subset):
    # FullQ_Train_Test_Split is the full set of questions which the
    X_Subset_train, X_Subset_test, y_Subset_train, y_Subset_test = FullQ_Train_Test_Split

    # SoS = Sum of Scores. Sum the scores for the questions in the training set.
    y_train_SoS = X_Subset_train.loc[:, Questionnaire_Subset].sum(axis=1)

    # Bin the sum of scores of the training set determined by the bin size in "TheBinSizeList".
    Subset_Binned_Level_y_train = DimensionalBinChoice(y_train_SoS, TheBinsizeList)

    # Create an array of the feature (question) values of questions defined in "Questionnaire_Subset".
    Train_Columns_Subset = np.asarray(X_Subset_train.loc[:, Questionnaire_Subset])
    # Create an array of target values of the training data comprised of the binned values from the subsets of questions
    # Sum of Scores (SoS).
    TrainTarget_Columns_Subset = np.asarray(Subset_Binned_Level_y_train)

    # start the binning from 0 for each of the subset binning.
    TrainTarget_Columns_Subset = TrainTarget_Columns_Subset - 1

    # Create the neural network model from the dictionary of (hyper)parameters.
    model_Subset = KerasModel(ParameterDictionary_Subset)
    # fit the model
    model_Subset.fit(Train_Columns_Subset, TrainTarget_Columns_Subset, epochs=30, batch_size=30, verbose=0)

    # Create an array of the testing feature data.
    Test_Columns_Subset = np.asarray(X_Subset_test.loc[:, Questionnaire_Subset])

    # Make a prediction from the test data.
    predictions_Subset = model_Subset.predict(Test_Columns_Subset)
    # Choose bin based on highest percentage probability.
    max_indices_Subset = np.argmax(predictions_Subset, axis=1)

    # Return the predictions.
    return max_indices_Subset

# Fit the grid model then sort, print and write the data to file.
def GridResults(grid, Train_Columns_Arr, TrainTarget_Columns_Arr, Test_Columns_Arr, TestTarget_Columns_Arr, resultsfile):
    # fitting the model for grid search
    grid.fit(Train_Columns_Arr, TrainTarget_Columns_Arr)

    # Pring the best grid scores.
    print(grid.best_score_)
    print(grid.best_params_)

    # Print a few columns to file.
    simple_results = pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
    simple_results.sort_values(by='mean_test_score', inplace=True, ascending=False)
    simple_results.to_csv(resultsfile + '.csv')

    # Print all grid results to file.
    results = pd.DataFrame(grid.cv_results_)
    results.sort_values(by='rank_test_score', inplace=True, ascending=True)
    results.to_csv(resultsfile + '_details.csv', encoding='utf-8', index=False)

    # Make a prediction using the particular grid model.
    grid_predictions = grid.predict(Test_Columns_Arr)

    # print classification report
    print(classification_report(TestTarget_Columns_Arr, grid_predictions))

# Create and return a hyperparameter dictionary from the best found neural network (NN) hyperparameters via GridSearch.
def NNParameters(InputDim, OutputDim):
    LossFunction, Layer1_Neurons, Layer2_Neurons, ActivationFunction, Optimizer, FittingMetric = \
        'sparse_categorical_crossentropy', 50, 60, 'sigmoid', 'RMSProp', 'accuracy';

    # Create the dictionary of parameters.
    ParameterDictionary = {'input_dim': InputDim, 'activation': ActivationFunction,
                           'L1Neurons': Layer1_Neurons, 'L2Neurons': Layer2_Neurons,
                           'output': (OutputDim), 'loss': LossFunction, 'optimizer': Optimizer,
                           'metrics': FittingMetric}

    return ParameterDictionary

# Calculate the SHAP values using the DeepExlainer which is specific computing SHAP values with a neural network model.
def SHAPInfo(model, X_Questionnaire_train_vals, X_Questionnaire_test_vals):

    explainer = shap.DeepExplainer(model, X_Questionnaire_train_vals)
    shap_values = explainer.shap_values(X_Questionnaire_test_vals)
    return shap_values

