# Good to go.
# Import external packages.

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.options.display.max_columns = None
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth', 150)
import numpy as np
from itertools import product
np.set_printoptions(threshold=sys.maxsize)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# local imports
from SupportFunctions import Train_Test_sets, ChoiceImputation, BinConfigurations
from NeuralNetworkMethods import  GridResults
from Questionnaires import PSWQ_Dutch_Positve

# Create a list of tuples of neurons of different sized layers for GridSearch.
def LayersofNodes():
    # Create a list of layers with all possible permutations of number of neurons of a set interval.
    neurons = np.arange(10, 61, 10).tolist()
    layers = 3


    # Create a list of layers consisting of tuples of neurons of all permutations of neurons of a set interval.
    LayersList = []
    for i in range(1, layers + 1):
        for j in product(neurons, repeat=i):
            LayersList.append(j)

    return LayersList

# Create the neural network model.
def createmodel(n_layers, optimizer, activation_func, loss_func):
    model = Sequential()
    for i, neurons in enumerate(n_layers):
        if i == 0:
            model.add(Dense(neurons, input_dim=X_Questions.shape[1], activation=activation_func))
        else:
            model.add(Dense(neurons, activation=activation_func))


    model.add(Dense(len(TheBinsizeList)+1, activation=activation_func))
    model.compile(optimizer=optimizer, loss=loss_func, metrics=["accuracy"])

    return model

# Import the particular dataset.
PSWQ = PSWQ_Dutch_Positve()

# If other questionnaires are added, cleaned, filtered etc. they can be concatenated here.
Questionnaire = pd.concat([PSWQ], axis=1)

# Choose whether to impute or drop subjects with missing item scores (1 = impute, 0 = drop columns with missing data)
Imputationchoice = 1
Questionnaire = ChoiceImputation(Questionnaire, Imputationchoice)

# Get the bin-configurations
bins = BinConfigurations()

# Sum item scores.
Questionnaire["Sum_of_Scores"] = Questionnaire.iloc[:, :].sum(axis=1)

# Choose the number of subjects to use in classification model.
Number_of_Subjects = len(PSWQ)

# Randomized dataframe of the participants with their answered questions.
Subjects = Questionnaire.sample(Number_of_Subjects)

# Get the questions which will be used in training and testing.
X_Questions = Subjects.iloc[:, 0:-1]

# Get the Sum of Scores (SoS) which will be binned and used as the dependent variable/target to predict.
y_SoS = Subjects[["Sum_of_Scores"]]

# Define parameters for GridSearch
activation_funcs = ['sigmoid','relu']
loss_funcs = ['sparse_categorical_crossentropy','categorical_crossentropy']
optimizers = ['adam','RMSProp']
LayersList = LayersofNodes()


# For all bin-size configurations in the list of bins in "bins"
for BinsizeName, TheBinsizeList in bins:

    # Train and Test the the particular configuration in "TheBinsizeList"
    Train_Columns_Arr, TrainTarget_Columns_Arr, Test_Columns_Arr, TestTarget_Columns_Arr \
        = Train_Test_sets(X_Questions, y_SoS, TheBinsizeList)

    # Wrap model into scikit-learn
    # Note: Many other activation functions, optimizers, loss functions etc. were tried, but due to an explosion of
    # parameter combinations, those below performed the best.
    model = KerasClassifier(build_fn=createmodel, verbose=False)

    # Create the grid of parameters which to train GridSearch on.
    param_grid = dict(n_layers=LayersList,optimizer=optimizers,
                      activation_func=activation_funcs, loss_func=loss_funcs, batch_size=[32], epochs=[10,30])

    # Create the Grid for the particular model.
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, verbose=2)

    # fit the model and print the Grid results.
    GridResults(grid, Train_Columns_Arr, TrainTarget_Columns_Arr, Test_Columns_Arr, TestTarget_Columns_Arr,
                'GridOptimizationNN')