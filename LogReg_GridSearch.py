# Good to go.
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.options.display.max_columns = None
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth', 150)
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# local imports
from SupportFunctions import Train_Test_sets, ChoiceImputation, BinConfigurations
from NeuralNetworkMethods import  GridResults
from Questionnaires import PSWQ_Dutch_Positve

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

# For all bin-size configurations in the list of bins in "bins"
for BinsizeName, TheBinsizeList in bins:

    # Train and Test the the particular configuration in "TheBinsizeList"
    Train_Columns_Arr, TrainTarget_Columns_Arr, Test_Columns_Arr, TestTarget_Columns_Arr \
        = Train_Test_sets(X_Questions, y_SoS, TheBinsizeList)

    # defining parameter range
    param_grid = {'C': [0.01, 0.1, 1, 2, 10, 100,1000], 'penalty': ['l1', 'l2']}

    # Prepare the Grid.
    grid = GridSearchCV(LogisticRegression(), param_grid, refit=True, verbose=3, cv=10)

    # fit the model and print the Grid results.
    GridResults(grid, Train_Columns_Arr, TrainTarget_Columns_Arr, Test_Columns_Arr, TestTarget_Columns_Arr,
                'GridOptimizationLogReg')
