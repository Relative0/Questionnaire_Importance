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
np.set_printoptions(threshold=sys.maxsize)
from matplotlib import pyplot as plt
import seaborn as sns
from statistics import mean

# Local Imports
from SupportFunctions import ChoiceImputation, BinConfigurations, Train_Test_sets, Scoring
from NeuralNetworkMethods import KerasModel, NNParameters
from Questionnaires import PSWQ_Dutch_Positve
from Models import AlgorithmModels

# Import the particular dataset.
PSWQ = PSWQ_Dutch_Positve()

# If other questionnaires are added, cleaned, filtered etc. they can be concatenated here.
Questionnaire = pd.concat([PSWQ], axis=1)

# Choose whether to impute or drop subjects with missing item scores.
Imputationchoice = 1
Questionnaire = ChoiceImputation(Questionnaire, Imputationchoice)

# Sum item scores.
Questionnaire["Sum_of_Scores"] = Questionnaire.iloc[:, :].sum(axis=1)

# Choose the number of subjects to use in classification model.
# Could have a list of subject sizes e.g., [50,100, 200, ..., len(PSWQ)]
Number_of_Subjects = [len(PSWQ)]

# Get the bin-configurations.
bins = BinConfigurations()

# Get the name and bin list.
BinName, BinsizeList = bins[0]

# Sizes of NN input and output layers
InputDim = len(Questionnaire.iloc[:, 0:-1].columns)
OutputDim = len(BinsizeList) + 1

# Create the dictionary of NN hyperparameters.
ParameterDictionary = NNParameters(InputDim, OutputDim)

# Create the NN model from hyperparameters.
model_Keras = KerasModel(ParameterDictionary)

# Return all of the models to test.
models = AlgorithmModels()

# Choose the number of subjects to test.
Subjects = len(Questionnaire)

# Randomized dataframe of subjects/questions.
Subjects_Subset = Questionnaire.sample(Subjects)

# Get the subject questions as the independent variable, and y_SoS) the sum of scores (SoS) as the dependent variable.
X_Questions = Subjects_Subset.iloc[:, 0:-1]
y_SoS = Subjects_Subset[["Sum_of_Scores"]]

# Create dataframes for storing found metrics.
Metrics = ['Subjects', 'Binsize', 'ModelName', 'Accuracy', 'Precision', 'Recall', 'F1']
ScoringMetricsConcat_DF = pd.DataFrame(columns=Metrics)
SubjectsandQuestions_DF = pd.DataFrame()

# Number of trials we want to average over.
NumTrials = 2

# For each of the models wanted to be tested:
for Modelname,model in models:
    print(Modelname)
    ScoringMetrics = []

    # For each of set of subjects we can compare the algorithms.
    for SubjectSubsetNumber in range(len(Number_of_Subjects)):
        print('Subject Subset counter' + str(SubjectSubsetNumber))
        Subjects = Number_of_Subjects[SubjectSubsetNumber]
        PSWQ_Subset = Questionnaire.sample(Subjects)
        PSWQ_Columns = list(PSWQ_Subset.columns)

        # Create lists to hold each of the metrics from the individual trials (they will then be averaged).
        AccuracyList = []; PrecisionList = []; RecallList = []; FscoreList = []

        # Need to do multiple tests now.
        for TrialNumber in range(NumTrials):
            print(TrialNumber)

            # Train and Test the the particular configuration in "TheBinsizeList"
            Train_Columns_Arr, TrainTarget_Columns_Arr, Test_Columns_Arr, TestTarget_Columns_Arr \
                = Train_Test_sets(X_Questions, y_SoS, BinsizeList)

            # NN takes some different arguments than many other algorithms.
            if Modelname == 'NN':
                Output = {'output': len(BinsizeList) + 1}
                ParameterDictionary.update(Output)
                # Fit the neural network to the training data.
                model_Keras.fit(Train_Columns_Arr, TrainTarget_Columns_Arr, epochs=30, batch_size=32, verbose=0)
                # Make a prediction from the test data.
                predictions = model_Keras.predict(Test_Columns_Arr)
                # Choose bin based on highest percentage probability.
                predictions = np.argmax(predictions, axis=1)
            else:
                model.fit(Train_Columns_Arr, TrainTarget_Columns_Arr)
                predictions = model.predict(Test_Columns_Arr)

            # Find the prediction scores for the algorithm at hand.
            Accuracy, Precision, Recall, Fscore = Scoring(TestTarget_Columns_Arr, predictions)

            # Append the metrics for each of the trials to their corresponding lists.
            AccuracyList.append(Accuracy)
            PrecisionList.append(Precision)
            RecallList.append(Recall)
            FscoreList.append(Fscore)

        # Find the average of each of the metrics.
        AccuracyAve = mean(AccuracyList)
        PrecisionAve = mean(PrecisionList)
        RecallAve = mean(RecallList)
        FscoreAve = mean(FscoreList)

        # For each model append the average metrics to the list holding metrics for all model predictions.
        ScoringMetrics.append(
            [Subjects, BinName, Modelname, AccuracyAve, PrecisionAve, RecallAve, FscoreAve])
        SubjectsandQuestions_DF = pd.DataFrame.from_records(ScoringMetrics, columns=Metrics)

    # Concatanate each of the algorithms scoring metrics into a dataframe.
    ScoringMetricsConcat_DF = pd.concat([ScoringMetricsConcat_DF, SubjectsandQuestions_DF], axis=0)

# Format the display of what is printed to the console.
with pd.option_context('display.max_rows', None):
    with pd.option_context('display.max_columns', None): # more options can be specified also
        print(ScoringMetricsConcat_DF)

# Graph the algorithm comparisons for each of the metrics.
Yaxis = ["Accuracy", "Precision", "Recall", "F1"]
for l in Yaxis:
    g = sns.catplot(x="ModelName", y=l, kind="bar", size=4, aspect=1, data=ScoringMetricsConcat_DF,
                    palette="Set1", legend=False)
    g.fig.suptitle(bins[0][0] + ": " + l + " Score", fontsize=12)
    g.set_ylabels(l, fontsize=12)
    g.set(ylim=(.9, 1))
    g.set_xlabels("Model", fontsize=12)
    g.set_xticklabels(size=10, rotation=90)
    plt.subplots_adjust(top=.9)
    plt.subplots_adjust(bottom=0.35)
    plt.subplots_adjust(left=0.25)

plt.show()
