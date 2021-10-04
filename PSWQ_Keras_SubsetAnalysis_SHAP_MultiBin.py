# Responsible for producing Table 1, Table 2, and Figure 4.

# This program computes metrics corresponding to different question configurations, both from the full set of questions
# and the subset of questions used to create the abbreviated questionnaires.

# Import external packages.
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.options.display.max_columns = None
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth', 150)
import sys
import seaborn as sns
import numpy as np
import csv
np.set_printoptions(threshold=sys.maxsize)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
from statistics import mean
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import tensorflow as tf

# Import local packages.
from DimensionalityBinning import DimensionalBinChoice
from SupportFunctions import ChoiceImputation, Standardize, RoundandPercent, AverageArr
from NeuralNetworkMethods import KerasModel, Subset_Analysis

# Create a Dataframe from dataset.
data = pd.read_csv('PSWQ_Dutch.csv', sep=',')
# Will hold question configurations from file.
SHAP_Orderings = []
# Open and pull in lines nad question configurations from file.
# with open('SHAP_QuestionOrderings.txt') as f:
with open('SHAP_QuestionOrderings.txt') as f:
    f_csv = csv.reader(f)
    headers = next(f_csv)
    lineiterator = 0
    # Add only those lines that are question configurations.
    for row in f_csv:
        if row:
            if not row[0].startswith("#"):
                SHAP_Orderings.append(row)
            lineiterator += 1

# Clean
SHAP_Orderings= [[x.strip() for x in y] for y in SHAP_Orderings]
# Create Dataframe from the list.
SHAP_Orderings_df = pd.DataFrame.from_records(SHAP_Orderings)
# Create and map first elements to the BinSize column
mapping = {SHAP_Orderings_df.columns[0]: 'BinSize'}
SHAP_Orderings_df = SHAP_Orderings_df.rename(columns=mapping)
# Filter those columns that match "PSWQ_".
PSWQ = data.filter(like='PSWQ_', axis=1)
# Remove reverse coded questions.
PSWQ.drop(PSWQ.columns[[0, 2, 7, 9, 10]], axis=1, inplace=True)
# If other questionnaires are added, cleaned, filtered etc. they can be concatenated here.
Questionnaire = pd.concat([PSWQ], axis=1)
# Choose whether to impute or drop subjects with missing item scores.
Imputationchoice = 1
Questionnaire = ChoiceImputation(Questionnaire, Imputationchoice)
# Sum item scores.
Questionnaire["Sum_of_Scores"] = Questionnaire.iloc[:, :].sum(axis=1)
# Choose the number of subjects to use in classification model.
Number_of_Subjects = [len(data)]
# Defines the number and AUC of each bin array.
# TwoBinMod_Low = [.842]
# TwoBinMod_Ave= [.842]
# TwoBinMod_High = [.842]
TwoBin = [0]
ThreeBin = [-.431, .431]
FourBin = [-.674, 0, .674]
FiveBin = [-.842, -0.253, 0.253, .842]
SixBin = [-0.967, -0.431, 0, 0.431, 0.967]
SevenBin = [-1.068, -0.566, -0.18, 0.18, 0.566, 1.068]
EightBin = [-1.15, -.674, -.319, 0, .319, .674, 1.15]

# Create a list for all bin arrays that will be used to build classification models.
bins = []
# We choose the number of and levels of the partitions. Comment out those arrays of partitions that are not in the
# input file containing the question configurations, e.g. if ThreeBin is not in the file containing the question
# configurations, comment it out here.
# bins.append(('Low',TwoBinMod_Low))
# bins.append(('Ave',TwoBinMod_Ave))
# bins.append(('High',TwoBinMod_High))
bins.append(('2-Bin', TwoBin))
bins.append(('3-Bin', ThreeBin))
bins.append(('4-Bin',FourBin))
bins.append(('5-Bin',FiveBin))
bins.append(('6-Bin',SixBin))
bins.append(('7-Bin',SevenBin))
bins.append(('8-Bin',EightBin))

# Define hyperparameters for the neural network type which will be used to create models for all classifications.
LossFunction, Layer1_Neurons, ActivationFunction, Optimizer, FittingMetric  =\
    'categorical_crossentropy', 60, 'sigmoid', 'Adam', 'accuracy';

# Create dummy dimensions for the NN. This is done to update the number of input features (= questions = DummyInputDim)
# and the number of output bins (=size of the bin array = DummyOutputDim) within the inner loops without having to
# rebuild the NN.
DummyInputDim , DummyOutputDim = 0,0 ;
# Creates a dictionary of hyperparameters and input and output sizes.
ParameterDictionary_Subset = {'L1Neurons': Layer1_Neurons, 'input_dim': DummyInputDim,
                              'activation': ActivationFunction,
                              'L2Neurons': (math.ceil(Layer1_Neurons / 2)), 'output': DummyOutputDim,
                              'loss': LossFunction, 'optimizer': Optimizer,
                              'metrics': FittingMetric}
# Create lists for the metrics we want to keep track of.
FullQuestionnaireMetrics = ['Set','Subjects', 'Binsize', 'Accuracy', 'Precision', 'Recall', 'F1', 'Model']
BriefQuestionnaireMetrics = ['Set', 'Subjects', 'Binsize', 'Accuracy', 'Precision', 'Recall', 'F1', 'SubsetInfo']
# Create df's for holing the output data for both the whole questoinnaire as well as each of the brief questionnaires.
ScoringMetricsConcat_DF = pd.DataFrame(columns=FullQuestionnaireMetrics)
ScoringMetricsConcat_DF_Subset = pd.DataFrame(columns=BriefQuestionnaireMetrics)
#
SubjectsandQuestions_DF, SubjectsandQuestions_DF_Subset  = pd.DataFrame(), pd.DataFrame()

# Number of Trials to perform (which the metrics are then averaged over).
NumTrials = 1

print(str(Layer1_Neurons) + ', ' + ActivationFunction + ', ' + Optimizer + ', ' + FittingMetric)

BinNames = []
Bin_Iterator = 0
# Do all computations for each bin array selected above.
for BinsizeName, TheBinsizeList in bins:
    Bin_Iterator = Bin_Iterator + 1
    BinNames.append(BinsizeName)
    print("For the BinSize " + BinsizeName)
    # Extract the row of questions (from file) associated to each bin array.
    RowofQuestions = SHAP_Orderings_df[SHAP_Orderings_df['BinSize'].str.contains(BinsizeName)]
    [Subset_TrainAndTest] = RowofQuestions.iloc[:, 1:].values.tolist()
    # Update the output size of the neural network to be that of the size of the current bin array.
    Output = {'output': len(TheBinsizeList) + 1}
    ParameterDictionary_Subset.update(Output)
    # These lists will hold the four tracked metrics for classifications over the bin array using all questions.
    Accuracy_Subset_AllQuestions_Arr, Precision_Subset_AllQuestions_Arr, Recall_Subset_AllQuestions_Arr,\
        Fscore_Subset_AllQuestions_Arr = [],[],[],[];
    # These lists hold the set of scoring metrics for the full and abbreviated questionnaires.
    ScoringMetrics, ScoringMetrics_Subset = [],[];

    for SubjectSubsetNumber in range(len(Number_of_Subjects)):
        # Holds the classification metrics for the full and abbreviated metrics at the model level (before averaging).
        AccuracyArr, PrecisionArr, RecallArr, FscoreArr = [],[],[],[];
        AccuracyArr_Subset,PrecisionArr_Subset,RecallArr_Subset,FscoreArr_Subset = [],[],[],[];

        print('Subject Subset counter ' + str(SubjectSubsetNumber + 1) + " In the Binsize " + BinsizeName)
        Subjects = Number_of_Subjects[SubjectSubsetNumber]

        # Randomized dataframe of subjects/questions.
        Questionnaire_Subjects_Subset = Questionnaire.sample(Subjects)

        # Do the computations for the number trials (which the metrics will be averaged over).
        for TrialNumber in range(NumTrials):
            print('Trial Number ' + str(TrialNumber + 1) + ' For the Subject Subset counter ' + str(SubjectSubsetNumber + 1) +
                  " In the Binsize " + BinsizeName)

            # Set of independent features (questions) .
            X_Questionnaire = Questionnaire_Subjects_Subset.iloc[:, 0:-1]
            # Set of dependent features (sum of scores)
            y_Questionnaire = Questionnaire_Subjects_Subset[["Sum_of_Scores"]]

            # Training and testing split.
            X_Questionnaire_train, X_Questionnaire_test, y_Questionnaire_train, y_Questionnaire_test = train_test_split(
                X_Questionnaire, y_Questionnaire, test_size=.2)

            # Standardizing the training set (apart from the test set).
            X_Questionnaire_train = Standardize(X_Questionnaire_train)
            y_Questionnaire_train = Standardize(y_Questionnaire_train)

            # Turning the training set into arrays for faster computation.
            Train_Columns = np.asarray(X_Questionnaire_train.iloc[:, :len(X_Questionnaire_train.columns)])

            # Bin the sum of scores of the training set.
            Questionnaire_Binned_Level_y_train = DimensionalBinChoice(y_Questionnaire_train['Sum_of_Scores'],
                                                                      TheBinsizeList)

            # Create an array of the Target values of the training data.
            TrainTarget_Columns = np.asarray(Questionnaire_Binned_Level_y_train)
            TrainTarget_Columns = TrainTarget_Columns - 1

            # Standardizing the test set (apart from the training set).
            X_Questionnaire_test = Standardize(X_Questionnaire_test)
            y_Questionnaire_test = Standardize(y_Questionnaire_test)

            # Bin the sum of scores of the test set.
            Questionnaire_Binned_Level_y_test = DimensionalBinChoice(y_Questionnaire_test['Sum_of_Scores'],
                                                                     TheBinsizeList)
            # Create an array of the testing data.
            Test_Columns = np.asarray(X_Questionnaire_test.iloc[:, :len(X_Questionnaire_test.columns)])
            TestTarget_Columns = np.asarray(Questionnaire_Binned_Level_y_test)

            # Change bin levels to start from 0 instead of 1.
            TestTarget_Columns = TestTarget_Columns - 1
            # One hot encode the training targets.
            TrainTarget_Columns_oneHot = tf.one_hot(TrainTarget_Columns, len(TheBinsizeList) + 1)

            # Set the (hyper)parameters for the neurla network.
            ParameterDictionary = {'L1Neurons': Layer1_Neurons, 'input_dim': len(X_Questionnaire_train.columns),
                                   'activation': ActivationFunction,
                                   'L2Neurons': (math.ceil(Layer1_Neurons / 2)), 'output': (len(TheBinsizeList) + 1),
                                   'loss': LossFunction, 'optimizer': Optimizer,
                                   'metrics': FittingMetric}

            # Create the neural network.
            model = KerasModel(ParameterDictionary)
            # Fit the neural network to the training data.
            model.fit(Train_Columns, TrainTarget_Columns_oneHot, epochs=30, batch_size=10, verbose=0)

            # Make a prediction from the test data.
            predictions = model.predict(Test_Columns)
            # Choose bin based on highest percentage probability.
            max_indices = np.argmax(predictions, axis=1)

            # Compute metrics by comparing actual (TestTarget) vs. predicted (max_indices).
            Accuracy = accuracy_score(TestTarget_Columns, max_indices)
            Precision = precision_score(TestTarget_Columns, max_indices, average="macro")
            Recall = recall_score(TestTarget_Columns, max_indices, average="macro")
            Fscore = f1_score(TestTarget_Columns, max_indices, average="macro")

            # These lists will hold the four tracked metrics for classifications over the bin array using subsets of questions.
            Accuracy_Subset_AllQuestions, Precision_Subset_AllQuestions, Recall_Subset_AllQuestions,\
                Fscore_Subset_AllQuestions= [],[],[],[];

            Train_Test_List = [X_Questionnaire_train, X_Questionnaire_test, y_Questionnaire_train, y_Questionnaire_test]
            # Holds the list of questions to be tested.
            Subset_to_Test = []
            for Question in Subset_TrainAndTest:
                # Iteratively build a larger list Question by question.
                Subset_to_Test.append(Question)
                # Update the input dimension based on how large the subset of questions is.
                Input = {'input_dim': len(Subset_to_Test)}
                ParameterDictionary_Subset.update(Input)

                # Choose bin based on highest percentage probability.
                max_indices_Subset = Subset_Analysis(Train_Test_List, ParameterDictionary_Subset, TheBinsizeList,
                                                     Subset_to_Test)

                # Note that we are testing the Subset predictions (max_indices_Subset) against the full measure
                # preditions (TestTarget_Columns). We don't want to test the Target Columns of the subset against the
                # Subset predictions as we want to compare the subset predictions against the full measure target values.

                # Compute the metrics of each of the subsets (brief questionnaires), iteratively (question by question)
                # and append those values to an array holding the values for each of the metrics. For example, for a 3
                # question brief, there will be one, two, and finally three values in each of the lists being appended.
                Accuracy_Subset = accuracy_score(TestTarget_Columns, max_indices_Subset)
                Accuracy_Subset_AllQuestions.append(Accuracy_Subset)
                Precision_Subset = precision_score(TestTarget_Columns, max_indices_Subset, average="macro")
                Precision_Subset_AllQuestions.append(Precision_Subset)
                Recall_Subset = recall_score(TestTarget_Columns, max_indices_Subset, average="macro")
                Recall_Subset_AllQuestions.append(Recall_Subset)
                Fscore_Subset = f1_score(TestTarget_Columns, max_indices_Subset, average="macro")
                Fscore_Subset_AllQuestions.append(Fscore_Subset)

            # Append the metrics for the full questionnaires for each trial to an array.
            AccuracyArr.append(Accuracy)
            PrecisionArr.append(Precision)
            RecallArr.append(Recall)
            FscoreArr.append(Fscore)
            # Append the lists for each metric (for each brief) for each trial to an array.
            Accuracy_Subset_AllQuestions_Arr.append(Accuracy_Subset_AllQuestions)
            Precision_Subset_AllQuestions_Arr.append(Precision_Subset_AllQuestions)
            Recall_Subset_AllQuestions_Arr.append(Recall_Subset_AllQuestions)
            Fscore_Subset_AllQuestions_Arr.append(Fscore_Subset_AllQuestions)

            # release the memory from building the model.
            K.clear_session()

        # Average the metrics for each of the full questionnaires over all of the trials.
        AccuracyAve = mean(AccuracyArr)
        PrecisionAve = mean(PrecisionArr)
        RecallAve = mean(RecallArr)
        FscoreAve = mean(FscoreArr)

        # Average the lists of metrics for each of the brief questionnaires over all of the trials.
        AccuracyAve_Subset = RoundandPercent(AverageArr(Accuracy_Subset_AllQuestions_Arr))
        PrecisionAve_Subset = RoundandPercent(AverageArr(Precision_Subset_AllQuestions_Arr))
        RecallAve_Subset = RoundandPercent(AverageArr(Recall_Subset_AllQuestions_Arr))
        FscoreAve_Subset = RoundandPercent(AverageArr(Fscore_Subset_AllQuestions_Arr))

        # Create a string to display the hyperparameters used in the neural network model.
        ModelInfo = 'Activation: ' + str(ActivationFunction) + ', Layer 1: ' + str(
            Layer1_Neurons) + ', LossFunction: ' + \
                    LossFunction + ', Optimizer: ' + Optimizer + ', FittingMetric: ' + FittingMetric + ', Questions: ' + \
                    str(len(X_Questionnaire_train.columns)) + ', OutputBins: ' + str(len(TheBinsizeList) + 1)

        # For each average of trials, Append the scoring metrics and model info for the full questionnaire.
        ScoringMetrics.append(
            [Bin_Iterator, Subjects, len(TheBinsizeList) + 1, AccuracyAve, PrecisionAve, RecallAve, FscoreAve, ModelInfo])

        Subset_Info = 'Questions: ' + \
                      str(len(Subset_TrainAndTest)) + ', OutputBins: ' + str(len(TheBinsizeList) + 1)

        # For each average of trials, append the scoring metrics and model info for the full questionnaire.
        ScoringMetrics_Subset.append(
            [Bin_Iterator, Subjects, len(TheBinsizeList) + 1, tuple(AccuracyAve_Subset), tuple(PrecisionAve_Subset),
             tuple(RecallAve_Subset),tuple(FscoreAve_Subset), Subset_Info])

        # Create dataframes for both the full and abbreviated questionnaires.
        SubjectsandQuestions_DF = pd.DataFrame.from_records(ScoringMetrics,columns=FullQuestionnaireMetrics)
        SubjectsandQuestions_DF_Subset = pd.DataFrame.from_records(ScoringMetrics_Subset,columns=BriefQuestionnaireMetrics)

    # Append the full and abbreviated questionnaires for each new bin array.
    ScoringMetricsConcat_DF = pd.concat([ScoringMetricsConcat_DF, SubjectsandQuestions_DF], axis=0)
    ScoringMetricsConcat_DF_Subset = pd.concat([ScoringMetricsConcat_DF_Subset, SubjectsandQuestions_DF_Subset], axis=0)

# Create a list of numbers to denote questions in the graph.
QuestionIterator = list(range(1, len(Subset_TrainAndTest) + 1))

# Add all metrics to the graph.
Yaxis = ["Accuracy", "Precision", "Recall", "F1"]
# Graph the results.
for l in Yaxis:
    df_long = ScoringMetricsConcat_DF_Subset.explode(l).reset_index()
    df_long.drop('index', axis=1, inplace=True)
    df_long['Questions'] = np.tile(QuestionIterator, len(ScoringMetricsConcat_DF_Subset))
    df_long[l] = df_long[l].astype(float)
    g = sns.relplot(x='Questions', y=l, hue="Set",
                    data=df_long, height=5, aspect=.8, kind='line')
    g._legend.remove()
    g.fig.suptitle(l + " Score")
    g.fig.subplots_adjust(top=.95)
    g.ax.set_xlabel('Questions', fontsize=12)
    g. ax.set_ylabel(l, fontsize=12)
    plt.xticks(QuestionIterator)
    legend_title = 'Bins/Levels'
    g._legend.set_title(legend_title)
    # Create new labels for the legend.
    Binsize_list = ScoringMetricsConcat_DF_Subset['Binsize'].tolist()
    new_labels = [str(x) for x in BinNames]
    for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
    plt.legend(title='Configurations', loc='lower right', labels=new_labels)
    g.tight_layout()

plt.show()
