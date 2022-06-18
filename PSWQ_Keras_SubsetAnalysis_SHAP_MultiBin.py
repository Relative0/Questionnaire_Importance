# Good to go.
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
np.set_printoptions(threshold=sys.maxsize)
import csv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from statistics import mean
from tensorflow.keras import backend as K

# Import local packages.
from DimensionalityBinning import DimensionalBinChoice
from SupportFunctions import ChoiceImputation, Standardize, RoundandPercent, AverageList, StdDevList, Scoring, \
    BinConfigurations, BinConfigurations_Toppers, BinConfiguration_Quartile_ForwardReverse,\
    BinConfiguration_HighLowAve_TopOctile, BinConfiguration_HighLowAve_TopQuartile

from NeuralNetworkMethods import KerasModel, Subset_Analysis, NNParameters

# Create a Dataframe from dataset.
data = pd.read_csv('PSWQ_Dutch.csv', sep=',')

# Will hold question configurations from file.
SHAP_Orderings = []

# Open and pull in lines and question configurations from file.
with open('SHAP_QuestionOrderings.txt') as f:
    f_csv = csv.reader(f)
    headers = next(f_csv)
    lineiterator = 0
    # Add only those lines that are question configurations.
    for row in f_csv:
        if row:
            # If the row doesn't start with a comment # then append it.
            if not row[0].startswith("#"):
                SHAP_Orderings.append(row)
            lineiterator += 1

# Clean whitespaces from strings.
SHAP_Orderings= [[x.strip() for x in y] for y in SHAP_Orderings]

# Create Dataframe from the list.
SHAP_Orderings_df = pd.DataFrame.from_records(SHAP_Orderings)

# Create and map the name 'BinSize' to the first column.
mapping = {SHAP_Orderings_df.columns[0]: 'BinSize'}

# Rename the columns to be of that mapped.
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
# Could have a list of subject sizes e.g., [50,100, 200, ..., len(PSWQ)]
Number_of_Subjects = [len(data)]

# We choose the number of and levels of the partitions. Comment out those arrays of partitions that are not in the
# input file containing the question configurations, e.g. if ThreeBin is not in the file containing the question
# configurations, comment it out here.
bins = BinConfigurations()
# bins = BinConfigurations_Toppers()
# bins = BinConfiguration_Quartile_ForwardReverse()
# bins = BinConfiguration_HighLowAve_TopQuartile()
# bins = BinConfiguration_HighLowAve_TopOctile()

# Create dummy dimensions for the NN. This is done to update the number of input features (= questions = DummyInputDim)
# and the number of output bins (=size of the bin array = DummyOutputDim) within the inner loops without having to
# rebuild the NN.
DummyInputDim , DummyOutputDim = 0,0 ;

# Define hyperparameters for the neural network type which will be used to create models for all classifications.
ParameterDictionary_Subset = NNParameters(DummyInputDim,DummyOutputDim)

# Create lists for the metrics we want to keep track of.
FullQuestionnaireMetrics = ['Model', 'Set','Subjects', 'Binsize', 'Accuracy', 'Precision', 'Recall', 'F1' ]
BriefQuestionnaireMetrics = [ 'SubsetInfo', 'Set', 'Subjects', 'Binsize', 'Accuracy', 'Precision', 'Recall', 'F1']
FullQuestionnaire_StDevMetrics = ['Accuracy_StDev', 'Precision_StDev', 'Recall_StDev', 'F1_StDev']

# Create dataframes for holding output data for both the whole questionnaire as well as each of the brief questionnaires
# (Subsets).
ScoringMetricsConcat_DF = pd.DataFrame(columns=FullQuestionnaireMetrics)
ScoringMetricsConcat_DF_Subset = pd.DataFrame(columns=BriefQuestionnaireMetrics)
ScoringMetricsStDevConcat_DF = pd.DataFrame(columns=FullQuestionnaire_StDevMetrics)
SubjectsandQuestions_DF, SubjectsandQuestions_DF_Subset  = pd.DataFrame(), pd.DataFrame()

# Number of Trials to perform (which the metrics are then averaged over).
NumTrials = 2

BinNames = []
Bin_Iterator = 0

# Do all computations for each bin configuration.
for BinsizeName, TheBinsizeList in bins:
    Bin_Iterator = Bin_Iterator + 1
    BinNames.append(BinsizeName)
    print("For the BinSize " + BinsizeName)

    # Extract the row of questions (from file) associated to each bin array.
    RowofQuestions = SHAP_Orderings_df[SHAP_Orderings_df['BinSize'].str.contains(BinsizeName)]

    # Create a list and pull off the outer [ ] brackets.
    [Subset_TrainAndTest] = RowofQuestions.iloc[:, 1:].values.tolist()

    # Remove None values in list.
    Subset_TrainAndTest = list(filter(None, Subset_TrainAndTest))

    # Update the output size of the neural network to be that of the size of the current bin array.
    Output = {'output': len(TheBinsizeList) + 1}
    ParameterDictionary_Subset.update(Output)

    # These lists will hold the four tracked metrics for classifications over the bin array using all questions.
    Accuracy_Subset_AllQuestions_List, Precision_Subset_AllQuestions_List, Recall_Subset_AllQuestions_List, \
    Fscore_Subset_AllQuestions_List = [], [], [], [];

    # These lists hold the set of scoring metrics for the full and abbreviated questionnaires.
    ScoringMetrics, ScoringMetrics_Subset, StDevMetrics = [],[],[];

    # For the current number of subjects:
    for SubjectSubsetNumber in range(len(Number_of_Subjects)):
        # Holds the classification metrics for the full and abbreviated metrics at the model level (before averaging).
        AccuracyList, PrecisionList, RecallList, FscoreList = [], [], [], [];
        AccuracyArr_Subset,PrecisionArr_Subset,RecallArr_Subset,FscoreArr_Subset = [],[],[],[];

        print('Subject Subset counter ' + str(SubjectSubsetNumber + 1) + " In the Binsize " + BinsizeName)

        # Randomized dataframe of subjects/questions.
        Subjects = Number_of_Subjects[SubjectSubsetNumber]
        Questionnaire_Subjects_Subset = Questionnaire.sample(Subjects)

        # Do the computations for the number trials (which the metrics will be averaged over).
        for TrialNumber in range(NumTrials):
            print('Trial Number ' + str(TrialNumber + 1) + ' For the Subject Subset counter ' +
                  str(SubjectSubsetNumber + 1) + " In the Binsize " + BinsizeName)

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

            # define the input and output layer sizes and create the dictionary of neural network parameters.
            inDim = len(X_Questionnaire_train.columns)
            outDim =  len(TheBinsizeList) + 1
            ParameterDictionary = NNParameters(inDim,outDim)

            # Create the neural network.
            model = KerasModel(ParameterDictionary)
            # Fit the neural network to the training data.
            model.fit(Train_Columns, TrainTarget_Columns, epochs=30, batch_size=32, verbose=0)

            # Make a prediction from the test data.
            predictions = model.predict(Test_Columns)

            # Choose bin based on highest percentage probability.
            max_indices = np.argmax(predictions, axis=1)

            # Compute metrics by comparing actual (TestTarget_Columns) vs. predicted (max_indices).
            Accuracy, Precision, Recall, Fscore = Scoring(TestTarget_Columns, max_indices)

            # These lists will hold the four tracked metrics for classifications over the bin array using subsets of questions.
            Accuracy_Subset_AllQuestions, Precision_Subset_AllQuestions, Recall_Subset_AllQuestions,\
                Fscore_Subset_AllQuestions= [],[],[],[];

            # Create a list from the training and testing splits.
            Train_Test_List = [X_Questionnaire_train, X_Questionnaire_test, y_Questionnaire_train, y_Questionnaire_test]

            # Holds the list of questions to be tested.
            Subset_to_Test = []

            # Build abbreviated questionnaires, question by question:
            for Question in Subset_TrainAndTest:
                # Iteratively build a larger list Question by question.
                Subset_to_Test.append(Question)

                # Update the input dimension keyword in the dictionary based on how large the subset of questions is.
                Input = {'input_dim': len(Subset_to_Test)}
                ParameterDictionary_Subset.update(Input)

                # Choose bin based on highest percentage probability.
                max_indices_Subset = Subset_Analysis(Train_Test_List, ParameterDictionary_Subset, TheBinsizeList,
                                                     Subset_to_Test)

                # Note that we are testing the Subset predictions (max_indices_Subset) against the full measure
                # predictions (TestTarget_Columns). We don't want to test the Target Columns of the subset against the
                # Subset predictions as we want to compare the subset predictions against the full measure target values.

                # Compute the metrics of each of the subsets (abbreviated questionnaires), iteratively (question by question)
                # and append those values to an array holding the values for each of the metrics. For example, for a 3
                # question abbreviated questionnaire, there will be one, two, and finally three values in each of the
                # lists being appended.

                # Score the subsets (abbreviated questionnaires) against total number of questions.
                Accuracy_Subset, Precision_Subset, Recall_Subset, Fscore_Subset = \
                    Scoring(TestTarget_Columns, max_indices_Subset)

                # Create lists to hold abbreviated questionnaire (Subset) metrics for each of the trials.
                Accuracy_Subset_AllQuestions.append(Accuracy_Subset)
                Precision_Subset_AllQuestions.append(Precision_Subset)
                Recall_Subset_AllQuestions.append(Recall_Subset)
                Fscore_Subset_AllQuestions.append(Fscore_Subset)

            # Append the metrics for the full questionnaires for each trial to an list.
            AccuracyList.append(Accuracy)
            PrecisionList.append(Precision)
            RecallList.append(Recall)
            FscoreList.append(Fscore)

            # Append the lists for each metric (for each brief) for each trial to a list.
            Accuracy_Subset_AllQuestions_List.append(Accuracy_Subset_AllQuestions)
            Precision_Subset_AllQuestions_List.append(Precision_Subset_AllQuestions)
            Recall_Subset_AllQuestions_List.append(Recall_Subset_AllQuestions)
            Fscore_Subset_AllQuestions_List.append(Fscore_Subset_AllQuestions)

            # release the memory from building the model.
            K.clear_session()

        # Average the metrics for each of the full questionnaires over all of the trials.
        AccuracyAve = mean(AccuracyList)
        PrecisionAve = mean(PrecisionList)
        RecallAve = mean(RecallList)
        FscoreAve = mean(FscoreList)

        # Average the lists of metrics for each of the brief questionnaires over all of the trials.
        AccuracyAve_Subset = RoundandPercent(AverageList(Accuracy_Subset_AllQuestions_List))
        StdevArr_Subset_Accuracy = RoundandPercent(StdDevList(Accuracy_Subset_AllQuestions_List))
        PrecisionAve_Subset = RoundandPercent(AverageList(Precision_Subset_AllQuestions_List))
        StdevArr_Subset_Precision = RoundandPercent(StdDevList(Precision_Subset_AllQuestions_List))
        RecallAve_Subset = RoundandPercent(AverageList(Recall_Subset_AllQuestions_List))
        StdevArr_Subset_Recall = RoundandPercent(StdDevList(Recall_Subset_AllQuestions_List))
        FscoreAve_Subset = RoundandPercent(AverageList(Fscore_Subset_AllQuestions_List))
        StdevArr_Subset_Fscore = RoundandPercent(StdDevList(Fscore_Subset_AllQuestions_List))

        # Retrieve hyperparameter values.
        InputDimension = ParameterDictionary['input_dim']
        OutputDimension = ParameterDictionary['output']
        ActivationFunction = ParameterDictionary['activation']
        FirstLayerNeurons = ParameterDictionary['L1Neurons']
        SecondLayerNeurons = ParameterDictionary['L2Neurons']
        Loss_Function=  ParameterDictionary['loss']
        TheOptimizer = ParameterDictionary['optimizer']
        Fitting_Metric = ParameterDictionary['metrics']

        # Create a string of hyperparameter values and their descriptions.
        ModelInfo = 'Activation: ' + str(ActivationFunction) + ', Layer 1: ' + str(FirstLayerNeurons) + ', LossFunction: ' + \
                    Loss_Function + ', Optimizer: ' + TheOptimizer + ', FittingMetric: ' + Fitting_Metric + ', Questions: ' + \
                    str(InputDimension) + ', OutputBins: ' + str(OutputDimension)

        # For each average of trials, Append the scoring metrics and model info for the full questionnaire.
        ScoringMetrics.append(
            [ModelInfo, Bin_Iterator, Subjects, len(TheBinsizeList) + 1, AccuracyAve, PrecisionAve, RecallAve, FscoreAve])

        # Create a string for subset info.
        Subset_Info = 'Questions: ' + \
                      str(len(Subset_TrainAndTest)) + ', OutputBins: ' + str(len(TheBinsizeList) + 1)

        # For each average of trials, append the scoring metrics and model info for the full questionnaire.
        ScoringMetrics_Subset.append(
            [Subset_Info, Bin_Iterator, Subjects, len(TheBinsizeList) + 1, tuple(AccuracyAve_Subset),
             tuple(PrecisionAve_Subset), tuple(RecallAve_Subset),tuple(FscoreAve_Subset)])

        # Append all of the standard deviations for the various metrics to a list.
        StDevMetrics.append([tuple(StdevArr_Subset_Accuracy), tuple(StdevArr_Subset_Precision),
             tuple(StdevArr_Subset_Recall),tuple(StdevArr_Subset_Fscore)])

        # Create dataframes for both the full and abbreviated questionnaire metrics and the metric standard deviations.
        SubjectsandQuestions_DF = pd.DataFrame.from_records(ScoringMetrics,columns=FullQuestionnaireMetrics)
        SubjectsandQuestions_DF_Subset = pd.DataFrame.from_records(ScoringMetrics_Subset,columns=BriefQuestionnaireMetrics)
        StDevMetrics_DF = pd.DataFrame.from_records(StDevMetrics, columns=FullQuestionnaire_StDevMetrics)

    # Append the full and abbreviated questionnaires for each new bin array.
    ScoringMetricsConcat_DF = pd.concat([ScoringMetricsConcat_DF, SubjectsandQuestions_DF], axis=0)
    ScoringMetricsConcat_DF_Subset = pd.concat([ScoringMetricsConcat_DF_Subset, SubjectsandQuestions_DF_Subset], axis=0)
    ScoringMetricsStDevConcat_DF = pd.concat([ScoringMetricsStDevConcat_DF, StDevMetrics_DF], axis=0)

print(ScoringMetricsConcat_DF_Subset)
print(ScoringMetricsStDevConcat_DF)

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
