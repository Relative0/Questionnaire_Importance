# Good to go
# Import external packages.
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.options.display.max_columns = None
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth', 150)
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import csv
from sklearn.model_selection import train_test_split
from statistics import pstdev
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix

# Import local packages.
from DimensionalityBinning import DimensionalBinChoice
from SupportFunctions import ChoiceImputation, Standardize, ComputeConfusionVals, BinConfigurations_Toppers
from NeuralNetworkMethods import KerasModel, NNParameters

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

# Filter those columns that match "PSWQ_". Note, all 16 PSWQ questions are used.
PSWQ = data.filter(like='PSWQ_', axis=1)

# If other questionnaires are added, cleaned, filtered etc. they can be concatenated here.
Questionnaire = pd.concat([PSWQ], axis=1)

# Choose whether to impute or drop subjects with missing item scores.
Imputationchoice = 1
Questionnaire = ChoiceImputation(Questionnaire, Imputationchoice)

# Sum item scores.
Questionnaire["Sum_of_Scores"] = Questionnaire.iloc[:, :].sum(axis=1)

# Choose the number of subjects to use in classification model.
Number_of_Subjects = [len(data)]

bins = BinConfigurations_Toppers()
# Define hyperparameters for the neural network type which will be used to create models for all classifications.

# Create dummy dimensions for the NN. This is done to update the number of input features (= questions = DummyInputDim)
# and the number of output bins (=size of the bin array = DummyOutputDim) within the inner loops without having to
# rebuild the NN.
DummyInputDim , DummyOutputDim = 0,0 ;
# Creates a dictionary of hyperparameters and input and output sizes.
ParameterDictionary_Subset = NNParameters(DummyInputDim, DummyOutputDim)

# Create lists for the metrics we want to keep track of.

# Create df's for holding the output data for both the whole questoinnaire as well as each of the brief questionnaires.
ScoringMetricsConfMatConcat_DF = pd.DataFrame(columns=['Subjects', 'Binsize', 'ModelName', 'Acc', 'Sense', 'Spec',
                                                       'Prec', 'NPV', 'FDR', 'FNR'])
ScoringMetricsStDevConcat_DF = pd.DataFrame(columns=['Trials', 'Binsize', 'ModelName', "TPR_StdDev", "TNR_StdDev", "PPV_StdDev",
                                    "NPV_StdDev", "FPR_StdDev", "FNR_StdDev", "FDR_StdDev", "ACC_StdDev", "F1_StdDev"])


# Number of Trials to perform (which the metrics are then averaged over).
NumTrials = 2

BinNames = []
Bin_Iterator = 0
# Do all computations for each bin array selected above.
for BinsizeName, TheBinsizeList in bins:
    Bin_Iterator = Bin_Iterator + 1
    BinNames.append(BinsizeName)
    print("For the BinSize " + BinsizeName)

    # Extract the row of questions (from file) associated to each bin array.
    RowofQuestions = SHAP_Orderings_df[SHAP_Orderings_df['BinSize'].str.contains(BinsizeName)]

    # Create a list and pull off the outer [ ] brackets.
    [Subset_TrainAndTest] = RowofQuestions.iloc[:, 1:].values.tolist()

    # Remove None values in list
    Subset_TrainAndTest = list(filter(None, Subset_TrainAndTest))

    # Update the output size of the neural network to be that of the size of the current bin array.
    Output = {'output': len(TheBinsizeList) + 1}
    ParameterDictionary_Subset.update(Output)

    for SubjectSubsetNumber in range(len(Number_of_Subjects)):
        # Holds the classification metrics for the full and abbreviated metrics at the model level (before averaging).
        ConfMat_ListofArrays = []
        cm_confmat = []
        ScoringMetricsConfMat = []
        ScoringMetrics_StdDevs = []
        TPR_List = []; TNR_List = []; PPV_List = []; NPV_List = []; FPR_List = []; FNR_List = [];
        FDR_List = []; ACC_List = []; F1_List = [];

        print('Subject Subset counter ' + str(SubjectSubsetNumber + 1) + " In the Binsize " + BinsizeName)

        # Randomized dataframe of subjects/questions.
        Subjects = Number_of_Subjects[SubjectSubsetNumber]
        Questionnaire_Subjects_Subset = Questionnaire.sample(Subjects)

        # Do the computations for the number trials (which the metrics will be averaged over).
        for TrialNumber in range(NumTrials):
            print('Trial Number ' + str(TrialNumber + 1) + ' For the Subject Subset counter '
                  + str(SubjectSubsetNumber + 1) + " In the Binsize " + BinsizeName)

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
            outDim = len(TheBinsizeList) + 1
            ParameterDictionary = NNParameters(inDim, outDim)

            # Create the neural network.
            model = KerasModel(ParameterDictionary)

            # Fit the neural network to the training data.
            model.fit(Train_Columns, TrainTarget_Columns, epochs=30, batch_size=32, verbose=0)

            # Make a prediction from the test data.
            predictions = model.predict(Test_Columns)

            # Choose bin based on highest percentage probability.
            max_indices = np.argmax(predictions, axis=1)

            # Create a confusion matrix from data.
            cm = confusion_matrix(TestTarget_Columns, max_indices, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            # Append the confusion matrix to an array where we will eventually average the confusion matrices together.
            ConfMat_ListofArrays.append(cm)

        # Average the confusion matrices together.
        ConfMat_Ave = np.mean(ConfMat_ListofArrays, axis = 0)

        # find each of the true and false values.
        tn, fp, fn, tp = ConfMat_Ave.ravel()

        # For each confusion matrix array append the computed metrics to a list.
        for j in range(len(ConfMat_ListofArrays)):
            # Compute the metrics for each confusion matrix.
            TPR_Individual, TNR_Individual, PPV_Individual, NPV_Individual, FPR_Individual, FNR_Individual, \
            FDR_Individual, ACC_Individual, F1_Individual = ComputeConfusionVals(ConfMat_ListofArrays[j])

            # Append each of the individual metrix to a list. These lists will eventually be averaged over.
            TPR_List.append(TPR_Individual), TNR_List.append(TNR_Individual), PPV_List.append(PPV_Individual), \
            NPV_List.append(NPV_Individual), FPR_List.append(FPR_Individual), FNR_List.append(FNR_Individual), \
            FDR_List.append(FDR_Individual), ACC_List.append(ACC_Individual), F1_List.append(F1_Individual)

        # Find the averaged metrics from the average confusion matrix.
        TPR_Averaged, TNR_Averaged, PPV_Averaged, NPV_Averaged, FPR_Averaged, FNR_Averaged, FDR_Averaged, \
        ACC_Averaged, F1_Averaged = ComputeConfusionVals(ConfMat_Ave)

        # find the metrics standard deviations from the lists of the particular metrics.
        TPR_std = pstdev(TPR_List); TNR_std = pstdev(TNR_List); PPV_std = pstdev(PPV_List); \
        NPV_std = pstdev(NPV_List); FPR_std = pstdev(FPR_List); FNR_std = pstdev(FNR_List); \
        FDR_std = pstdev(FDR_List); ACC_std = pstdev(ACC_List); F1_std = pstdev(F1_List)

        print(ConfMat_Ave)

        # release the memory from building the model.
        K.clear_session()

        # Append a list of averaged metrics and other data for the particular bin configuration and subjects.
        ScoringMetricsConfMat.append(
            [Subjects, len(TheBinsizeList) + 1, BinsizeName, ACC_Averaged, TPR_Averaged, TNR_Averaged, PPV_Averaged,
             NPV_Averaged, FDR_Averaged, FNR_Averaged])

        # Append a list of averaged metrics standard deviations and other data for the particular bin configuration and
        # subjects.
        ScoringMetrics_StdDevs.append([NumTrials, len(TheBinsizeList) + 1, BinsizeName, TPR_std, TNR_std, PPV_std,
                                       NPV_std, FPR_std, FNR_std, FDR_std,ACC_std, F1_std])

        # Create dataframes for both the metrics and standard deviations.
        SubjectsandQuestionsConfMat_DF = pd.DataFrame.from_records(ScoringMetricsConfMat,
            columns=['Subjects', 'Binsize', 'ModelName', 'Acc', 'Sense', 'Spec', 'Prec', 'NPV', 'FDR', 'FNR'])
        ScoringMetricsStDev_DF = pd.DataFrame.from_records(ScoringMetrics_StdDevs,
            columns=['Trials', 'Binsize', 'ModelName', "TPR_StdDev", "TNR_StdDev", "PPV_StdDev","NPV_StdDev", "FPR_StdDev",
                     "FNR_StdDev", "FDR_StdDev", "ACC_StdDev", "F1_StdDev"])

    # concatenate dataframes for full metric and standard deviations for different bin-configuration and subject trials.
    ScoringMetricsConfMatConcat_DF = pd.concat([ScoringMetricsConfMatConcat_DF,SubjectsandQuestionsConfMat_DF], axis=0)
    ScoringMetricsStDevConcat_DF = pd.concat([ScoringMetricsStDevConcat_DF, ScoringMetricsStDev_DF], axis=0)

print(ScoringMetricsConfMatConcat_DF)
print(ScoringMetricsStDevConcat_DF)