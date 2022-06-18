# Ready to go after a few more comments

# This program finds SHAP values associated with each particular bin/level, does this over multiple trials, and then
# averages the trials together to find the averaged SHAP values for each question associated to each bin.

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
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import matplotlib.pyplot as plt


# Import local packages.
from SupportFunctions import ChoiceImputation, BinConfigurations, Train_Test_sets, Scoring
from NeuralNetworkMethods import KerasModel, SHAPInfo, NNParameters
from Questionnaires import PSWQ_Dutch_Positve

PSWQ = PSWQ_Dutch_Positve()
# If other questionnaires are added, cleaned, filtered etc. they can be concatenated here.
Questionnaire = pd.concat([PSWQ], axis = 1)
# Capture the independent features in the questionnaire.
Features = Questionnaire
# Choose whether to impute or drop subjects with missing item scores (1 = impute, 0 = drop columns with missing data)
Imputationchoice = 1
Questionnaire = ChoiceImputation(Questionnaire, Imputationchoice)
# Sum item scores.
Questionnaire["Sum_of_Scores"]= Questionnaire.iloc[:, :].sum(axis=1)
# Choose the number of subjects ot use in the classification model.
Number_of_Subjects = [len(PSWQ)]
# Create a dataframe from the independent features.
df = pd.DataFrame(columns=Features.columns)
# Create a column for the trial number.
df.insert(0, 'Trial',0)

bins = BinConfigurations()

BinName, TheBinsizeList = bins[0]
InputDim = len(Questionnaire.iloc[:, 0:-1].columns)
OutputDim = len(TheBinsizeList) + 1

# Creates a dictionary of hyperparameters and input and output sizes.
ParameterDictionary = NNParameters(InputDim, OutputDim)
model_Keras = KerasModel(ParameterDictionary)

# The number of subjects to be used in the model.
Subjects = len(Questionnaire)
# Randomized dataframe of subjects/questions.
Subjects_Subset = Questionnaire.sample(Subjects)
# Select the independent features (Questions).
X_Questions = Subjects_Subset.iloc[:, 0:-1]
# Select the dependent feature (Sum of Scores = SoS) which will be binned.
y_SoS = Subjects_Subset[["Sum_of_Scores"]]
# Number of Trials to perform.
NumTrials = 2
# Do the computations for the number trials (which the metrics will be averaged over).
for TrialNumber in range(NumTrials):
    print(TrialNumber)
    # Training and testing split.
    Train_Columns_Arr, TrainTarget_Columns_Arr, Test_Columns_Arr, TestTarget_Columns_Arr = \
        Train_Test_sets(X_Questions, y_SoS,TheBinsizeList)

    # Fit the neural network to the training data.
    model_Keras.fit(Train_Columns_Arr, TrainTarget_Columns_Arr, epochs=30, batch_size=32, verbose=0)
    # Make a prediction from the test data.
    predictions = model_Keras.predict(Test_Columns_Arr)
    # Choose bin based on highest percentage probability.
    predictions = np.argmax(predictions, axis=1)

    # Compute metrics by comparing actual (TestTarget) vs. predicted (predictions).
    Accuracy, Precision, Recall, Fscore = Scoring(TestTarget_Columns_Arr, predictions)

    # Calculate SHAP values.
    shapoutput = SHAPInfo(model_Keras, Train_Columns_Arr, Test_Columns_Arr)
    output = []

    shapoutputsize = len(shapoutput)
    for i in range(shapoutputsize):
        # First, averaging is being done over each of the SHAP values for each question in the *shapoutput[i].
        # For example, if there are 500 subjects in the training set the first elem will be all of their SHAP values
        # for question 0, the second elem will be the 500 SHAP values for question 1 etc. This is being done for each
        # of the outputs in shapouputsize which means that for 4 bins there will be four SHAP levels for shapoutput[i]
        # where i = 0 to 3. Now in general there are the k bins/levels for each fo the n trials. These k * n are all
        # inserted in the dataframe in an order such that the levels associated for a particular trial are subsequently
        # displayed.
        df.loc[(len(TheBinsizeList) + 1)*TrialNumber + i] = ['Trial_' + str(TrialNumber + 1) + " " + str(i)] \
                                    + [sum(np.abs(elem)) / len(elem) for elem in zip(*shapoutput[i])]


# Combined_SHAP_df holds each of averaged SHAP values.
Combined_SHAP_df = pd.DataFrame()
for i in range(shapoutputsize):
    # shapoutputsize = the number of bins = number of bin configurations.
    # i::shapoutputsize 'slices' a range, it is a type of modulo operator. So when i = 0, letting shapoutputsize = 4
    # and n >= 0 we get n * shapoutputsize + i. For example, with shapoutputsize = 4 and i = 0, starting with n = 0
    # which increments by 1 we get the records: (0 * 4 + 0, 1 * 4 + 0, 2 * 4 + 0, ...,n*4+0) = (0,1*4+0,...,n*4+0).
    # If i = 2 then we get (2,6,...,n*4 + 2).
    SHAPLevel = df.iloc[i::shapoutputsize, :]

    # Each of the records associated to each bin/level e.g. (0,4,...) or (2,6,...) can be averaged together and put
    # in a dataframe.
    SHAP_Level_df = SHAPLevel.mean(axis=0).to_frame().T
    # One by one, the averaged item level SHAP values associated to each bin array are put concatentated into a dataframe.
    Combined_SHAP_df = pd.concat([Combined_SHAP_df, SHAP_Level_df])

# Reset the index.
Combined_SHAP_df = Combined_SHAP_df.reset_index()
Combined_SHAP_df.index = Combined_SHAP_df.index + 1
# Drop the extra column that was created when resetting the index.
Combined_SHAP_df = Combined_SHAP_df.drop('index', 1)

SHAP_Column_Names_ls = []
Combined_SHAP_df.index = np.arange(1, len(Combined_SHAP_df) + 1)
for i in range(len(Combined_SHAP_df)):
    # Here we sort the items by the overall SHAP value and append them to a list.
    SHAP_Column_Names = Combined_SHAP_df.iloc[[i]].apply(lambda x: x.sort_values(ascending=False), axis=1)
    SHAP_Column_Names_ls.append(SHAP_Column_Names.columns.values.tolist())


# Here, we are averaging all the SHAP values for all question configurations together.
# That is, we average each of the SHAP values by question for all levels.
Summed = Combined_SHAP_df.sum().to_frame().T
print('Summed values for ' +str(shapoutputsize) +" Bins \n" + str(Summed))
print("Ordered by SHAP importance: \n" + str(Summed.apply(lambda x: x.sort_values(ascending=False), axis=1)))

# Print the SHAP value ordered questions.
for i in Combined_SHAP_df.index:
    value = Combined_SHAP_df.loc[i, :].to_frame().T
    print('For PSWQ Bin level ' + str(i) + "\n" + str(value.apply(lambda x: x.sort_values(ascending=False), axis=1)))

# Display the SHAP values by question.
Combined_SHAP_df_Transposed = Combined_SHAP_df.transpose()
plt.rcParams["figure.figsize"] = [4,4]
theplot = Combined_SHAP_df_Transposed.iloc[:,:].plot(kind="bar", stacked=True, width=.6)
plt.title('Mean(|SHAP|) values per question')
plt.xlabel('Question')
plt.ylabel('mean(|SHAP|) value')
plt.legend(title='Bins', bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()

# Display the SHAP values by bins.
plt.rcParams["figure.figsize"] = [6,4]
Combined_SHAP_df.iloc[:,:].plot(kind="bar")
plt.title('Question importance for each bin')
plt.xlabel('Bins')
plt.ylabel('mean(|SHAP|) value')
plt.legend(title='Questions', bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()

# Display the Averaged Combined SHAP values.
plt.rcParams["figure.figsize"] = [4,4]
Averaged_Combined_SHAP_df = Combined_SHAP_df.iloc[:,:].mean().to_frame().transpose()
Averaged_Combined_SHAP_df.plot(kind="bar")
plt.xlabel('Averaged Configuration')
plt.ylabel('Averaged mean(|SHAP|) value')
plt.legend(title='Questions', bbox_to_anchor=(1.0, 1.0))
plt.xticks([])
plt.tight_layout()

plt.show()