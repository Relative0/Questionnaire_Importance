# Good to go.

# Import libraries.
import numpy as np
from sklearn import linear_model, preprocessing
from statistics import pstdev
from sklearn.metrics import accuracy_score, f1_score,precision_score,recall_score

from DimensionalityBinning import DimensionalBinChoice
from sklearn.model_selection import train_test_split

# Average an input list.
def AverageList(InList):
    output = [sum(elem) / len(elem) for elem in zip(*InList)]
    # print(output)
    return output

def StdDevList(InList):
    output = [pstdev(elem) for elem in zip(*InList)]
    return output

def ListAverage(InList):
    output = sum(InList) / len(InList)
    # print(output)
    return output

# Standardize the dataframe.
def Standardize(InDataframe):
    # Get column names first
    Dataframe = InDataframe.copy()
    col_names = Dataframe.columns
    features = Dataframe[col_names]
    Standardized = preprocessing.StandardScaler()
    scaler = Standardized.fit(features.values)
    features = scaler.transform(features.values)
    Dataframe[col_names] = features
    return Dataframe

# Choose whether or not to imput missing values from a dataframe.
def ChoiceImputation(df,choice):
    # We create a limit between 1 and 5 inclusive to find and remove outliers.
    limit = [1, 2, 3, 4, 5]
    df_Cleaned = df[df.isin(limit)]
    if choice == 0:
        #Don't impute
        df_Cleaned = df_Cleaned.dropna(axis = 0, how ='any')

    elif choice == 1:
        #Impute
        df_Cleaned.fillna(df_Cleaned.mean(), inplace=True)
    else:
        print("Issue in ChoiceImputation function")
    return df_Cleaned

# Round an individual value.
def Round(value):
    value = np.round(value, 2)
    return value

# Round each value in a list.
def RoundandPercent(InList):
    rounded = ["{:.3f}".format(round(num, 3)) for num in InList]
    return rounded

# Plot the GridSearch.
def GridSearch_table_plot(grid_clf, param_name, num_results=15,negative=True,graph=True,display_all_params=True):
    '''Display grid search results

    Arguments
    ---------

    grid_clf           the estimator resulting from a grid search
                       for example: grid_clf = GridSearchCV( ...

    param_name         a string with the name of the parameter being tested

    num_results        an integer indicating the number of results to display
                       Default: 15

    negative           boolean: should the sign of the score be reversed?
                       scoring = 'neg_log_loss', for instance
                       Default: True

    graph              boolean: should a graph be produced?
                       non-numeric parameters (True/False, None) don't graph well
                       Default: True

    display_all_params boolean: should we print out all of the parameters, not just the ones searched for?
                       Default: True

    Usage
    -----

    GridSearch_table_plot(grid_clf, "min_samples_leaf")

                          '''
    from matplotlib      import pyplot as plt
    from IPython.display import display
    import pandas as pd

    clf = grid_clf.best_estimator_
    clf_params = grid_clf.best_params_
    if negative:
        clf_score = -grid_clf.best_score_
    else:
        clf_score = grid_clf.best_score_
    clf_stdev = grid_clf.cv_results_['std_test_score'][grid_clf.best_index_]
    cv_results = grid_clf.cv_results_

    print("best parameters: {}".format(clf_params))
    print("best score:      {:0.5f} (+/-{:0.5f})".format(clf_score, clf_stdev))
    if display_all_params:
        import pprint
        pprint.pprint(clf.get_params())

    # pick out the best results
    # =========================
    scores_df = pd.DataFrame(cv_results).sort_values(by='rank_test_score')

    best_row = scores_df.iloc[0, :]
    if negative:
        best_mean = -best_row['mean_test_score']
    else:
        best_mean = best_row['mean_test_score']
    best_stdev = best_row['std_test_score']
    best_param = best_row['param_' + param_name]

    # display the top 'num_results' results
    # =====================================
    display(pd.DataFrame(cv_results) \
            .sort_values(by='rank_test_score').head(num_results))

    # plot the results
    # ================
    scores_df = scores_df.sort_values(by='param_' + param_name)

    if negative:
        means = -scores_df['mean_test_score']
    else:
        means = scores_df['mean_test_score']
    stds = scores_df['std_test_score']
    params = scores_df['param_' + param_name]

    # plot
    if graph:
        plt.figure(figsize=(8, 8))
        plt.errorbar(params, means, yerr=stds)

        plt.axhline(y=best_mean + best_stdev, color='red')
        plt.axhline(y=best_mean - best_stdev, color='red')
        plt.plot(best_param, best_mean, 'or')

        plt.title(param_name + " vs Score\nBest Score {:0.5f}".format(clf_score))
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.show()

# Compute micro-averaging for various metrics.
def ComputeConfusionVals(confusion_matrix):
    # print(confusion_matrix)
    TN, FP, FN, TP = confusion_matrix.ravel()
    TPR = TP / (TP + FN)

    # print("TPR = Sensitivity/Recall: " + str(TPR_Average))
    # Specificity or true negative rate
    TNR = TN / (TN + FP)

    # print("TNR = Specificity: " + str(TNR_Average))
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # PPV_Average = ListAverage(PPV.tolist())
    # print("PPV = Precision: " + str(PPV_Average))
    # Negative predictive value
    NPV = TN / (TN + FN)
    # NPV_Average = ListAverage(NPV.tolist())
    # print("NPV: " + str(NPV_Average))
    # Fall out or false positive rate
    FPR = FP / (FP + TN)

    # print("FPR = Fall Out: " + str(FPR_Average))
    # False negative rate
    FNR = FN / (TP + FN)
    # FNR_Average = ListAverage(FNR.tolist())
    # print("FNR: " + str(FNR_Average))
    # False discovery rate
    FDR = FP / (TP + FP)
    # FDR_Average = ListAverage(FDR.tolist())
    # print("FDR: " + str(FDR_Average))
    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    # ACC_Average = ListAverage(ACC.tolist())
    # print("Accuracy: " + str(ACC_Average))
    # F1 Score (Harmonic mean between precision and sensitivity)
    F1 = 2*TP / (2*TP + FP + FN)
    # F1_Average = ListAverage(F1.tolist())
    # print("F1 Score: " + str(F1_Average))

    return TPR, TNR, PPV, NPV, FPR, FNR, FDR, ACC, F1

# Choose the bin-configurations that should be used to create a model. Uncomment only those for which you would like to
# create a model.
def BinConfigurations():
    # Defines the number and AUC of each bin array.
    TwoBin = [0]
    ThreeBin = [-.431, .431]
    FourBin = [-.674, 0, .674]
    FiveBin = [-.842, -0.253, 0.253, .842]
    SixBin = [-0.967, -0.431, 0, 0.431, 0.967]
    SevenBin = [-1.068, -0.566, -0.18, 0.18, 0.566, 1.068]
    EightBin = [-1.15, -.674, -.319, 0, .319, .674, 1.15]

    # Create a list for all bin arrays that will be used to build classification models.
    bins = []

    # We choose the bin-configuration to use. Note: If Question orderings are used from "SHAP_QuestionOrderings.txt"
    # then comment out those configurations that are not in the file, otherwise an error will be thrown. If they are
    # not linked with question orderings from a file, then they will just be used to bin data into n-bins.
    # These bin-configurations are ultimately what are going to be used to be trained/tested on. Comment out those
    # configurations that will not be used to build a model.
    # bins.append(('2-Bin',TwoBin))
    # bins.append(('3-Bin',ThreeBin))
    bins.append(('4-Bin',FourBin))
    # bins.append(('5-Bin',FiveBin))
    # bins.append(('6-Bin',SixBin) )
    # bins.append(('7-Bin',SevenBin))
    bins.append(('8-Bin',EightBin))

    return bins

# Bin-configuration particular to the questionnaire ordering in the file "SHAP_QuestionOrderings.txt" e.g.
# 'Four-Bin-Forward' should match the line in the SHAP_QuestionOrderings.txt file.
def BinConfiguration_Quartile_ForwardReverse():
    # Defines the number and AUC of each bin array.
    FourBin = [-.674, 0, .674]

    # Create a list for all bin arrays that will be used to build classification models.
    bins = []

    # We choose the bin-configuration to use. Note: If Question orderings are used from "SHAP_QuestionOrderings.txt"
    # then comment out those configurations that are not in the file, otherwise an error will be thrown. If they are
    # not linked with question orderings from a file, then they will just be used to bin data into n-bins.
    # These bin-configurations are ultimately what are going to be used to be trained/tested on. Comment out those
    #     # configurations that will not be used to build a model.
    bins.append(('Four-Bin-Forward', FourBin))
    bins.append(('Four-Bin-Reversed', FourBin))

    return bins

# Bin-configuration particular to the questionnaire ordering in the file "SHAP_QuestionOrderings.txt" e.g.
# 'Topper_from_Paper' should match the line in the SHAP_QuestionOrderings.txt file.
def BinConfigurations_Toppers():
    # Defines the number and AUC of each bin array.
    # For Topper, comparing to his choice of optimal cutoff of a sum of scores of 15 for 5 questions (15/25 = .6)
    # we have a z-score of .253.
    # Toppers = [.253]
    # In Toppers paper he says 75% for the cutoff
    Topper_from_Paper = [.674]
    Topper_our_Comparison = [.674]

    # Create a list for all bin arrays that will be used to build classification models.
    bins = []

    # We choose the bin-configuration to use. Note: If Question orderings are used from "SHAP_QuestionOrderings.txt"
    # then comment out those configurations that are not in the file, otherwise an error will be thrown. If they are
    # not linked with question orderings from a file, then they will just be used to bin data into n-bins.
    # These bin-configurations are ultimately what are going to be used to be trained/tested on. Comment out those
    #     # configurations that will not be used to build a model.
    bins.append(('Topper_from_Paper', Topper_from_Paper))
    bins.append(('Topper_our_Comparison', Topper_our_Comparison))

    return bins

# Bin-configuration particular to the questionnaire ordering in the file "SHAP_QuestionOrderings.txt" e.g.
# 'High' should match the line in the SHAP_QuestionOrderings.txt file.
def BinConfiguration_HighLowAve_TopQuartile():
    # Defines the number and AUC of each bin array.
    TwoBin_fromFour_topQuartile = [.674]

    # Create a list for all bin arrays that will be used to build classification models.
    bins = []

    # We choose the bin-configuration to use. Note: If Question orderings are used from "SHAP_QuestionOrderings.txt"
    # then comment out those configurations that are not in the file, otherwise an error will be thrown. If they are
    # not linked with question orderings from a file, then they will just be used to bin data into n-bins.
    # These bin-configurations are ultimately what are going to be used to be trained/tested on. Comment out those
    #     # configurations that will not be used to build a model.
    bins.append(('High',TwoBin_fromFour_topQuartile))
    bins.append(('Ave',TwoBin_fromFour_topQuartile))
    bins.append(('Low',TwoBin_fromFour_topQuartile))

    return bins

# Bin-configuration particular to the questionnaire ordering in the file "SHAP_QuestionOrderings.txt" e.g.
# 'High' should match the line in the SHAP_QuestionOrderings.txt file.
def BinConfiguration_HighLowAve_TopOctile():
    # Defines the number and AUC of each bin array.
    TwoBin_fromEight_topOctile = [1.15]

    # Create a list for all bin arrays that will be used to build classification models.
    bins = []

    # We choose the bin-configuration to use. Note: If Question orderings are used from "SHAP_QuestionOrderings.txt"
    # then comment out those configurations that are not in the file, otherwise an error will be thrown. If they are
    # not linked with question orderings from a file, then they will just be used to bin data into n-bins.
    # These bin-configurations are ultimately what are going to be used to be trained/tested on. Comment out those
    #     # configurations that will not be used to build a model.
    bins.append(('High',TwoBin_fromEight_topOctile))
    bins.append(('Ave',TwoBin_fromEight_topOctile))
    bins.append(('Low',TwoBin_fromEight_topOctile))

    return bins

# Bin-configuration particular to the questionnaire ordering in the file "SHAP_QuestionOrderings.txt" e.g.
# 'Top5%_11-item' should match the line in the SHAP_QuestionOrderings.txt file.
def BinConfiguration_HighLowAve_Top5percent():
    # Defines the number and AUC of each bin array.
    # 1.6449 Standard Deviations for the top 5%
    TwoBinMod = [1.645]

    # Create a list for all bin arrays that will be used to build classification models.
    bins = []

    # We choose the bin-configuration to use. Note: If Question orderings are used from "SHAP_QuestionOrderings.txt"
    # then comment out those configurations that are not in the file, otherwise an error will be thrown. If they are
    # not linked with question orderings from a file, then they will just be used to bin data into n-bins.
    # These bin-configurations are ultimately what are going to be used to be trained/tested on. Comment out those
    #     # configurations that will not be used to build a model.
    bins.append(('Top5%_11-item', TwoBinMod))

# Here we do the basic train/test split, separate standardization of the testing and training data and binning. Training
# and testing was done separately as Binning depends on the standard deviations thus it is important to not compute
# s.d. from mixed training/testing data.
def Train_Test_sets(X_Questions,y_SoS, TheBinsizeList):
    # Training and testing split.
    X_Questions_train, X_Questions_test, y_SoS_train, y_SoS_test = train_test_split(X_Questions, y_SoS,
                                                                                    test_size=.2)
    # Standardizing the training set (apart from the test set).
    X_Questions_train = Standardize(X_Questions_train)

    # Bin the sum of scores of the training set.
    y_SoS_train['Bin_Level'] = DimensionalBinChoice(y_SoS_train['Sum_of_Scores'], TheBinsizeList)

    # Create an array of the training data.
    Train_Columns_Arr = np.asarray(X_Questions_train.iloc[:, :len(X_Questions_train.columns)].values)
    TrainTarget_Columns_Arr = np.asarray(y_SoS_train['Bin_Level'].values - 1)

    # Standardizing the testing set (apart from the training set).
    X_Questions_test = Standardize(X_Questions_test)

    # Bin the sum of scores of the testing set.
    y_SoS_test['Bin_Level'] = DimensionalBinChoice(y_SoS_test['Sum_of_Scores'], TheBinsizeList)

    # Create an array of the testing data.
    Test_Columns_Arr = np.asarray(X_Questions_test.iloc[:, :len(X_Questions_test.columns)].values)
    TestTarget_Columns_Arr = np.asarray(y_SoS_test['Bin_Level'].values - 1)

    return Train_Columns_Arr, TrainTarget_Columns_Arr, Test_Columns_Arr, TestTarget_Columns_Arr

# This is a weighted averaging method for computing several different metrics.
def Scoring(TestTarget_Columns_Arr, predictions):
    Accuracy = accuracy_score(TestTarget_Columns_Arr, predictions)
    Precision = precision_score(TestTarget_Columns_Arr, predictions, average="weighted")
    Recall = recall_score(TestTarget_Columns_Arr, predictions, average="weighted")
    Fscore = f1_score(TestTarget_Columns_Arr, predictions, average="weighted")

    return Accuracy, Precision, Recall, Fscore