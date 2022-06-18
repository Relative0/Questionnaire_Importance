# Good to go.
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


# Uncomment all algorithms to be used for testing.
def AlgorithmModels():
    models = []
    models.append(("NN",  "Defined Below"))
    models.append(("SVM",  svm.SVC(kernel="linear",C=1000, gamma=.001)))
    models.append(("LR", LogisticRegression(C=100, penalty='l2')))
    # models.append(("RBFSVM",  svm.SVC(kernel="rbf")))
    # models.append(("PolySVM",  svm.SVC(kernel="poly")))
    # models.append(("RBFSVM",  svm.SVC(kernel="rbf")))
    # models.append(("SigmoidVM",  svm.SVC(kernel="sigmoid")))
    # # models.append(("L-Regression", LinearRegression()))
    # models.append(("Naive Bayes:",GaussianNB()))
    # # models.append(("K-Nearest Neighbour:",KNeighborsClassifier(n_neighbors=8)))
    # models.append(("Decision Tree:",DecisionTreeClassifier()))
    # # # models.append(("Support Vector Machine-linear:",SVC(kernel="linear")))
    # models.append(("Support Vector Machine-rbf:",SVC(kernel="rbf")))
    # models.append(("Random Forest:",RandomForestClassifier(n_estimators=8)))
    # # # # models.append(("eXtreme Gradient Boost:",XGBClassifier()))
    # HiddenLayerSize = 2
    # models.append(("MLP " + str(HiddenLayerSize) + "Neuron:",MLPClassifier(hidden_layer_sizes=HiddenLayerSize ,solver='adam', activation='tanh', \
    #                                     learning_rate='adaptive',alpha=.05, max_iter=1000)))
    # models.append(("skNN",MLPClassifier(hidden_layer_sizes=HiddenLayerSize, \
    #                                      solver='lbfgs', activation='logistic', \
    #                                      learning_rate='adaptive',alpha=.05)))
    # models.append(("AdaBoostClassifier:",AdaBoostClassifier()))
    # models.append(("GradientBoostingClassifier:",GradientBoostingClassifier()))

    return models
