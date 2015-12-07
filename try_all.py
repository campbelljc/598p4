from __future__ import division
import common
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn import metrics

# ref : http://scikit-learn.org/stable/auto_examples/plot_classifier_comparison.html

names = ["SGD", "Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "LDA", "QDA"]
classifiers = [
    SGDClassifier(loss='hinge', penalty='l2', alpha=0.005, n_iter=10, random_state=42, n_jobs=-1, average=True),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LDA(),
    QDA()
]

X_train, X_test, y_train, y_test = common.load_train_data_and_split()

# iterate over classifiers
for name, clf in zip(names, classifiers):
    print("Fitting " + name + "...")
        
    predicted_test = clf.fit(X_train, y_train).predict(X_test)
    test_p = ((y_test != predicted_test).sum())/(len(X_test))*100
    print("Error on test set: %d" % test_p)
    
    print(metrics.classification_report(y_test, predicted_test))
    
'''

results (f1-score)
-> rows with missing medical speciality removed

sgd         0.46
knn         0.49
lin-svm     0.49 (warning: precision/f-score are being set to 0.0 in labels with no predicted samples)
rbf-svm     0.46
dt          0.52
rand-forest 0.46
adaboost    0.54 ***
n.bayes     0.03
lda         0.51 (warning: variables are colinear)
qda         0.02 (warning: variables are colinear)

results (f1-score)
-> rows with missing medical speciality removed, diagnoses grouped together by icd9 code category

sgd         0.49 (warning: precision/f-score are being set to 0.0 in labels with no predicted samples)
knn         0.51
lin-svm     0.49 
rbf-svm     0.46
dt          0.52
rand-forest 0.46
adaboost    0.54 ***
n.bayes     0.03
lda         0.51 (warning: variables are colinear)
qda         0.02 (warning: variables are colinear)

'''