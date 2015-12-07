import common
import numpy as np
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn import linear_model, cross_validation, metrics
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA

# col 6 is med. speciality
data_train, data_test, target_train, target_test = common.load_train_data_and_split(targetcol=6)

def random_methods():
    names = ["SGD", "Nearest Neighbors", "linear-SVM","SVC","Decision Tree",
             "Random Forest", "AdaBoost", "Naive Bayes", "LDA", "QDA"]
    classifiers = [
        SGDClassifier(loss='hinge', penalty='l2', alpha=0.0005, n_iter=200, random_state=42, n_jobs=-1, average=True), # 0.3959
        KNeighborsClassifier(10), # 0.3950
        SVC(kernel="linear", C=0.025), # 0.4172
        SVC(gamma=2, C=1), # 0.3005
        DecisionTreeClassifier(max_depth=11), # 0.4998 ***
        RandomForestClassifier(max_depth=21, n_estimators=21, max_features=1), # 0.4469
        AdaBoostClassifier(), # 0.2992
        GaussianNB(), # 0.002
        LDA(), # 0.3749
        QDA()
    ]
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        print("Fitting " + name + "...")
        clf.fit(data_train, target_train)
        print("Predicting...")
        score = clf.score(data_test, target_test)
        print(score)
        #print("Predict" % (metrics.classification_report(target_test,clf.predict(data_test))))
#random_methods()
def Logistic_cross_vaildation():
    logistic = linear_model.LogisticRegression()
    #cross-validation for logistic regression with RBM
    rbm = BernoulliRBM(random_state=0, verbose=True)
    classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])   
    rbm.n_iter=100
    cv = cross_validation.StratifiedKFold(output, 3)
    score_func = metrics.f1_score
    parameters = { "rbm__learning_rate": [0.1, 0.01, 0.001,0.0001],"rbm__n_components":[100,200,300,400,500,600,700,800],"logistic__C":[1,100,1000,5000]}
    grid_search = GridSearchCV(classifier,parameters,score_func=score_func,cv=cv)
    grid_search.fit(input,output)
    print "Best %s: %0.3f" % (score_func.__name__, grid_search.best_score_)
    print "Best parameters set:"
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print "\t%s: %r" % (param_name, best_parameters[param_name])
Logistic_cross_validation()
def SGD_cross_validation():
    SGD = linear_model.SGDClassifier(loss='hinge',penalty='l2',random_state=42,n_jobs=-1,epsilon=0.001)
    # cross-validaiotn for SGD classifier
    rbm = BernoulliRBM(random_state=0, verbose=True)
    classifier = Pipeline(steps=[('rbm', rbm), ('SGD', SGD)])  
    rbm.n_iter=100
    cv = cross_validation.StratifiedKFold(output, 3)
    score_func = metrics.f1_score
    parameters = { "rbm__learning_rate": [0.1, 0.01, 0.001,0.0001],"rbm__n_components":[100,200,300,400,500,600,700,800],"SGD__alpha":[0.1,0.01,0.001,0.0001], "SGD__C":[1,100,1000,10000]}
    grid_search = GridSearchCV(classifier,parameters,score_func=score_func,cv=cv)
    grid_search.fit(input,output)
    print "Best %s: %0.3f" % (score_func.__name__, grid_search.best_score_)
    print "Best parameters set:"
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print "\t%s: %r" % (param_name, best_parameters[param_name])
#SGD_cross_validation()
def SGD():
    SGD = linear_model.SGDClassifier(loss='hinge',penalty='l2',random_state=42,n_jobs=-1,epsilon=0.001)
    rbm = BernoulliRBM(random_state=0, verbose=True)
    classifier = Pipeline(steps=[('rbm', rbm), ('SGD', SGD)])
    # RBM parameters obtained after cross-validation
    rbm.learning_rate = 0.01
    rbm.n_iter = 15
    rbm.n_components = 50
    SGD.alpha=0.0001
    SGD.C=1 
    # Training SGD
    SGD_classifier = linear_model.SGDClassifier(loss='hinge',penalty='l2',random_state=42,n_jobs=-1,alpha=0.0001, epsilon=0.001)
    SGD_classifier.fit(data_train,target_train)
    # Training RBM-SGD Pipeline    
    classifier.fit(data_train,target_train)
    print("printing_results")
    
    print("SGD using RBM features:\n%s\n" % (metrics.classification_report(target_test,classifier.predict(data_test))))
    cm = confusion_matrix(target_test,classifier.predict(data_test))
    plt.matshow(cm)
    plt.title('Confusion Matrix SVM with SDG with RBM Features')
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix1.jpg')
    print("SGD using raw pixel features:\n%s\n" % (metrics.classification_report(target_test,SGD_classifier.predict(data_test))))
    cm1 = confusion_matrix(target_test,SGD_classifier.predict(data_test))
    plt.matshow(cm1)
    plt.title('Confusion Matrix SVM with SDG Raw Features')
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix2.jpg')
#SGD()
def Logistic():
    logistic = linear_model.LogisticRegression()
    rbm = BernoulliRBM(random_state=0, verbose=True)
    classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
    # RBM parameters obtained after cross-validation
    rbm.learning_rate = 0.01
    rbm.n_iter = 121
    rbm.n_components = 700
    logistic.C= 1.0  
    # Training RBM-Logistic Pipeline
    classifier.fit(data_train,target_train)
    # Training Logistic regression
    logistic_classifier = linear_model.LogisticRegression(C=1.0)
    logistic_classifier.fit(data_train,target_train)    
    print("printing_results")
    print("Logistic regression using RBM features:\n%s\n" % (metrics.classification_report(target_test,classifier.predict(data_test))))
    cm3 = confusion_matrix(target_test,classifier.predict(data_test))
    plt.matshow(cm3)
    plt.title('Confusion Matrix Logistic Regression with RBM Features')
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix3.jpg')
    print("Logistic regression using raw pixel features:\n%s\n" % (metrics.classification_report(target_test,logistic_classifier.predict(data_test))))
    cm4 = confusion_matrix(target_test,logistic_classifier.predict(data_test))
    plt.matshow(cm4)
    plt.title('Confusion Matrix Logistic Regression')
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix4.jpg')
#Logistic()