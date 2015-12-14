import numpy as np
import common
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
from unbalanced_dataset import UnderSampler, NearMiss, CondensedNearestNeighbour, OneSidedSelection, NeighbourhoodCleaningRule, TomekLinks, ClusterCentroids, OverSampler, SMOTE, SMOTETomek, SMOTEENN, EasyEnsemble, BalanceCascade
import itertools
#data_train, data_test, target_train, target_test = common.load_test_train_as_two_class()
#data_train1 = np.asarray(data_train)
#target_train1 = np.array(target_train)
def random_methods(data_train1,target_train1):
    rng = np.random.RandomState(96235)
    names = ["SGD", "Nearest Neighbors", "ensembel","Decision Tree","Random Forest", "AdaBoost", "Naive Bayes", "LDA", "QDA"]
    classifiers = [
        SGDClassifier(loss='hinge', penalty='l2', alpha=0.0005, n_iter=200, random_state=42, n_jobs=-1, average=True),
        KNeighborsClassifier(10),
        AdaBoostRegressor(DecisionTreeRegressor(max_depth=25),n_estimators=300, random_state=rng),
        DecisionTreeClassifier(max_depth=11),
        RandomForestClassifier(max_depth=21, n_estimators=21, max_features=1),
        AdaBoostClassifier(),
        GaussianNB(),
        LDA(),
        QDA()
    ]
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        print("Fitting " + name + "...")
        clf.fit(data_train1, target_train1)
        print("Predicting...")
        score = clf.score(data_test, target_test)
        print(score)
        predicted_test = clf.fit(data_train1, target_train1).predict(data_test)
        print(metrics.classification_report(target_test, predicted_test))

def smote_data(data_train, target_train):
    y = np.bincount(target_train)
    ratio = 1.5 # float(y[2] + y[1]) / float(y[0])
  #  smote = SMOTE(ratio=ratio, verbose=True, kind='regular')
  #  smox, smoy = smote.fit_transform(data_train, target_train)
    OS = OverSampler(ratio=ratio, verbose=True)
    osx, osy = OS.fit_transform(data_train, target_train)
    return osx, osy
    
def sampling():
    verbose = False
    y = np.bincount(target_train1)
    print y
    ratio = float(y[2]) / float(y[1])
    # 'Random over-sampling'
    OS = OverSampler(ratio=ratio, verbose=verbose)
    osx, osy = OS.fit_transform(data_train1, target_train1)
    random_methods(osx,osy)
    # 'SMOTE'
    smote = SMOTE(ratio=ratio, verbose=verbose, kind='regular')
    smox, smoy = smote.fit_transform(data_train1, target_train1)
    random_methods(smox,smoy)
    # 'SMOTE bordeline 1'
    bsmote1 = SMOTE(ratio=ratio, verbose=verbose, kind='borderline1')
    bs1x, bs1y = bsmote1.fit_transform(data_train, target_train)
    random_methods(bs1x,bs1y)
    # 'SMOTE bordeline 2'
    bsmote2 = SMOTE(ratio=ratio, verbose=verbose, kind='borderline2')
    bs2x, bs2y = bsmote2.fit_transform(data_train1, target_train1)
    random_methods(bs2x,bs2y)
    # 'SMOTE SVM'
    svm_args={'class_weight' : 'auto'}
    svmsmote = SMOTE(ratio=ratio, verbose=verbose, kind='svm', **svm_args)
    svsx, svsy = svmsmote.fit_transform(data_train1, target_train1)
    random_methods(svsx,svsy)
    # 'SMOTE Tomek links'
    STK = SMOTETomek(ratio=ratio, verbose=verbose)
    stkx, stky = STK.fit_transform(data_train1, target_train1)
    random_methods(stkx,stky)
    # 'SMOTE ENN'
    SENN = SMOTEENN(ratio=ratio, verbose=verbose)
    ennx, enny = SENN.fit_transform(data_train1, target_train1)
    random_methods(ennx,enny)
    # 'EasyEnsemble'
    EE = EasyEnsemble(verbose=verbose)
    eex, eey = EE.fit_transform(data_train1, target_train1)
    random_methods(eex,eey)
    # 'BalanceCascade'
    BS = BalanceCascade(verbose=verbose)
    bsx, bsy = BS.fit_transform(data_train1, target_train1)
    random_methods(bsx,bsy)
#sampling()
