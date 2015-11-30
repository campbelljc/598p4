from __future__ import division
import csv
import common
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split

data_train, data_test, target_train, target_test = common.load_train_data_and_split()

clf = SGDClassifier(loss='hinge', penalty='l2', alpha=0.005, n_iter=10, random_state=42, n_jobs=-1, average=True)

predicted_train = clf.fit(data_train, target_train).predict(data_train)
train_p = ((target_train != predicted_train).sum())/(len(data_train))*100
print("Error on train set: %d" % train_p)

predicted_train = clf.fit(data_train, target_train).predict(data_test)
train_p = ((target_test != predicted_train).sum())/(len(data_test))*100
print("Error on test set: %d" % train_p)