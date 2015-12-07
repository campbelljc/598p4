from __future__ import division
import common
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

data_train, data_test, target_train, target_test = common.load_train_data_and_split()

clf = SGDClassifier(loss='hinge', penalty='l2', alpha=0.005, n_iter=10, random_state=42, n_jobs=-1, average=True)

predicted_train = clf.fit(data_train, target_train).predict(data_train)
train_p = ((target_train != predicted_train).sum())/(len(data_train))*100
print("Error on train set: %d" % train_p)

predicted_test = clf.fit(data_train, target_train).predict(data_test)
test_p = ((target_test != predicted_test).sum())/(len(data_test))*100
print("Error on test set: %d" % test_p)

print("SVM:\n%s\n" % (
    metrics.classification_report(target_test, predicted_test)))

print(clf.score(data_test, target_test))

cm = confusion_matrix(target_test, predicted_test)
print (cm)
plt.figure(1)
plt.matshow(cm)
plt.title('Confusion matrix (SVM)')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confmatrix_svm.png')