__author__ = 'Charlie'

import common
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np

# learners = {
#     ('sgd', SGDClassifier()): {
#         'sgd__loss': ('hinge', 'squared_hinge', 'modified_huber'),
#         'sgd__penalty': ('l2', 'l1', 'elasticnet'),
#         'sgd__kernel': ('rbf', 'sigmoid', 'linear'),
#         'sgd__alpha': tuple([0.1 ** x for x in range(1, 5)])
#     }
# }

params = {
    'loss': ['hinge', 'squared_hinge', 'modified_huber'],
    # 'loss': ['hinge'],

    'penalty': ['l2', 'l1', 'elasticnet'],
    # 'penalty': ['l2', 'l1', 'elasticnet'],

    'alpha': [0.1 ** x for x in range(1, 5)]
    # 'alpha': [.001]
}

#data_train, data_test, target_train, target_test = common.load_test_train_as_two_class(f='data/processed_missing_filled_in.csv')
#data_train, data_test, target_train, target_test = common.load_test_train_as_two_class(f='data/processed_without_missing.csv')
#data_train, data_test, target_train, target_test = common.load_train_data_and_split() # 0.53
#data_train, data_test, target_train, target_test = common.load_train_data_and_split(num_samples_per_class=3000) # 0.24
#data_train, data_test, target_train, target_test = common.load_train_data_and_split(num_samples_per_class=6000, file='data/processed_missing_filled_in.csv') # 0.21
data_train, data_test, target_train, target_test = common.load_train_data_and_split(file='data/processed_missing_filled_in.csv') # 0.49
sgd = SGDClassifier()
grid = GridSearchCV(sgd, params, cv=10, verbose=10)
grid.fit(data_train, target_train)

print(grid.best_estimator_)
print(grid.best_params_)
print(grid.best_score_)

predictions = grid.predict(data_test)
np.save('data/predictions', predictions)
print(metrics.precision_recall_fscore_support(target_test, predictions))
print(metrics.classification_report(target_test, predictions))

cm = confusion_matrix(target_test, predictions)
print (cm)
plt.figure(1)
plt.matshow(cm)
plt.title('Confusion matrix (SVM)')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('figures/confmatrix_csvm.png')

# for score in grid.grid_scores_:
#     print(score.parameters, \
#           '[Mean]          = %5.4f%%' % score.mean_validation_score, \
#           '[Std Deviation] = %5.4f%%' % np.std(score.cv_validation_scores))