__author__ = 'Charlie'


import common
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import metrics

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

data_train, data_test, target_train, target_test = common.load_test_train_as_two_class()
sgd = SGDClassifier()
grid = GridSearchCV(sgd, params, cv=10)
grid.fit(data_train, target_train)

print(grid.best_estimator_)
print(grid.best_params_)
print(grid.best_score_)

predictions = grid.predict(data_test)
print(metrics.precision_recall_fscore_support(target_test, predictions))

# for score in grid.grid_scores_:
#     print(score.parameters, \
#           '[Mean]          = %5.4f%%' % score.mean_validation_score, \
#           '[Std Deviation] = %5.4f%%' % np.std(score.cv_validation_scores))