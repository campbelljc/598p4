__author__ = 'Charlie'


import common
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.decomposition import PCA
from sklearn import metrics
import numpy as np

def build_pipelines(ls, ss):
    pipelines = []
    for learner_tup, learner_params in ls.items():
        for selector_tup, selector_params in ss.items():
            learner_name = learner_tup[0]
            learner = learner_tup[1]
            selector_name = selector_tup[0]
            selector = selector_tup[1]
            pipe = Pipeline([
                (selector_name, selector),
                (learner_name, learner)
            ])
            params = dict(learner_params, **selector_params)
            pipelines.append((pipe, params))
    return pipelines


selectors = {
    ('percentile', SelectPercentile()): {
        'percentile__percentile': (1, 5)
    },

    ('pca', PCA()): {
        'pca__n_components': (2, 4, 8, 16, 32)
    }
}

learners = {
    ('sgd', SGDClassifier()): {
        'sgd__loss': ('hinge', 'squared_hinge', 'modified_huber'),
        'sgd__penalty': ('l2', 'l1', 'elasticnet'),
        'sgd__alpha': tuple([0.1 ** x for x in range(1, 5)])
    }
}

data_train, data_test, target_train, target_test = common.load_train_data_and_split(num_samples_per_class=6000, file='data/processed_missing_filled_in.csv') # 0.21

for pipeline, params in build_pipelines(learners, selectors):
    grid = GridSearchCV(pipeline, params, cv=3, scoring='f1')
    grid.git(data_train, target_train)
    predictions = grid.predict(data_test)

    print(grid.best_estimator_)
    print(grid.best_params_)
    print(grid.best_score_)
    print(metrics.classification_report(target_test, predictions))
