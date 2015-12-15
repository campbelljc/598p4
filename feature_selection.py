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
import get_results

def build_pipelines(ls, ss):
    pipelines = []
    for learner_tup, learner_params in ls.items():
        for selector_tup, selector_params in ss.items():
            learner_name = learner_tup[0]
            learner = learner_tup[1]
            selector_name = selector_tup[0]
            selector = selector_tup[1]
            alg_name = "-".join([learner_name, selector_name])
            pipe = Pipeline([
                (selector_name, selector),
                (learner_name, learner)
            ])
            params = dict(learner_params, **selector_params)
            pipelines.append((alg_name, pipe, params))
    return pipelines


selectors = {
    ('Percentile', SelectPercentile()): {
        'Percentile__percentile': (1, 5)
    },

    ('PCA', PCA()): {
        'PCA__n_components': (2, 4, 8, 16, 32)
    }
}

learners = {
    ('SGD', SGDClassifier()): {
        'SGD__loss': ('hinge', 'squared_hinge', 'modified_huber'),
        'SGD__penalty': ('l2', 'l1', 'elasticnet'),
        'SGD__alpha': tuple([0.1 ** x for x in range(1, 5)])
    }
}

data_train, data_test, target_train, target_test = common.load_train_data_and_split(num_samples_per_class=6000, file='data/processed_missing_filled_in.csv') # 0.21

for alg_name, pipeline, params in build_pipelines(learners, selectors):
    grid = GridSearchCV(pipeline, params, cv=3, scoring='f1_weighted')
    grid.fit(data_train, target_train)
    predictions = grid.predict(data_test)
    get_results.save_results(target_test, predictions, alg_name, alg_name)


    print(grid.best_estimator_)
    print(grid.best_params_)
    print(grid.best_score_)
    print(metrics.classification_report(target_test, predictions))
