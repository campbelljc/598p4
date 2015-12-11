# ref : https://github.com/ottogroup/kaggle/blob/master/Otto_Group_Competition.ipynb
# ref : https://gist.github.com/dnouri/fe855653e9757e1ce8c4

import common
import smote
import numpy as np
from lasagne import layers
from lasagne.layers import DenseLayer, InputLayer, DropoutLayer
from lasagne.nonlinearities import rectify, softmax, tanh, linear
from lasagne.updates import nesterov_momentum, rmsprop, momentum
from lasagne.objectives import categorical_crossentropy
from nolearn.lasagne import NeuralNet
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform as sp_rand
from scipy.stats import randint as sp_randint

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# ref : https://github.com/dnouri/nolearn/issues/95
def lasagne_grid_search(X, y):

    
    param_dist = {"update_learning_rate": [0.001, 0.01],
                  "dense0_num_units": [100, 300],
                  "dense1_num_units": [100, 300],
                  "dropout0_p": [0.1, 0.75],
                  "dropout1_p": [0.1, 0.75],
                  "update_momentum" : [0.9, 0.99] #,
                 }
    
        # run 1 best : [CV]  update_learning_rate=0.001, dropout1_p=0.1, update_momentum=0.99, dropout0_p=0.1, dense1_num_units=300, dense0_num_units=300, score=0.603377 -  16.8s
        
    param_dist = {"update_learning_rate": [0.0001, 0.001],
                  "dense0_num_units": [300, 600],
                  "dense1_num_units": [300, 600],
                  "dropout0_p": [0.05, 0.1],
                  "dropout1_p": [0.05, 0.1],
                  "update_momentum" : [0.99, 0.9999]
                 }
                 
        # run 2 best : [CV]  update_learning_rate=0.001, dropout1_p=0.1, update_momentum=0.99, dropout0_p=0.05, dense1_num_units=600, dense0_num_units=300, score=0.605281 -  25.1s
        
    param_dist = {"update_learning_rate": [0.001],
                  "dense0_num_units": [300],
                  "dense1_num_units": [450, 600, 750],
                  "dropout0_p": [0.025, 0.05, 0.075, 0.1, 0.25],
                  "dropout1_p": [0.1],
                  "update_momentum" : [0.99]
                 }

        # run 3 best : [CV]  update_learning_rate=0.001, dropout1_p=0.1, update_momentum=0.99, dropout0_p=0.075, dense1_num_units=750, dense0_num_units=300, score=0.605437 -  29.5s


    layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),  
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),  
           ('output', DenseLayer)]
          
    net0 = NeuralNet(layers=layers0,
                     input_shape=(None, X.shape[1]),
                     output_num_units=3,
                     output_nonlinearity=softmax,
                 ##    objective_loss_function =categorical_crossentropy
                     update=nesterov_momentum,                 
                     eval_size=0.2,
                     verbose=1,
                     max_epochs=20
                     )
                     
    random_search = GridSearchCV(net0, param_grid=param_dist, cv=2, verbose=4) #, scoring=NeuralNet.score , cv=2, n_jobs=1, verbose=5, refit=False)
    random_search.fit(X, y)
    
def classify(X, y, X_test, y_test):
    layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),  
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),  
           ('output', DenseLayer)]
               
    net = NeuralNet(layers=layers0,
                     input_shape=(None, X.shape[1]),
                     dense0_num_units=300,
                     dropout0_p=0.075,
                     dropout1_p=0.1,
                     dense1_num_units=750,
                     output_num_units=3,
                     output_nonlinearity=softmax,
                     update=nesterov_momentum,
                     update_learning_rate=0.001,
                     update_momentum=0.99,
                 
                     eval_size=0.2,
                     verbose=1,
                     max_epochs=15)

    net.fit(X, y)
    print(net.score(X, y))
    
    preds = net.predict(X_test)
    print(classification_report(y_test, preds))
    cm = confusion_matrix(y_test, preds)
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confmatrix.png')
    plt.show()
    
    print(cm)

def main():
    data_train, data_test, target_train, target_test = common.load_train_data_and_split(file='data/processed_missing_filled_in.csv')
   
    data_train = np.asarray(data_train)
    target_train = np.array(target_train)
    target_train = target_train.astype(np.int32)
    
 #   data_train, target_train = smote.smote_data(data_train, target_train)
    
    classify(data_train, target_train, data_test, target_test)
 #   lasagne_grid_search(data_train, target_train)

if __name__ == '__main__':
    main()