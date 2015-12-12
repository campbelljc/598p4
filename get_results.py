import common
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from nolearn.lasagne import NeuralNet
from lasagne import layers
from lasagne.layers import DenseLayer, InputLayer, DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum

def save_results(real, predicted, algdesc, algname):
    conf_matrix = confusion_matrix(real, predicted)
    
    with open('results/'+algdesc+'_results.txt', "w") as f:
        f.write(classification_report(real, predicted))
        f.write(np.array_str(conf_matrix))
    
    plt.matshow(conf_matrix)
    plt.title("Confusion matrix (" + algname + ")")
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('figures/confmatrix_' + algdesc + '.png')
    plt.close()

def main():
    datafiles = ['data/processed_missing_filled_in.csv', 'data/processed_without_missing.csv', 'data/processed.csv']
    datanames = ['md=imputed', 'md=deleted', 'md=0s']

    num_samples_per_class = [-1, 6000]
    nsnames = ['ns=all', 'ns=6000']
    
    num_classes = [2, 3]
    cnames = ['nc=2', 'nc=3']
    
    oversample = [True, False]
    osnames = ["os=t", "os=f"]
    
    algnames = ["NN", "DT", "RandomForest", "AdaBoost", "GaussianNB", "LDA", "QDA", "SGD", "NNet"]
    algs = [
        KNeighborsClassifier(5),
        DecisionTreeClassifier(max_depth=25),
        RandomForestClassifier(max_depth=25, n_estimators=10, max_features=1),
        AdaBoostClassifier(),
        GaussianNB(),
        LDA(),
        QDA(),
        SGDClassifier(penalty='elasticnet', alpha=0.1, loss='modified_huber'),
        0
    ]
    
    for alg, algname in zip(algs, algnames):
        for dat, datname in zip(datafiles, datanames):
            for numspl, sname in zip(num_samples_per_class, nsnames):
                for numcls, cname in zip(num_classes, cnames):
                    for os, osname in zip(oversample, osnames):
                        algdesc = algname + "_" + datname + "_" + sname + "_" + cname + "_" + osname
                        print(algdesc)
                        input_train, input_test, output_train, output_test = common.load_train_data_and_split(file=dat, num_samples_per_class=numspl, num_classes=numcls, smote=os)
                        
                        if algname is "NNet":
                            alg = NeuralNet(layers=[('input', InputLayer), ('dense0', DenseLayer), ('dropout0', DropoutLayer), ('dense1', DenseLayer), ('dropout1', DropoutLayer), ('output', DenseLayer)], input_shape=(None, input_train.shape[1]), dense0_num_units=300, dropout0_p=0.075, dropout1_p=0.1, dense1_num_units=750, output_num_units=numcls, output_nonlinearity=softmax, update=nesterov_momentum, update_learning_rate=0.001, update_momentum=0.99, eval_size=0.33, verbose=1, max_epochs=15)
                        
                        model = alg.fit(input_train, output_train)
                        
                        print("TRAIN ", algdesc)
                        predictions_train = model.predict(input_train)
                        save_results(output_train, predictions_train, algdesc+"_train", algname)
                        
                        print("TEST ", algdesc)
                        predictions_test = model.predict(input_test)
                        save_results(output_test, predictions_test, algdesc+"_test", algname)
    
if __name__ == '__main__':
    main()