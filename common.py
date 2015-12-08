import csv
from sklearn.cross_validation import train_test_split
import numpy as np

def load_train_data_and_split(testsize=0.3, targetcol=-1, file='data/processed_without_missing.csv', split=True):
    print("Loading dataset.")

    headers = []
    dataset = []
    ifile  = open(file, "r")
    reader = csv.reader(ifile)
    first = True
    for row in reader:
        if first:
            first = False
            headers.append(row)
            continue
        dataset.append(row)
        for i in row:
            if not i.isdigit():
                print (row)

    ifile.close()
    
    inputs = [[int(y) for y in x] for x in dataset]
    
    outputs = []
    for row in inputs:
        outputs.append(row[targetcol])
        del row[targetcol]
        
    print("Num inputs: ", len(inputs))
    print("Done loading")
    
    if split:
        input_train, input_test, output_train, output_test = train_test_split(inputs, outputs, test_size=testsize, random_state=42)
        return input_train, input_test, output_train, output_test
    else:
        return inputs

def load_test_train_as_two_class(ts=0.3, f='data/processed_missing_filled_in.csv'):
    x_train, x_test, y_train, y_test = load_train_data_and_split(testsize=ts, file=f)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    y_train[y_train == 3] = 2
    y_test[y_test == 3] = 2
    return x_train, x_test, y_train, y_test
