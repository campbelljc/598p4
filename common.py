import csv
import smote as sm
import numpy as np
from sklearn.cross_validation import train_test_split

def load_train_data_and_split(testsize=0.3, targetcol=-1, file='data/processed_without_missing.csv', split=True, num_samples_per_class=-1, num_classes=3, smote=False):
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
        
    if num_samples_per_class > 0: # we want equal subsets of each class.
        # first get the number of classes and their values.
        output_vals = set()
        for i, row in enumerate(inputs):
            output_vals.add(outputs[i])
        print("Number of classes: ", len(output_vals))
        
        # then delete samples that go over the 3000 limit.
        counts = [0, 0, 0]
        remove_indices = []
        for i, row in enumerate(inputs):
            counts[outputs[i]-1] += 1 # mapping from target (1,2,3) to array index (0,1,2)
            if (counts[outputs[i]-1] > num_samples_per_class): # we exceeded the count so delete this row.
                remove_indices.append(i)
                                
        for i in reversed(range(len(remove_indices))):
            del inputs[remove_indices[i]]
            del outputs[remove_indices[i]]
        print("Final counts: ", counts)
        
    print("Num rows: ", len(inputs))
    print("Done loading")
    
    for i in range(len(outputs)):
        outputs[i] -= 1
        
    if num_classes is 2: # convert all outputs of 3 to outputs of 2.
        outputs[outputs == 3] = 2
        
    if split:
        input_train, input_test, output_train, output_test = train_test_split(inputs, outputs, test_size=testsize, random_state=42)
        input_train = np.array(input_train)
        input_test = np.array(input_test)
        output_train = np.array(output_train)
        output_train = output_train.astype(np.int32)
        output_test = np.array(output_test)
        output_test = output_test.astype(np.int32)
        
        if smote:
            input_train, output_train = sm.smote_data(input_train, output_train)
        
        return input_train, input_test, output_train, output_test
    else:
        inputs = np.array(inputs)
        outputs = np.array(outputs)
        return inputs, outputs
