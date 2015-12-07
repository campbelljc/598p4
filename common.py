import csv
from sklearn.cross_validation import train_test_split

def load_train_data_and_split(testsize=0.3):
    print("Loading dataset.")

    headers = []
    data = []
    ifile  = open('data/processed_without_missing.csv', "r")
    reader = csv.reader(ifile)
    i = 0
    for row in reader:
        if i == 0:
            i = 1
            headers.append(row)
            continue
        data.append(row)
        for i in row:
            if not i.isdigit():
                print (row)

    ifile.close()
    
    inputs = [[int(y) for y in x] for x in data]
    
    outputs = []
    for row in inputs:
        outputs.append(row[-1])
        del row[-1]
        
    print("Num inputs: ", len(inputs))
    
    input_train, input_test, output_train, output_test = train_test_split(inputs, outputs, test_size=testsize, random_state=42)
    print("Done loading")
    return input_train, input_test, output_train, output_test