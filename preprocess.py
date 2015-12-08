import csv
import re

# Take in dataset and turn values of specified columns into categorical
# range 1...n, reserving 0 for missing values.
def categorize(data, colnum, missingvals, ranges=[]):
    categories = set()
    for row in data:
        if row[colnum] not in missingvals:
            categories.add(row[colnum])
    catlist = list(categories)
    catlist.sort()
#    print(', '.join(['%i: %s' % (n, catlist[n]) for n in xrange(len(catlist))]), "(with missing vals:", missingvals, ")")
    
    missing_indices = []
    for index, row in enumerate(data):
        if row[colnum] in missingvals: # missing data
            row[colnum] = 0
            missing_indices.append(index)
        else: # this row doesn't have missing data.
            if len(ranges) > 0: # find val in ranges and use that index.
                found = False
                for i, r in enumerate(ranges):
                    if isinstance(r, basestring): # compare strings.
                        if r in row[colnum]:
                            row[colnum] = i
                            found = True
                            break
                    elif isinstance(r, ( int, long )) and not re.search('[a-zA-Z]', row[colnum]):
                        # ref : http://stackoverflow.com/questions/3501382/checking-whether-a-variable-is-an-integer-or-not
                        if float(row[colnum]) >= r and len(ranges) > i+1 and isinstance(ranges[i+1], ( int, long )) and float(row[colnum]) < ranges[i+1]:
                            row[colnum] = i
                            found = True
                            break
                if not found:
                    print(row[colnum]) # error here
            else: # no ranges given, so just set category of appearance.
                row[colnum] = catlist.index(row[colnum])+1
    return missing_indices

def load_dataset():
    headers = []
    data = []
    ifile  = open('data/diabetic_data.csv', "r")
    reader = csv.reader(ifile)
    i = 0
    for row in reader:
        if i == 0:
            i = 1
            headers.append(row)
            continue
        data.append(row)
    ifile.close()
    return headers, data

# Load the dataset from the csv. file.
print("Loading dataset.")
headers, data = load_dataset()


print("Removing duplicate patient encounters.")

# Sort data by patient ID.
data.sort(key=lambda x: x[1])
    
delete = []
for i in range(1, len(data)): # iterate over each encounter
    if (data[i-1][1] == data[i][1]): # if the patient IDs are the same...
        if (data[i-1][0] < data[i][0]): # if the i-1 entry is lower encounter...
            delete.append(i)

# Delete duplicate patient encounters (leave only one remaining).
for i in reversed(range(len(delete))):
    del data[delete[i]]
    
print("Categorizing...")

missing_indices = set() # will store the indices for the rows that have missing data

# Turn all indicated features into categorical values from 1 to n. 0 is reserved for missing value.
categorize(data, 2, ['?']) # race
#categorize(data, 3, ['Unknown/Invalid']) # gender
categorize(data, 4, []) # age
#categorize(data, 5, ['?']) # weight
#categorize(data, 10, ['?']) # payer code

# deal with medical speciality (feature 11)
missing_indices = missing_indices.union(categorize(data, 11, ['?'])) # remove data with missing values
#categorize(data, 11, ['?']) # include all data

# feats 18, 19, 20 -> diagnosis 1, 2, 3.
# To not group various diagnoses together corresponding to ICD9 code category, then remove the last list argument.
# ICD9 code category ref : https://en.wikipedia.org/wiki/List_of_ICD-9_codes
categorize(data, 18, ['?'], [001, 140, 240, 280, 290, 320, 360, 390, 460, 520, 580, 630, 680, 710, 740, 760, 780, 800, 1000, 'E', 'V'])
categorize(data, 19, ['?'], [001, 140, 240, 280, 290, 320, 360, 390, 460, 520, 580, 630, 680, 710, 740, 760, 780, 800, 1000, 'E', 'V'])
categorize(data, 20, ['?'], [001, 140, 240, 280, 290, 320, 360, 390, 460, 520, 580, 630, 680, 710, 740, 760, 780, 800, 1000, 'E', 'V'])

for i in range(22, len(data[0])):
    categorize(data, i, [])

data = headers + data

# ref: http://stackoverflow.com/questions/7588934/deleting-columns-in-a-csv-with-python
# save back to csv

# remove encounterID, patientID, gender, weight, payer code
# this saves entire processed dataset
with open("data/processed.csv", "wb") as f:
    wtr = csv.writer(f)
    for r in data:
        wtr.writerow((r[2], r[4], r[6], r[7], r[8], r[9], r[11], r[12], r[13], r[14], r[15], r[16], r[17], r[18], r[19], r[20], r[21], r[22], r[23], r[24], r[25], r[26], r[27], r[28], r[29], r[30], r[31], r[32], r[33], r[34], r[35], r[36], r[37], r[38], r[39], r[40], r[41], r[42], r[43], r[44], r[45], r[46], r[47], r[48], r[49]))
        
# this saves dataset except for entries where there is missing data in the missing speciality feature.
with open("data/processed_without_missing.csv", "wb") as f:
    wtr = csv.writer(f)
    for index, r in enumerate(data):
        if (index-1) not in missing_indices: # index+1 since we added the header row to the top.
            wtr.writerow((r[2], r[4], r[6], r[7], r[8], r[9], r[11], r[12], r[13], r[14], r[15], r[16], r[17], r[18], r[19], r[20], r[21], r[22], r[23], r[24], r[25], r[26], r[27], r[28], r[29], r[30], r[31], r[32], r[33], r[34], r[35], r[36], r[37], r[38], r[39], r[40], r[41], r[42], r[43], r[44], r[45], r[46], r[47], r[48], r[49]))
        
with open("data/processed_only_missing.csv", "wb") as f:
    wtr = csv.writer(f)
    for index, r in enumerate(data):
        if (index-1) in missing_indices or index is 0: # index+1 since we added the header row to the top.
            wtr.writerow((r[2], r[4], r[6], r[7], r[8], r[9], r[11], r[12], r[13], r[14], r[15], r[16], r[17], r[18], r[19], r[20], r[21], r[22], r[23], r[24], r[25], r[26], r[27], r[28], r[29], r[30], r[31], r[32], r[33], r[34], r[35], r[36], r[37], r[38], r[39], r[40], r[41], r[42], r[43], r[44], r[45], r[46], r[47], r[48], r[49]))
            
print("Filling in missing medical speciality values...")
import imputation
predictions = imputation.dt_classifier()
#print(predictions)
with open("data/processed_missing_filled_in.csv", "wb") as f:
    wtr = csv.writer(f)
    count = 0
    for index, r in enumerate(data):
        if (index-1) in missing_indices: # index+1 since we added the header row to the top.
            r[11] = predictions[count]
            count = count+1
        wtr.writerow((r[2], r[4], r[6], r[7], r[8], r[9], r[11], r[12], r[13], r[14], r[15], r[16], r[17], r[18], r[19], r[20], r[21], r[22], r[23], r[24], r[25], r[26], r[27], r[28], r[29], r[30], r[31], r[32], r[33], r[34], r[35], r[36], r[37], r[38], r[39], r[40], r[41], r[42], r[43], r[44], r[45], r[46], r[47], r[48], r[49]))