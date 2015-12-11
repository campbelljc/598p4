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
    # print(', '.join(['%i: %s' % (n, catlist[n]) for n in xrange(len(catlist))]), "(with missing vals:", missingvals, ")")
    
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

def binarize(data, colnum, missingvals):
    # return a list of arrays, with one column for each different value in the specified colnum of input data array
    # also return the headers of this array.
    
    col_vals = set()
    for row in data: # get all the different values in this column.
        if row[colnum] not in missingvals: # don't record missing vals.
            col_vals.add(row[colnum])
    
    new_data_cols = []
    for i, row in enumerate(data):
        column_to_binary_vals = []
        for cval in col_vals:
            if row[colnum] == cval:
                column_to_binary_vals.append(1)
            else:
                column_to_binary_vals.append(0)
        new_data_cols.append(column_to_binary_vals) # one row for each data item

    headers = []
    for cval in col_vals:
        name = "is" + str(cval)
        headers.append(name)
    
    return headers, new_data_cols

def add_cols(data, headers, new_headers, new_data_cols):    
    headers[0] = headers[0] + list(new_headers)
    for i in range(len(data)):
        data[i] = data[i] + new_data_cols[i]
    return headers, data

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
    
def remove_duplicate_encounters(data):
    data.sort(key=lambda x: x[1]) # sort data by patient ID    
    delete = []
    for i in range(1, len(data)): # iterate over each encounter
        if (data[i-1][1] == data[i][1]): # if the patient IDs are the same...
            if (data[i-1][0] < data[i][0]): # if the i-1 entry is lower encounter...
                delete.append(i)
                
    for i in reversed(range(len(delete))): # delete duplicate patient encounters (leave only one remaining)
        del data[delete[i]]
        
    return data
    
def main():
    print("Loading dataset.")
    headers, data = load_dataset()

    print("Removing duplicate patient encounters.")
    data = remove_duplicate_encounters(data)
   
    print("Categorizing...")
    missing_indices = set() # will store the indices for the rows that have missing data

    # Turn all indicated features into categorical values from 1 to n. 0 is reserved for missing value.
    
   # categorize(data, 2, ['?']) # race
    
    headers_to_add = []
    cols_to_add = [[] for x in xrange(len(data))]
    del_cols = [0, 1, 3, 5, 10] # remove patient id, encounter id, gender, weight, payer code.
    
    new_headers, new_data_cols = binarize(data, 2, ['?'])
    headers_to_add += new_headers
    cols_to_add += new_data_cols
    del_cols.append(2)
    
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
        
    # remove last column and re-add later.
    outputs = []
    output_header = headers[0][-1]
    for row in data:
        outputs.append(row[-1])
        del row[-1]
    del headers[0][-1]

    del_cols.sort()
    for colnum in reversed(del_cols):
    #    print("Headers length is ", len(headers[0]), " and we are popping ", colnum)
        headers[0].pop(colnum)
        for row in data:
            row.pop(colnum)
            #del row[colnum]
            
    headers, data = add_cols(data, headers, new_headers, new_data_cols)

    headers[0].append(output_header)
    data = headers + data
    
    for i in range(len(data)):
        if i is 0:
            continue
        data[i].append(outputs[i-1])

    # ref: http://stackoverflow.com/questions/7588934/deleting-columns-in-a-csv-with-python
    # save back to csv

    # remove encounterID, patientID, gender, weight, payer code
    # this saves entire processed dataset
    with open("data/processed.csv", "wb") as f:
        wtr = csv.writer(f)
        for r in data:
            wtr.writerow(r)
#            wtr.writerow((r[2], r[4], r[6], r[7], r[8], r[9], r[11], r[12], r[13], r[14], r[15], r[16], r[17], r[18], r[19], r[20], r[21], r[22], r[23], r[24], r[25], r[26], r[27], r[28], r[29], r[30], r[31], r[32], r[33], r[34], r[35], r[36], r[37], r[38], r[39], r[40], r[41], r[42], r[43], r[44], r[45], r[46], r[47], r[48], r[49]))
    
    # this saves dataset except for entries where there is missing data in the missing speciality feature.
    with open("data/processed_without_missing.csv", "wb") as f:
        wtr = csv.writer(f)
        for index, r in enumerate(data):
            if (index-1) not in missing_indices: # index+1 since we added the header row to the top.
                wtr.writerow(r)
              #  wtr.writerow((r[2], r[4], r[6], r[7], r[8], r[9], r[11], r[12], r[13], r[14], r[15], r[16], r[17], r[18], r[19], r[20], r[21], r[22], r[23], r[24], r[25], r[26], r[27], r[28], r[29], r[30], r[31], r[32], r[33], r[34], r[35], r[36], r[37], r[38], r[39], r[40], r[41], r[42], r[43], r[44], r[45], r[46], r[47], r[48], r[49]))
    
    with open("data/processed_only_missing.csv", "wb") as f:
        wtr = csv.writer(f)
        for index, r in enumerate(data):
            if (index-1) in missing_indices or index is 0: # index+1 since we added the header row to the top.
                wtr.writerow(r)
               # wtr.writerow((r[2], r[4], r[6], r[7], r[8], r[9], r[11], r[12], r[13], r[14], r[15], r[16], r[17], r[18], r[19], r[20], r[21], r[22], r[23], r[24], r[25], r[26], r[27], r[28], r[29], r[30], r[31], r[32], r[33], r[34], r[35], r[36], r[37], r[38], r[39], r[40], r[41], r[42], r[43], r[44], r[45], r[46], r[47], r[48], r[49]))
        
    print("Filling in missing medical speciality values...")
    import imputation
    predictions = imputation.dt_classifier()
    
    #find med specialty col.
    medcol = -1
    for i, h in enumerate(headers[0]):
        if 'medical_specialty' in h:
            medcol = i
            break
    print(medcol)
    
    predcount = 0
    for index in range(len(data)):
        if (index-1) in missing_indices:
            data[index][medcol] = predictions[predcount]
            predcount += 1
            
    data.pop(0)
    new_headers, new_data_cols = binarize(data, medcol, ['?'])
    headers_to_add = new_headers
    cols_to_add = new_data_cols
    
    headers[0].pop(medcol)
    
    # remove last column and re-add later.
    outputs = []
    output_header = headers[0][-1]
    for row in data:
        outputs.append(row[-1])
        del row[-1]
    del headers[0][-1]
    
    for row in data:
        row.pop(medcol)
    headers, data = add_cols(data, headers, new_headers, new_data_cols)
    
    

    headers[0].append(output_header)
    data = headers + data
    
    for i in range(len(data)):
        if i is 0:
            continue
        data[i].append(outputs[i-1])
    
    with open("data/processed_missing_filled_in.csv", "wb") as f:
        wtr = csv.writer(f)
   #     count = 0
        for index, r in enumerate(data):
    #        if (index-1) in missing_indices: # index+1 since we added the header row to the top.
     #           r[11] = predictions[count]
      #          count = count+1
            wtr.writerow(r)
       #     wtr.writerow((r[2], r[4], r[6], r[7], r[8], r[9], r[11], r[12], r[13], r[14], r[15], r[16], r[17], r[18], r[19], r[20], r[21], r[22], r[23], r[24], r[25], r[26], r[27], r[28], r[29], r[30], r[31], r[32], r[33], r[34], r[35], r[36], r[37], r[38], r[39], r[40], r[41], r[42], r[43], r[44], r[45], r[46], r[47], r[48], r[49]))
            
if __name__ == '__main__':
    main()