import csv

def categorize(data, colnum, missingvals):
    categories = set()
    for row in data:
        if row[colnum] not in missingvals:
            categories.add(row[colnum])
    catlist = list(categories)
    catlist.sort()
    print(catlist, "(with missing vals:", missingvals, ")")
    for row in data:
        if row[colnum] in missingvals: # missing data
            row[colnum] = 0;
        else:
            row[colnum] = catlist.index(row[colnum])+1

print("Loading dataset.")

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


print("Removing duplicate patient encounters.")

data.sort(key=lambda x: x[1])
    
delete = []
for i in range(1, len(data)): # iterate over each encounter
    if (data[i-1][1] == data[i][1]): # if the patient IDs are the same...
        if (data[i-1][0] < data[i][0]): # if the i-1 entry is lower encounter...
            delete.append(i)
            
for i in reversed(range(len(delete))):
    del data[delete[i]]
    

print("Turning race into category...")

categorize(data, 2, ['?']) # race
categorize(data, 3, ['Unknown/Invalid']) # gender
categorize(data, 4, []) # age
categorize(data, 5, ['?']) # weight
categorize(data, 10, ['?']) # payer code
categorize(data, 11, ['?']) # medical_specialty

# 19, 20 -> diag_2, diag_3...

for i in range(22, len(data[0])):
    categorize(data, i, [])

for i in range(10):
    print(data[i])
    print("\n")
    
data = headers + data
    
# ref: http://stackoverflow.com/questions/7588934/deleting-columns-in-a-csv-with-python
# save back to csv

 # remove encounterID, patientID, gender, weight, payer code
with open("data/processed.csv","wb") as f:
    wtr = csv.writer(f)
    for r in data:
        wtr.writerow((r[2], r[4], r[6], r[7], r[8], r[9], r[11], r[12], r[13], r[14], r[15], r[16], r[17], r[18], r[19], r[20], r[21], r[22], r[23], r[24], r[25], r[26], r[27], r[28], r[29], r[30], r[31], r[32], r[33], r[34], r[35], r[36], r[37], r[38], r[39], r[40], r[41], r[42], r[43], r[44], r[45], r[46], r[47], r[48], r[49]))