import pandas as pd

path = "./data/cs-training.csv"
data = pd.read_csv(path).drop("Unnamed: 0", axis=1)
#data.rename(columns={"Unnamed: 0": "ID"}, inplace=True)
#data.drop("ID", axis=1)
print(data.head())
print(data.columns)
cleancolunms = []
for i in range(len(data.columns)):
    cleancolunms.append(data.columns[i].replace('-', '').lower())
data.columns = cleancolunms
print(data.head())
data_col = data[data.columns[1:]]
print(data_col.describe())
print('median(중위수):\n', data_col.median(), '\n')
print('mean(평균):', data_col.mean())
print('colunms:', data.columns)
total_len = len(data['seriousdlqin2yrs'])
percentage_labels = (data['seriousdlqin2yrs'].value_counts()/total_len)*100
print(percentage_labels)
print(data['seriousdlqin2yrs'].value_counts())

