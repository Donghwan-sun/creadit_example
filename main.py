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