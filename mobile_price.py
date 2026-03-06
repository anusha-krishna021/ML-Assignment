import pandas as pd

data = pd.read_csv("train.csv")

print(data.head())
print(data.info())
print(data.columns)
