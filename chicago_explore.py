import pandas as pd

df = pd.read_csv('chicago_traffic.csv')
print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nFirst 3 rows:\n", df.head(3))
print("\nMissing values:\n", df.isnull().sum())
print("\nSpeed stats:\n", df.describe())