import pandas as pd

# Read CSV
df = pd.read_csv("data.csv")

# Write to Excel
df.to_excel("data.xlsx", index=False)
