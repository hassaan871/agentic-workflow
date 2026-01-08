import pandas as pd

# Read CSV
df = pd.read_csv("mim_data.csv")

# Write to Excel
df.to_excel("mim_data.xlsx", index=False)
