import pandas as pd

# Read CSV
df = pd.read_csv("itf_data.csv")

# Write to Excel
df.to_excel("itf_data.xlsx", index=False)
