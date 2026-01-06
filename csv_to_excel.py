import pandas as pd

# Read CSV
df = pd.read_csv("qc_data.csv")

# Write to Excel
df.to_excel("nemotron_data.xlsx", index=False)
