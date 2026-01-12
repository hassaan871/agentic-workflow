import pandas as pd

# Read CSV
df = pd.read_csv("new_qc_data_20260112_203447.csv")

# Write to Excel
df.to_excel("new_qc_data_20260112_203447.xlsx", index=False)
