import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/insurance2.csv')
print("Total samples:", len(df))
print("Regions:", df['region'].unique())

# Test each region
for region in df['region'].unique():
    df_filtered = df[df['region'] == region]
    high_risk = df_filtered[df_filtered['charges'] > 10000]
    print(f"{region}: {len(df_filtered)} samples, {len(high_risk)} high risk")
