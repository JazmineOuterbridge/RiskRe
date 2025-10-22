import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('data/insurance2.csv')
print("Total samples:", len(df))
print("Regions:", df['region'].unique())

# Test each region
for region in df['region'].unique():
    df_filtered = df[df['region'] == region]
    print(f"\n{region} region:")
    print(f"  Samples: {len(df_filtered)}")
    
    # Check high risk samples
    high_risk = df_filtered[df_filtered['charges'] > 10000]
    print(f"  High risk samples: {len(high_risk)}")
    
    # Test stratification
    y_class = (df_filtered['charges'] > 10000).astype(int)
    class_counts = np.bincount(y_class)
    print(f"  Class distribution: {class_counts}")
    print(f"  Min samples per class: {min(class_counts)}")
    
    # Test if stratification would work
    if min(class_counts) >= 2:
        print("  ✅ Stratification OK")
    else:
        print("  ⚠️  Stratification would fail - need fallback")
