import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

### Data Preprocessing Functions ###

### Removing missing values ###

# ...existing code...
df = pd.read_csv('2022_2025_city_of_london_street.csv', low_memory=False)

# Basic info
print("Initial dataframe info:")
df.info()

# Compute missing percentage per column from the actual dataset
missing_pct = df.isna().mean() * 100
missing_pct = missing_pct.sort_values(ascending=False)

print("\nTop 20 columns by missing percentage:")
print(missing_pct.head(20))

# Automatically pick columns with >50% missing
cols_to_drop = missing_pct[missing_pct > 50].index.tolist()

# Manual candidates to drop (keep these generic; only existing columns will be used)
manual_drop_candidates = [
    'LearnCodeOnline', 'OpSysPersonal use', 'ICorPM', 'ProfessionalTech', 'TBranch',
    'AISearchHaveWorkedWith', 'MiscTechHaveWorkedWith',
    'OfficeStackAsyncHaveWorkedWith', 'OfficeStackSyncHaveWorkedWith', 'NEWCollabToolsHaveWorkedWith',
    'Respondent'  # Common meta/id column in some survey exports
]

# Filter manual list to only columns that exist in this dataset
manual_drop_existing = [c for c in manual_drop_candidates if c in df.columns]

if manual_drop_candidates:
    missing_manual = [c for c in manual_drop_candidates if c not in df.columns]
    if missing_manual:
        print("\nNote: the following manual drop candidates were NOT present in the dataset and will be ignored:")
        print(missing_manual)

# Consolidate drops
cols_to_drop.extend(manual_drop_existing)
cols_to_drop = list(set(cols_to_drop))  # remove duplicates

print(f"\nDropping {len(cols_to_drop)} columns (those >50% missing plus selected manual candidates).")
print(cols_to_drop)

# Perform drop safely
df_cleaned = df.drop(columns=cols_to_drop, errors='ignore')

print(f"\nRemaining columns: {df_cleaned.shape[1]}")
print(f"\nRemaining columns list:\n{df_cleaned.columns.tolist()}")

# Optionally save cleaned file
# df_cleaned.to_csv('survey_results_slim_cleaned.csv', index=False)
# ...existing code...