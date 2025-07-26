import pandas as pd
import numpy as np
import os
from linearmodels.panel import PanelOLS
import matplotlib.pyplot as plt
import seaborn as sns

# Set your folder path here
folder_path = r"C:\Users\ASUS\Downloads\Apple-20250726T090535Z-1-001\Apple"

# STEP 0: Show actual column names in the first file
for file in os.listdir(folder_path):
    if file.endswith(".xlsx") or file.endswith(".xls"):
        file_path = os.path.join(folder_path, file)
        df = pd.read_excel(file_path)
        print(f"\nInspecting file: {file}")
        print("Original columns:", df.columns.tolist())
        break

# STEP 1: Read and clean all Excel files
df_list = []

for file in os.listdir(folder_path):
    if file.endswith(".xlsx") or file.endswith(".xls"):
        file_path = os.path.join(folder_path, file)
        df = pd.read_excel(file_path)

        print(f"\nReading file: {file}")
        print("Original columns:", df.columns.tolist())

        # Clean column names
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace(".", "")
            .str.replace("(", "")
            .str.replace(")", "")
        )

        # Rename based on actual names
        df = df.rename(columns={
            'state_name': 'state',
            'district_name': 'district',
            'market_name': 'market',
            'arrivals_tonnes': 'arrivals',
            'modal_price_rs/quintal': 'modal_price'
        })

        required_cols = ['state', 'district', 'market', 'date', 'modal_price', 'arrivals']
        if all(col in df.columns for col in required_cols):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['modal_price', 'arrivals', 'date'])

            df['commodity'] = 'Apple'
            df['year'] = df['date'].dt.year

            df_list.append(df)
        else:
            print(f"Skipping file due to missing columns: {file}")

# STEP 2: Combine all files
if not df_list:
    raise ValueError("No valid Excel files to concatenate. Check column names.")

apple_df = pd.concat(df_list, ignore_index=True)

# STEP 3: Unique counts
print("\nUnique States:", apple_df['state'].nunique())
print("Unique Districts:", apple_df['district'].nunique())
print("Unique Markets:", apple_df['market'].nunique())

# STEP 4: Create market_id and panel structure
apple_df['market_id'] = apple_df['state'] + '_' + apple_df['district'] + '_' + apple_df['market']
apple_df = apple_df.sort_values(['market_id', 'date']).reset_index(drop=True)
apple_df = apple_df.set_index(['market_id', 'date'])

# STEP 5: Log transformations and filtering
apple_df['log_modal_price'] = np.log(apple_df['modal_price'])
apple_df['log_arrivals'] = np.log(apple_df['arrivals'])
apple_df['year'] = apple_df.index.get_level_values('date').year

# 🔥 Remove rows with infs or NaNs caused by log(0) or missing values
apple_df = apple_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['log_modal_price', 'log_arrivals'])

# STEP 6: Run panel regression
model = PanelOLS.from_formula('log_modal_price ~ log_arrivals + EntityEffects + C(year)', data=apple_df)
results = model.fit()
print(results.summary)

# STEP 7: Visualizations

# a) Average modal price by year
yearly_avg = apple_df.reset_index().groupby('year')['modal_price'].mean()
yearly_avg.plot(marker='o', title='Average Modal Price by Year')
plt.ylabel('Modal Price')
plt.show()

# b) State-level heatmap
state_year = apple_df.reset_index().groupby(['state', 'year'])['modal_price'].mean().unstack()
plt.figure(figsize=(12, 6))
sns.heatmap(state_year, cmap='YlGnBu')
plt.title('State-Level Average Modal Price (Heatmap)')
plt.show()

# c) Price volatility by market
volatility = apple_df.groupby('market_id')['modal_price'].std().sort_values()
print("\nTop 5 Stable Markets (Least Volatility):")
print(volatility.head())

print("\nTop 5 Volatile Markets:")
print(volatility.tail())

# d) Anomalies: price spikes
apple_df_reset = apple_df.reset_index()
apple_df_reset['z_score'] = (apple_df_reset['modal_price'] - apple_df_reset['modal_price'].mean()) / apple_df_reset['modal_price'].std()
anomalies = apple_df_reset[apple_df_reset['z_score'].abs() > 3]

print("\nAnomalous Price Events:")
print(anomalies[['market_id', 'date', 'modal_price', 'z_score']])


