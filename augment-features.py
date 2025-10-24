######################   ADDING ENGINEERED FEATURES TO ALIGNED DATA . CSV  ####################

import pandas as pd
import numpy as np
from scipy.stats import rankdata, percentileofscore

df = pd.read_csv("misreport_data_test/aligned_dataset.csv")
load_cols = [col for col in df.columns if col.startswith("Load_S")]
label_cols = [col for col in df.columns if col.startswith("Label_S")]

# Phase 1: Basic features
for col in load_cols:
    df[f"{col}_delta"] = df[col].diff().fillna(0)
    df[f"{col}_rollmean"] = df[col].rolling(window=10, min_periods=1).mean()
    df[f"{col}_percentile"] = df[col].expanding().apply(lambda x: rankdata(x)[-1] / len(x))

# Phase 2: Behavioral features
window = 100
for i, col in enumerate(load_cols):
    df[f"{col}_rolling_percentile"] = df[col].rolling(window=window, min_periods=10).apply(
        lambda x: percentileofscore(x, x.iloc[-1]) / 100.0
    )
    df[f"{col}_std_recent"] = df[col].rolling(window=window, min_periods=10).std().fillna(0)
    others = [c for j, c in enumerate(load_cols) if j != i]
    df[f"{col}_load_ratio"] = df[col] / df[others].mean(axis=1)

df.fillna(0).to_csv("misreport_data_test/augmented_dataset.csv", index=False)
print("Augmented dataset saved.")




###################### ADDING NEW FEATURES  #######################


from scipy.stats import zscore, skew, kurtosis
from statsmodels.tsa.stattools import acf
from tqdm import tqdm

# Load the augmented dataset
df = pd.read_csv("misreport_data_test/augmented_dataset.csv")

# Extract switch IDs dynamically
switch_ids = sorted(set(col.split("_")[1] for col in df.columns if col.startswith("Load_") and col.split("_")[1].startswith("S")))

# Function to compute new features for each switch
def compute_features(df, switch_ids, window=10):
    for sw in switch_ids:
        base_col = f"Load_{sw}"
        if base_col not in df.columns:
            continue

        values = df[base_col].values

        # Initialize lists to store computed features
        delta_mean = []
        mad = []
        unique_count = []
        autocorr = []
        skewness = []
        kurt = []
        z_scores = []

        for i in range(len(values)):
            if i < window:
                delta_mean.append(np.nan)
                mad.append(np.nan)
                unique_count.append(np.nan)
                autocorr.append(np.nan)
                skewness.append(np.nan)
                kurt.append(np.nan)
                z_scores.append(np.nan)
                continue

            window_data = values[i-window:i]
            delta_mean.append(np.mean(np.diff(window_data)))
            mad.append(np.median(np.abs(window_data - np.median(window_data))))
            unique_count.append(len(set(window_data)))
            autocorr.append(acf(window_data, nlags=1, fft=False)[1] if len(set(window_data)) > 1 else 0)
            skewness.append(skew(window_data))
            kurt.append(kurtosis(window_data))
            z_scores.append(zscore(window_data)[-1])

        # Append features to the dataframe
        df[f"{base_col}_delta_mean"] = delta_mean
        df[f"{base_col}_mad"] = mad
        df[f"{base_col}_unique_count"] = unique_count
        df[f"{base_col}_autocorr"] = autocorr
        df[f"{base_col}_skew"] = skewness
        df[f"{base_col}_kurtosis"] = kurt
        df[f"{base_col}_zscore"] = z_scores

    return df

# Apply the computation
augmented_df = compute_features(df.copy(), switch_ids)

# === Fill NaNs from rolling window effect ===
augmented_df.ffill(inplace=True)
augmented_df.bfill(inplace=True)

# Save to new file
output_path = "misreport_data_test//augmented_dataset_with_behavioral.csv"
augmented_df.to_csv(output_path, index=False)
