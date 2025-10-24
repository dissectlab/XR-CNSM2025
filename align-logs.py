import pandas as pd

# === Step 1: Load Raw Logs ===
misreport_log = pd.read_csv("misreport_log1.csv")
record_log = pd.read_csv("record_log1.csv")
normal_log2 = pd.read_csv("normal_log2.csv")
normal_log3 = pd.read_csv("normal_log3.csv")
normal_log4 = pd.read_csv("normal_log4.csv")

# === Step 2: Prepare Switch 1 Data (Attacker) ===
# Use LOAD_BYTES and REAL/FAKE label
record_df = record_log[["LOAD_BYTES", "REAL/FAKE"]].rename(columns={"LOAD_BYTES": "Load_S1", "REAL/FAKE": "Label_S1"})
misreport_df = misreport_log[["LOAD_BYTES", "REAL/FAKE"]].rename(columns={"LOAD_BYTES": "Load_S1", "REAL/FAKE": "Label_S1"})

# Concatenate reconnaissance and attack phases
switch1_df = pd.concat([record_df, misreport_df], ignore_index=True)

# Convert 'FAKE' to 1 and 'REAL' to 0
switch1_df["Label_S1"] = (switch1_df["Label_S1"] == "FAKE").astype(int)

# === Step 3: Prepare Other Switches ===
switch2_df = normal_log2[["LOAD_BYTES"]].rename(columns={"LOAD_BYTES": "Load_S2"})
switch3_df = normal_log3[["LOAD_BYTES"]].rename(columns={"LOAD_BYTES": "Load_S3"})
switch4_df = normal_log4[["LOAD_BYTES"]].rename(columns={"LOAD_BYTES": "Load_S4"})

# Add label columns with all zeros (they never fake)
switch2_df["Label_S2"] = 0
switch3_df["Label_S3"] = 0
switch4_df["Label_S4"] = 0

# === Step 4: Concatenate All Switches Per Epoch ===
aligned_df = pd.concat([switch1_df, switch2_df, switch3_df, switch4_df], axis=1)

# === Step 5: Save to CSV ===
aligned_df.to_csv("aligned_dataset.csv", index=False)
print("Aligned dataset saved to 'aligned_dataset.csv'")
