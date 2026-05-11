import wandb
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid as trap
import matplotlib.pyplot as plt

# --- Pulling run data from wandb ---

# Accessing API
api = wandb.Api()

# Pulling data for specified project and group
good_runs = api.runs("lipscomb-robotics/PegInsert_results", {"group":"ours"})
baseline_runs = api.runs("lipscomb-robotics/PegInsert_results", {"group":"theirs"})
# Specifying data to consider - wall_time instead of _step is likely what we need for this
good_hist = [r.history(keys=["_step", "success_rate", "_runtime"]) for r in good_runs]
#good_hist["replay_buffer_type"] = r.config.get("replay_buffer_type")

baseline_hist = [r.history(keys=["_step", "success_rate", "_runtime"]) for r in baseline_runs]

# Labeling data by run number
for i,df in enumerate(good_hist): df["run"]=i
for i,df in enumerate(baseline_hist): df["run"]=i

# Building dataframe of all runs
good_df = pd.concat(good_hist)
baseline_df = pd.concat(baseline_hist)

# Merging all runs
good_time_sorted = good_df.sort_values("_step").reset_index(drop=True)
baseline_time_sorted = baseline_df.sort_values("_step").reset_index(drop=True)

# Smoothing runs
good_time_avg = good_df.groupby('_step', as_index=False)['success_rate'].mean()
baseline_time_avg = baseline_df.groupby('_step', as_index=False)['success_rate'].mean()
good_time_avg['group']='ours'
baseline_time_avg['group']='theirs'

# Checking work
good_time_avg.plot(x='_step', y='success_rate')
plt.savefig('Ours - PegInsert Results')
baseline_time_avg.plot(x='_step', y='success_rate')
plt.savefig('Baseline - PegInsert Results')

# --- Calculating integrals from run data ---

# Combining run dfs
combined_df = pd.concat([good_time_avg, baseline_time_avg])

# Creating blank dictionaries to group run data and integrals
combined_rundata_arrays = {}
combined_integrals = {}
combined_runlengths = {}
combined_maxrewards = {}
combined_normalizingfactor = {}
combined_integrals_normalized = {}

# Computing run integrals

for run_label, group in combined_df.groupby("group"):
    x = group["_step"].to_numpy()      # Uncomment to run by steps. Comment to run by time
    #x = group["_runtime"].to_numpy()    # Comment line to run by steps. Uncomment to run by time
    y = group["success_rate"].to_numpy()
    combined_rundata_arrays[run_label] = (x,y)
    combined_integrals[run_label] = trap(y,x)
    combined_runlengths[run_label] = float(x[-1])
    combined_maxrewards[run_label] = float(max(y))
    combined_normalizingfactor[run_label] = combined_runlengths[run_label] * combined_maxrewards[run_label]
    combined_integrals_normalized[run_label] = float(combined_integrals[run_label] / combined_normalizingfactor[run_label])

pd.DataFrame(
    combined_integrals_normalized.items(),
    columns=["group", "normalized_integral"]
).to_csv(
    "PegInsert Results - Normalized Integrals.csv",
    index=False
)