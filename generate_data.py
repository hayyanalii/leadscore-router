"""
generate_data.py
Creates fake call center data for training our lead scoring model.
Run this once. It creates 'call_data.csv' with 10,000 records.
"""

import pandas as pd
import numpy as np

# Set random seed so results are reproducible
np.random.seed(42)

N = 10000  # number of calls

print(f"Generating {N} fake call records...")

# --- Simulate realistic call center features ---

# Call duration: most calls are 2-10 min, some long ones
call_duration_sec = np.random.gamma(shape=2.0, scale=120, size=N).astype(int) + 30

# Time of day: 0=morning, 1=afternoon, 2=evening
time_of_day = np.random.choice([0, 1, 2], size=N, p=[0.4, 0.4, 0.2])

# IVR option selected: 0=sales, 1=support, 2=billing
ivr_option = np.random.choice([0, 1, 2], size=N, p=[0.5, 0.3, 0.2])

# Agent experience in years
agent_experience_yrs = np.random.randint(0, 15, size=N)

# Number of previous calls from this customer
previous_calls = np.random.poisson(lam=1.2, size=N)

# Customer waited in queue (seconds)
wait_time_sec = np.random.exponential(scale=45, size=N).astype(int)

# --- Create 'converted' target with realistic logic ---
# This is what we want the AI to LEARN.
# More experienced agents + sales IVR + shorter wait = higher chance of converting.

base_score = (
    0.30                                    # baseline
    + 0.02 * agent_experience_yrs           # experience helps
    + 0.15 * (ivr_option == 0)              # sales calls convert more
    - 0.0008 * wait_time_sec                # long waits hurt
    + 0.05 * (time_of_day == 0)             # mornings slightly better
    + 0.03 * previous_calls                 # loyal customers convert
)

# Add noise so it's not perfectly predictable
prob_converted = np.clip(base_score + np.random.normal(0, 0.1, N), 0.02, 0.95)

# Roll the dice to determine actual conversion
converted = (np.random.random(N) < prob_converted).astype(int)

# --- Build the DataFrame ---
df = pd.DataFrame({
    "call_duration_sec": call_duration_sec,
    "time_of_day": time_of_day,
    "ivr_option": ivr_option,
    "agent_experience_yrs": agent_experience_yrs,
    "previous_calls": previous_calls,
    "wait_time_sec": wait_time_sec,
    "converted": converted,
})

# Save to CSV
df.to_csv("call_data.csv", index=False)

print(f"✅ Done! Created 'call_data.csv' with {len(df)} rows.")
print(f"✅ Conversion rate: {df['converted'].mean() * 100:.1f}%")
print(f"✅ File size: {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())