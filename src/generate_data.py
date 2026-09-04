import numpy as np
import pandas as pd

np.random.seed(42)

# --------------------------------------------------
# 1. Create hourly timestamps
# --------------------------------------------------

timestamps = pd.date_range(
    start="2026-03-01",
    end="2026-09-01",
    freq="h",
    inclusive="left"
)

df = pd.DataFrame({"timestamp": timestamps})

# --------------------------------------------------
# 2. Calendar features
# --------------------------------------------------

df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["day_of_year"] = df["timestamp"].dt.dayofyear
df["month"] = df["timestamp"].dt.month

# --------------------------------------------------
# 3. Simulate Delhi-like temperature
# --------------------------------------------------

seasonal_temp = (
    27
    + 9 * np.sin(2 * np.pi * (df["day_of_year"] - 80) / 365)
)

daily_temp = (
    4 * np.sin(2 * np.pi * (df["hour"] - 7) / 24)
)

noise_temp = np.random.normal(0, 1.2, len(df))

df["temperature"] = (
    seasonal_temp
    + daily_temp
    + noise_temp
)

# Keep temperature realistic
df["temperature"] = df["temperature"].clip(15, 45)

# --------------------------------------------------
# 4. Simulate humidity
# --------------------------------------------------

df["humidity"] = (
    75
    - 0.9 * (df["temperature"] - 25)
    + np.random.normal(0, 5, len(df))
)

df["humidity"] = df["humidity"].clip(20, 95)

# --------------------------------------------------
# 5. Electricity demand
# --------------------------------------------------

# Base demand
demand = np.full(len(df), 5200.0)

# Daily demand pattern
morning_peak = 700 * np.exp(
    -((df["hour"] - 9) / 3) ** 2
)

evening_peak = 1100 * np.exp(
    -((df["hour"] - 19) / 4) ** 2
)

night_reduction = np.where(
    (df["hour"] >= 0) & (df["hour"] <= 5),
    -700,
    0
)

# Temperature effect
temperature_effect = (
    120 * np.maximum(df["temperature"] - 28, 0)
)

# Weekly pattern
weekend_effect = np.where(
    df["day_of_week"] >= 5,
    -350,
    0
)

# Seasonal effect
seasonal_effect = (
    500 * np.sin(
        2 * np.pi * (df["day_of_year"] - 100) / 365
    )
)

# Random demand noise
noise = np.random.normal(0, 120, len(df))

df["demand_mw"] = (
    demand
    + morning_peak
    + evening_peak
    + night_reduction
    + temperature_effect
    + weekend_effect
    + seasonal_effect
    + noise
)

# --------------------------------------------------
# 6. Add a few unusual demand events
# --------------------------------------------------

event_indices = np.random.choice(
    len(df),
    size=20,
    replace=False
)

df.loc[event_indices, "demand_mw"] += np.random.uniform(
    400,
    900,
    size=len(event_indices)
)

# --------------------------------------------------
# 7. Clean up
# --------------------------------------------------

df["demand_mw"] = df["demand_mw"].round(2)
df["temperature"] = df["temperature"].round(2)
df["humidity"] = df["humidity"].round(2)

# Keep only useful columns
df = df[
    [
        "timestamp",
        "demand_mw",
        "temperature",
        "humidity"
    ]
]

# --------------------------------------------------
# 8. Save
# --------------------------------------------------

output_path = "data/demand_weather.csv"

df.to_csv(output_path, index=False)

print(f"Dataset created: {output_path}")
print(f"Rows: {len(df):,}")
print()
print(df.head())
print()
print("Demand statistics:")
print(df["demand_mw"].describe())
