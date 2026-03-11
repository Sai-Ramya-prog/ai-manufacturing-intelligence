import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR        = Path(__file__).resolve().parent.parent
PROCESS_PATH    = BASE_DIR / "data" / "_h_batch_process_data.xlsx"
PRODUCTION_PATH = BASE_DIR / "data" / "_h_batch_production_data.xlsx"
EMISSION_FACTOR = 0.82  # India grid kg CO2 per kWh


def compute_energy_from_process(process_df):
    """Real energy + asset health features from process sensor data."""
    mean_power        = process_df['Power_Consumption_kW'].mean()
    total_energy_kwh  = process_df['Power_Consumption_kW'].sum() / 60
    peak_power_kw     = process_df['Power_Consumption_kW'].max()
    power_variability = process_df['Power_Consumption_kW'].std()
    avg_vibration     = process_df['Vibration_mm_s'].mean()
    asset_health      = max(0, 100
                            - (power_variability / (mean_power + 1e-8)) * 50
                            - (avg_vibration / 10) * 50)
    return {
        'process_energy_kwh':  round(total_energy_kwh,  3),
        'peak_power_kw':       round(peak_power_kw,     3),
        'power_variability':   round(power_variability, 3),
        'avg_vibration':       round(avg_vibration,     3),
        'asset_health_score':  round(asset_health,      3),
    }


def load_and_prepare():
    process    = pd.read_excel(PROCESS_PATH)
    production = pd.read_excel(PRODUCTION_PATH)

    # Add real process energy features to all production batches
    energy_feats = compute_energy_from_process(process)
    for col, val in energy_feats.items():
        production[col] = val

    return production


def get_feature_target_split(df):
    """
    16 real features → 4 targets.
    CO2 is NOT a target (it's a constant here — derived post-prediction in dashboard).
    """
    controllable = [
        'Granulation_Time', 'Binder_Amount', 'Drying_Temp', 'Drying_Time',
        'Compression_Force', 'Machine_Speed', 'Lubricant_Conc', 'Moisture_Content'
    ]
    tablet = ['Tablet_Weight', 'Hardness', 'Disintegration_Time']
    energy = [
        'process_energy_kwh', 'peak_power_kw',
        'power_variability', 'avg_vibration', 'asset_health_score'
    ]

    feature_cols = controllable + tablet + energy

    # 4 real varying targets from production data
    target_cols = [
        'Content_Uniformity',  # Quality
        'Dissolution_Rate',    # Yield
        'Friability',          # Performance
        'Disintegration_Time', # Process efficiency
    ]

    feature_cols = [c for c in feature_cols if c in df.columns]
    target_cols  = [c for c in target_cols  if c in df.columns]

    return df[feature_cols], df[target_cols], feature_cols, target_cols


if __name__ == "__main__":
    df = load_and_prepare()
    X, y, fc, tc = get_feature_target_split(df)
    print("Features :", X.shape)
    print("Targets  :", y.shape)
    print("Cols     :", fc)
    print("Targets  :", tc)
    print("\n", y.describe())