"""
Feature engineering pipeline for wildfire prediction.
Creates derived features from weather and environmental data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
import pickle


PROCESSED_DIR = Path("data/processed")
FEATURES_DIR = Path("data/features")
FEATURES_DIR.mkdir(exist_ok=True, parents=True)


def create_temporal_features(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """Create temporal features from date column."""
    df = df.copy()
    
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Cyclical encoding for time features
        df['month_sin'] = np.sin(2 * np.pi * df[date_col].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df[date_col].dt.month / 12)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df[date_col].dt.dayofyear / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df[date_col].dt.dayofyear / 365)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df[date_col].dt.dayofweek / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df[date_col].dt.dayofweek / 7)
        
        # Season indicator
        df['season'] = df[date_col].dt.month % 12 // 3 + 1  # 1=Winter, 2=Spring, 3=Summer, 4=Fall
        df['is_summer'] = (df['season'] == 3).astype(int)
    
    return df


def create_weather_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction and derived weather features."""
    df = df.copy()
    
    # Temperature range (indicator of daily variability)
    if 'temp_max' in df.columns and 'temp_min' in df.columns:
        df['temp_range'] = df['temp_max'] - df['temp_min']
    
    # Heat index approximation (temperature + humidity interaction)
    if 'temp_avg' in df.columns and 'relative_humidity' in df.columns:
        df['heat_humidity_index'] = df['temp_avg'] * (1 - df['relative_humidity'] / 100)
    
    # Dryness index (high temp + low humidity)
    if 'temp_max' in df.columns and 'relative_humidity' in df.columns:
        df['dryness_index'] = df['temp_max'] / (df['relative_humidity'] + 1)
    
    # Wind-driven fire spread potential
    if 'wind_speed' in df.columns and 'relative_humidity' in df.columns:
        df['wind_fire_potential'] = df['wind_speed'] * (100 - df['relative_humidity']) / 100
    
    # Fuel moisture approximation (inverse of dryness)
    if 'precipitation' in df.columns and 'relative_humidity' in df.columns:
        df['fuel_moisture'] = (df['precipitation'] + 1) * df['relative_humidity'] / 100
    
    return df


def create_rolling_features(df: pd.DataFrame, date_col: str = 'date', windows=[7, 14, 30]) -> pd.DataFrame:
    """Create rolling window statistics for weather variables."""
    df = df.copy()
    
    if date_col not in df.columns:
        return df
    
    # Sort by date
    df = df.sort_values(date_col)
    
    # Weather columns to create rolling features for
    weather_cols = ['temp_avg', 'temp_max', 'relative_humidity', 'wind_speed', 'precipitation']
    available_cols = [col for col in weather_cols if col in df.columns]
    
    for window in windows:
        for col in available_cols:
            df[f'{col}_roll_mean_{window}d'] = df[col].rolling(window=window, min_periods=1).mean()
            df[f'{col}_roll_std_{window}d'] = df[col].rolling(window=window, min_periods=1).std()
    
    # Cumulative precipitation (last 30 days)
    if 'precipitation' in df.columns:
        df['precip_cumsum_30d'] = df['precipitation'].rolling(window=30, min_periods=1).sum()
    
    return df


def engineer_features(input_file: str, output_file: str = None):
    """
    Main feature engineering pipeline.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (optional)
    """
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"  Initial shape: {df.shape}")
    
    # Apply feature engineering steps
    print("Creating temporal features...")
    df = create_temporal_features(df)
    
    print("Creating weather interaction features...")
    df = create_weather_interaction_features(df)
    
    print("Creating rolling window features...")
    df = create_rolling_features(df)
    
    # Handle missing values from rolling calculations
    df = df.fillna(method='bfill').fillna(0)
    
    print(f"  Final shape: {df.shape}")
    print(f"  New features created: {df.shape[1] - pd.read_csv(input_file).shape[1]}")
    
    # Save
    if output_file is None:
        output_file = FEATURES_DIR / f"engineered_{Path(input_file).name}"
    else:
        output_file = Path(output_file)
    
    df.to_csv(output_file, index=False)
    print(f"✓ Saved engineered features to {output_file}")
    
    return df


def main():
    """Engineer features for all datasets."""
    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
    print()
    
    # Process synthetic dataset
    if (PROCESSED_DIR / "synthetic_wildfire_data.csv").exists():
        engineer_features(
            PROCESSED_DIR / "synthetic_wildfire_data.csv",
            FEATURES_DIR / "engineered_synthetic_data.csv"
        )
        print()
    
    # Process UCI dataset
    if (PROCESSED_DIR / "uci_forest_fires.csv").exists():
        engineer_features(
            PROCESSED_DIR / "uci_forest_fires.csv",
            FEATURES_DIR / "engineered_uci_data.csv"
        )
        print()
    
    print("=" * 60)
    print("Feature engineering complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
