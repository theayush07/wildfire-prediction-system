"""
Data ingestion script for wildfire prediction project.
Downloads and prepares raw data into data/processed/ folder.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import requests
from io import StringIO
import zipfile
from tqdm import tqdm

# Create data directories
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

for dir in [DATA_DIR, RAW_DIR, PROCESSED_DIR]:
    dir.mkdir(exist_ok=True)


def download_file(url: str, destination: Path, description: str = "Downloading"):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=description,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)


def load_uci_forest_fires():
    """
    Load UCI Forest Fires dataset.
    Source: https://archive.ics.uci.edu/ml/datasets/Forest+Fires
    """
    print("Loading UCI Forest Fires dataset...")
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv"
    
    try:
        df = pd.read_csv(url)
        
        # Basic preprocessing
        # Convert month and day to numeric
        month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                     'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
        day_map = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7}
        
        df['month_numeric'] = df['month'].map(month_map)
        df['day_numeric'] = df['day'].map(day_map)
        
        # Create binary fire occurrence label (area > 0 means fire occurred)
        df['fire_occurred'] = (df['area'] > 0).astype(int)
        
        # Save to processed
        output_path = PROCESSED_DIR / "uci_forest_fires.csv"
        df.to_csv(output_path, index=False)
        print(f"✓ Saved to {output_path}")
        print(f"  Shape: {df.shape}")
        print(f"  Fire occurrences: {df['fire_occurred'].sum()} / {len(df)}")
        
        return df
        
    except Exception as e:
        print(f"✗ Error loading UCI dataset: {e}")
        return None


def create_synthetic_weather_fire_data(n_samples: int = 5000):
    """
    Create synthetic weather and fire occurrence data for demonstration.
    This simulates realistic wildfire patterns based on weather conditions.
    """
    print(f"Creating synthetic wildfire dataset with {n_samples} samples...")
    
    np.random.seed(42)
    
    # Generate features
    data = {
        'date': pd.date_range(start='2018-01-01', periods=n_samples, freq='D'),
        'latitude': np.random.uniform(35, 50, n_samples),  # Northern hemisphere
        'longitude': np.random.uniform(-120, -100, n_samples),
        'temp_max': np.random.uniform(5, 45, n_samples),  # Celsius
        'temp_min': np.random.uniform(0, 30, n_samples),
        'temp_avg': np.random.uniform(10, 35, n_samples),
        'relative_humidity': np.random.uniform(10, 95, n_samples),  # Percentage
        'wind_speed': np.random.uniform(0, 30, n_samples),  # km/h
        'precipitation': np.random.exponential(2, n_samples),  # mm
        'ndvi': np.random.uniform(0.1, 0.9, n_samples),  # Vegetation index
        'elevation': np.random.uniform(100, 2500, n_samples),  # meters
        'distance_to_road': np.random.exponential(5, n_samples),  # km
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic fire occurrence based on weather conditions
    # High risk: high temp, low humidity, high wind, low precipitation
    fire_risk = (
        (df['temp_max'] - 20) / 25 * 0.3 +  # Higher temp increases risk
        (100 - df['relative_humidity']) / 100 * 0.3 +  # Lower humidity increases risk
        df['wind_speed'] / 30 * 0.2 +  # Higher wind increases risk
        (5 - df['precipitation'].clip(0, 5)) / 5 * 0.15 +  # Less rain increases risk
        (df['ndvi'] - 0.3).clip(0, 1) / 0.7 * 0.05  # More vegetation slightly increases risk
    )
    
    # Add seasonal effect (summer has higher risk)
    df['day_of_year'] = df['date'].dt.dayofyear
    seasonal_factor = np.sin((df['day_of_year'] - 80) * 2 * np.pi / 365) * 0.3 + 0.7
    fire_risk = fire_risk * seasonal_factor
    
    # Convert to binary with some randomness
    fire_probability = fire_risk.clip(0, 1) ** 2  # Square to make fires rarer
    df['fire_occurred'] = (np.random.random(n_samples) < fire_probability).astype(int)
    
    # Add burned area for fires (log-normal distribution)
    df['burned_area'] = 0.0
    fire_mask = df['fire_occurred'] == 1
    df.loc[fire_mask, 'burned_area'] = np.random.lognormal(2, 1.5, fire_mask.sum())
    
    # Extract temporal features
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['year'] = df['date'].dt.year
    
    # Save
    output_path = PROCESSED_DIR / "synthetic_wildfire_data.csv"
    df.to_csv(output_path, index=False)
    print(f"✓ Saved to {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Fire occurrences: {df['fire_occurred'].sum()} / {len(df)} ({df['fire_occurred'].mean()*100:.1f}%)")
    
    return df


def main():
    """Main data ingestion pipeline."""
    print("=" * 60)
    print("WILDFIRE PREDICTION - DATA INGESTION")
    print("=" * 60)
    print()
    
    # Load UCI Forest Fires dataset
    uci_df = load_uci_forest_fires()
    print()
    
    # Create synthetic dataset for more comprehensive training
    synthetic_df = create_synthetic_weather_fire_data(n_samples=5000)
    print()
    
    print("=" * 60)
    print("Data ingestion complete!")
    print(f"Processed data saved to: {PROCESSED_DIR.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
