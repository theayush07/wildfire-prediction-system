"""
Exploratory Data Analysis for Wildfire Prediction
This script can be run directly or converted to a Jupyter notebook.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/eda")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


def load_data():
    """Load the processed datasets."""
    print("Loading datasets...")
    
    datasets = {}
    
    # Load synthetic data
    synthetic_path = PROCESSED_DIR / "synthetic_wildfire_data.csv"
    if synthetic_path.exists():
        datasets['synthetic'] = pd.read_csv(synthetic_path)
        print(f"  Loaded synthetic data: {datasets['synthetic'].shape}")
    
    # Load UCI data
    uci_path = PROCESSED_DIR / "uci_forest_fires.csv"
    if uci_path.exists():
        datasets['uci'] = pd.read_csv(uci_path)
        print(f"  Loaded UCI data: {datasets['uci'].shape}")
    
    return datasets


def basic_statistics(df, name="Dataset"):
    """Print basic statistics about the dataset."""
    print(f"\n{'='*60}")
    print(f"{name} - Basic Statistics")
    print(f"{'='*60}")
    
    print(f"\nShape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\nTarget Distribution:")
    if 'fire_occurred' in df.columns:
        fire_counts = df['fire_occurred'].value_counts()
        print(f"  No Fire (0): {fire_counts.get(0, 0)} ({fire_counts.get(0, 0)/len(df)*100:.1f}%)")
        print(f"  Fire (1): {fire_counts.get(1, 0)} ({fire_counts.get(1, 0)/len(df)*100:.1f}%)")
    
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("  No missing values")
    
    print(f"\nData Types:")
    print(df.dtypes.value_counts())
    
    return df.describe()


def plot_target_distribution(df, name="Dataset"):
    """Plot the distribution of fire occurrences."""
    if 'fire_occurred' not in df.columns:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Count plot
    fire_counts = df['fire_occurred'].value_counts()
    axes[0].bar(['No Fire', 'Fire'], fire_counts.values, color=['steelblue', 'orangered'])
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'{name} - Fire Occurrence Distribution')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add percentages
    for i, v in enumerate(fire_counts.values):
        axes[0].text(i, v + max(fire_counts.values)*0.02, f'{v}\n({v/len(df)*100:.1f}%)', 
                    ha='center', va='bottom')
    
    # Pie chart
    axes[1].pie(fire_counts.values, labels=['No Fire', 'Fire'], autopct='%1.1f%%',
                colors=['steelblue', 'orangered'], startangle=90)
    axes[1].set_title(f'{name} - Fire Occurrence Proportion')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{name}_target_distribution.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {name}_target_distribution.png")
    plt.close()


def plot_weather_distributions(df, name="Dataset"):
    """Plot distributions of weather variables."""
    weather_vars = ['temp_max', 'temp_min', 'temp_avg', 'relative_humidity', 
                    'wind_speed', 'precipitation']
    available_vars = [var for var in weather_vars if var in df.columns]
    
    if not available_vars:
        return
    
    n_vars = len(available_vars)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, var in enumerate(available_vars):
        axes[idx].hist(df[var].dropna(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[idx].set_xlabel(var.replace('_', ' ').title())
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'{var.replace("_", " ").title()} Distribution')
        axes[idx].grid(axis='y', alpha=0.3)
    
    # Hide extra subplots
    for idx in range(len(available_vars), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{name}_weather_distributions.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {name}_weather_distributions.png")
    plt.close()


def plot_correlation_matrix(df, name="Dataset"):
    """Plot correlation matrix for numeric features."""
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove date-related and identifier columns
    exclude_cols = ['year', 'burned_area', 'X', 'Y']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if len(numeric_cols) < 2:
        return
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title(f'{name} - Feature Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{name}_correlation_matrix.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {name}_correlation_matrix.png")
    plt.close()


def plot_weather_by_fire(df, name="Dataset"):
    """Compare weather conditions for fire vs no-fire events."""
    if 'fire_occurred' not in df.columns:
        return
    
    weather_vars = ['temp_max', 'relative_humidity', 'wind_speed', 'precipitation']
    available_vars = [var for var in weather_vars if var in df.columns]
    
    if not available_vars:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, var in enumerate(available_vars):
        if idx >= len(axes):
            break
        
        fire_data = df[df['fire_occurred'] == 1][var].dropna()
        no_fire_data = df[df['fire_occurred'] == 0][var].dropna()
        
        axes[idx].hist([no_fire_data, fire_data], bins=30, label=['No Fire', 'Fire'],
                      color=['steelblue', 'orangered'], alpha=0.7, edgecolor='black')
        axes[idx].set_xlabel(var.replace('_', ' ').title())
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'{var.replace("_", " ").title()} by Fire Occurrence')
        axes[idx].legend()
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{name}_weather_by_fire.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {name}_weather_by_fire.png")
    plt.close()


def plot_temporal_patterns(df, name="Dataset"):
    """Analyze temporal patterns in fire occurrence."""
    if 'fire_occurred' not in df.columns:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Monthly pattern
    if 'month' in df.columns:
        monthly = df.groupby('month')['fire_occurred'].agg(['sum', 'count', 'mean'])
        axes[0, 0].bar(monthly.index, monthly['sum'], color='orangered', alpha=0.7)
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Number of Fires')
        axes[0, 0].set_title('Fires by Month')
        axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Day of week pattern
    if 'day_of_week' in df.columns:
        dow = df.groupby('day_of_week')['fire_occurred'].agg(['sum', 'count', 'mean'])
        dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[0, 1].bar(range(len(dow)), dow['sum'], color='orangered', alpha=0.7)
        axes[0, 1].set_xticks(range(len(dow)))
        axes[0, 1].set_xticklabels(dow_labels)
        axes[0, 1].set_xlabel('Day of Week')
        axes[0, 1].set_ylabel('Number of Fires')
        axes[0, 1].set_title('Fires by Day of Week')
        axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Fire rate by month
    if 'month' in df.columns:
        axes[1, 0].plot(monthly.index, monthly['mean'] * 100, marker='o', 
                       color='orangered', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Fire Occurrence Rate (%)')
        axes[1, 0].set_title('Fire Rate by Month')
        axes[1, 0].grid(alpha=0.3)
    
    # Day of year pattern (if date available)
    if 'day_of_year' in df.columns:
        doy = df.groupby('day_of_year')['fire_occurred'].mean()
        axes[1, 1].plot(doy.index, doy.values * 100, color='orangered', alpha=0.7)
        axes[1, 1].set_xlabel('Day of Year')
        axes[1, 1].set_ylabel('Fire Occurrence Rate (%)')
        axes[1, 1].set_title('Fire Rate Throughout Year')
        axes[1, 1].grid(alpha=0.3)
    else:
        axes[1, 1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{name}_temporal_patterns.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {name}_temporal_patterns.png")
    plt.close()


def plot_spatial_distribution(df, name="Dataset"):
    """Plot spatial distribution of fires (if lat/lon available)."""
    if not all(col in df.columns for col in ['latitude', 'longitude', 'fire_occurred']):
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # All points
    fires = df[df['fire_occurred'] == 1]
    no_fires = df[df['fire_occurred'] == 0]
    
    axes[0].scatter(no_fires['longitude'], no_fires['latitude'], 
                   c='steelblue', alpha=0.3, s=10, label='No Fire')
    axes[0].scatter(fires['longitude'], fires['latitude'], 
                   c='orangered', alpha=0.6, s=20, label='Fire')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].set_title('Spatial Distribution of Fire Events')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Density heatmap
    from scipy.stats import gaussian_kde
    if len(fires) > 10:
        xy = np.vstack([fires['longitude'], fires['latitude']])
        z = gaussian_kde(xy)(xy)
        
        scatter = axes[1].scatter(fires['longitude'], fires['latitude'], 
                                 c=z, s=50, cmap='hot', alpha=0.6)
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')
        axes[1].set_title('Fire Density Heatmap')
        plt.colorbar(scatter, ax=axes[1], label='Density')
        axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{name}_spatial_distribution.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {name}_spatial_distribution.png")
    plt.close()


def generate_summary_report(datasets):
    """Generate a summary report of all datasets."""
    report_path = RESULTS_DIR / "eda_summary_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("WILDFIRE PREDICTION - EDA SUMMARY REPORT\n")
        f.write("="*60 + "\n\n")
        
        for name, df in datasets.items():
            f.write(f"\n{name.upper()} DATASET\n")
            f.write("-"*60 + "\n")
            f.write(f"Shape: {df.shape}\n")
            f.write(f"Date range: {df['date'].min()} to {df['date'].max()}\n" if 'date' in df.columns else "")
            
            if 'fire_occurred' in df.columns:
                fire_rate = df['fire_occurred'].mean() * 100
                f.write(f"Fire occurrence rate: {fire_rate:.2f}%\n")
                f.write(f"Total fires: {df['fire_occurred'].sum()}\n")
            
            f.write("\nKey Statistics:\n")
            desc = df.describe()
            f.write(desc.to_string())
            f.write("\n\n")
    
    print(f"\n  Saved: eda_summary_report.txt")


def main():
    """Run complete EDA pipeline."""
    print("="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Load data
    datasets = load_data()
    
    if not datasets:
        print("No datasets found. Please run data_ingestion.py first.")
        return
    
    # Analyze each dataset
    for name, df in datasets.items():
        print(f"\n{'='*60}")
        print(f"Analyzing {name.upper()} dataset")
        print(f"{'='*60}")
        
        # Basic statistics
        stats = basic_statistics(df, name)
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        plot_target_distribution(df, name)
        plot_weather_distributions(df, name)
        plot_correlation_matrix(df, name)
        plot_weather_by_fire(df, name)
        plot_temporal_patterns(df, name)
        plot_spatial_distribution(df, name)
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(datasets)
    
    print("\n" + "="*60)
    print("EDA Complete!")
    print(f"Results saved to: {RESULTS_DIR.absolute()}")
    print("="*60)


if __name__ == "__main__":
    main()
