"""
Complete pipeline runner for wildfire prediction project.
Executes all steps from data ingestion to model evaluation.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and print status."""
    print("\n" + "="*60)
    print(f"STEP: {description}")
    print("="*60)
    
    try:
        result = subprocess.run(cmd, check=True, shell=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error: {e}")
        return False


def main():
    """Run the complete ML pipeline."""
    print("\n" + "="*70)
    print(" "*15 + "WILDFIRE PREDICTION PIPELINE")
    print("="*70)
    
    steps = [
        ("python scripts/data_ingestion.py", "Data Ingestion"),
        ("python scripts/feature_engineering.py", "Feature Engineering"),
        ("python notebooks/01_exploratory_data_analysis.py", "Exploratory Data Analysis"),
        ("python src/models.py --dataset synthetic", "Model Training"),
        ("python src/evaluation.py --dataset synthetic", "Model Evaluation"),
    ]
    
    results = []
    
    for cmd, description in steps:
        success = run_command(cmd, description)
        results.append((description, success))
        
        if not success:
            print(f"\n⚠️  Pipeline stopped at: {description}")
            print("Please fix the error and re-run the pipeline.")
            sys.exit(1)
    
    # Summary
    print("\n" + "="*70)
    print(" "*20 + "PIPELINE SUMMARY")
    print("="*70)
    
    for description, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {description}")
    
    print("\n" + "="*70)
    print("Pipeline completed successfully!")
    print("="*70)
    print("\nNext steps:")
    print("1. Review the results in the 'results/' directory")
    print("2. Launch the web app: streamlit run app.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
