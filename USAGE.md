# Usage Guide

This guide provides detailed instructions for using the Wildfire Prediction System.

## Quick Start

### 1. Setup Environment

\`\`\`bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pandas, sklearn, xgboost, lightgbm, streamlit; print('All dependencies installed!')"
\`\`\`

### 2. Run Complete Pipeline

The easiest way to get started is to run the complete pipeline:

\`\`\`bash
python run_pipeline.py
\`\`\`

This will automatically execute:
1. Data ingestion
2. Feature engineering
3. Exploratory data analysis
4. Model training
5. Model evaluation

### 3. Launch Web App

\`\`\`bash
# On Linux/Mac
./run_app.sh

# On Windows
run_app.bat

# Or directly
streamlit run app.py
\`\`\`

## Step-by-Step Usage

### Data Ingestion

Download and prepare datasets:

\`\`\`bash
python scripts/data_ingestion.py
\`\`\`

This creates:
- `data/raw/` - Raw data files
- `data/processed/` - Cleaned data ready for analysis

### Feature Engineering

Create derived features:

\`\`\`bash
python scripts/feature_engineering.py
\`\`\`

Output: `data/features/` - Engineered features for modeling

### Exploratory Data Analysis

Generate visualizations and statistics:

\`\`\`bash
python notebooks/01_exploratory_data_analysis.py
\`\`\`

Output: `results/eda/` - Charts and summary reports

### Model Training

Train all models:

\`\`\`bash
# Train on synthetic dataset
python src/models.py --dataset synthetic

# Train on UCI dataset
python src/models.py --dataset uci

# With hyperparameter tuning
python src/models.py --dataset synthetic --tune
\`\`\`

Output: `models/` - Trained model files

### Model Evaluation

Evaluate trained models:

\`\`\`bash
python src/evaluation.py --dataset synthetic
\`\`\`

Output: `results/evaluation_*/` - Metrics, plots, and SHAP analysis

### Making Predictions

#### Using Python

\`\`\`python
from src.models import WildfireModel
import pandas as pd

# Load model
model = WildfireModel.load('models/xgboost_synthetic.pkl')

# Prepare input data
data = {
    'temp_max': 35,
    'temp_min': 20,
    'relative_humidity': 25,
    'wind_speed': 20,
    'precipitation': 0,
    # ... other features
}
input_df = pd.DataFrame([data])

# Preprocess
if model.scaler:
    input_scaled = model.scaler.transform(input_df[model.feature_names])
else:
    input_scaled = input_df[model.feature_names].values

# Predict
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

print(f"Fire Risk: {probability*100:.1f}%")
\`\`\`

#### Using Web App

1. Launch the app: `streamlit run app.py`
2. Navigate to the "Prediction" tab
3. Choose input method:
   - **Manual Input**: Use sliders to set conditions
   - **Upload CSV**: Batch predictions from file
4. Click "Predict Fire Risk"
5. View results and risk factors

## Web App Features

### Prediction Tab
- Manual input with interactive sliders
- CSV upload for batch predictions
- Risk level visualization
- Key risk factors identification

### Model Performance Tab
- Performance metrics comparison
- ROC and PR curves
- Confusion matrix
- Cross-model comparison

### Feature Importance Tab
- Feature importance rankings
- SHAP summary plots
- Contribution analysis

### About Tab
- System documentation
- Model limitations
- Use cases and recommendations

## Advanced Usage

### Custom Feature Engineering

Edit `scripts/feature_engineering.py` to add custom features:

\`\`\`python
def create_custom_features(df):
    # Add your custom feature logic
    df['my_custom_feature'] = df['temp_max'] * df['wind_speed']
    return df
\`\`\`

### Hyperparameter Tuning

Optimize model parameters:

\`\`\`python
from src.models import hyperparameter_tuning_xgboost

best_params = hyperparameter_tuning_xgboost(
    X_train, y_train, X_val, y_val,
    n_trials=100
)
\`\`\`

### Testing

Run unit tests:

\`\`\`bash
pytest tests/ -v --cov=src
\`\`\`

## Troubleshooting

### Missing Data

**Error**: `Dataset not found`

**Solution**: Run data ingestion first
\`\`\`bash
python scripts/data_ingestion.py
\`\`\`

### Model Not Found

**Error**: `Model file not found`

**Solution**: Train models before evaluation/prediction
\`\`\`bash
python src/models.py --dataset synthetic
\`\`\`

### Memory Issues

**Error**: `MemoryError during SHAP analysis`

**Solution**: Reduce SHAP sample size in `config.py`:
\`\`\`python
SHAP_SAMPLE_SIZE = 50  # Reduce from 100
\`\`\`

### Import Errors

**Error**: `ModuleNotFoundError`

**Solution**: Install all dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Best Practices

1. **Always run the complete pipeline** for new datasets
2. **Use version control** to track model changes
3. **Document any custom modifications** in the code
4. **Evaluate on held-out test data** to avoid overfitting
5. **Regular model retraining** as new data becomes available
6. **Monitor prediction distributions** for data drift

## Support

For issues or questions:
1. Check this usage guide
2. Review the README.md
3. Examine error logs
4. Contact the development team
