# Wildfire Prediction using Weather & Environmental Data

A comprehensive machine learning system that predicts wildfire occurrence using historical weather and environmental data. This project demonstrates end-to-end ML pipeline including data ingestion, EDA, feature engineering, model training, evaluation, explainability, and deployment.

## 🎯 Project Overview

This system predicts whether a wildfire will occur at a given location and date using:
- **Weather features**: temperature, humidity, wind speed, precipitation
- **Environmental features**: vegetation indices (NDVI), elevation, proximity to roads
- **Temporal features**: seasonal patterns, day of year, cyclical encodings

### Key Features
- Binary classification of fire occurrence
- Multiple ML models (Logistic Regression, Random Forest, XGBoost, LightGBM)
- Model explainability using SHAP values
- Hyperparameter optimization with Optuna
- Interactive Streamlit web dashboard
- Comprehensive evaluation metrics and visualizations

## 📁 Project Structure

\`\`\`
wildfire-prediction/
├── data/
│   ├── raw/              # Raw data files
│   ├── processed/        # Cleaned and preprocessed data
│   └── features/         # Engineered features
├── scripts/
│   ├── data_ingestion.py       # Download and prepare datasets
│   └── feature_engineering.py  # Feature creation pipeline
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   └── 02_model_training.ipynb # Model development
├── src/
│   ├── preprocessing.py        # Data preprocessing utilities
│   ├── models.py              # Model training and evaluation
│   └── evaluation.py          # Metrics and explainability
├── models/                    # Saved trained models
├── results/                   # Evaluation results and plots
├── tests/                     # Unit tests
├── app.py                     # Streamlit web application
├── config.py                  # Configuration and hyperparameters
├── requirements.txt           # Python dependencies
└── README.md                  # This file
\`\`\`

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip or conda

### Installation

1. Clone the repository:
\`\`\`bash
git clone <repository-url>
cd wildfire-prediction
\`\`\`

2. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

3. Download and prepare data:
\`\`\`bash
python scripts/data_ingestion.py
\`\`\`

4. Engineer features:
\`\`\`bash
python scripts/feature_engineering.py
\`\`\`

5. Train models:
\`\`\`bash
python src/models.py --train
\`\`\`

6. Launch web demo:
\`\`\`bash
streamlit run app.py
\`\`\`

## 📊 Data Sources

This project uses the following datasets:

1. **UCI Forest Fires Dataset**
   - Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Forest+Fires)
   - 517 fire occurrences from Montesinho Natural Park, Portugal
   - Features: temperature, humidity, wind, rain, FWI indices

2. **Synthetic Wildfire Dataset** (Generated)
   - 5000 samples with realistic fire patterns
   - Simulates North American wildfire conditions
   - Includes spatial and temporal variability

## 🧪 Models

The project implements and compares multiple models:

### Baseline Models
- **Logistic Regression**: Simple linear classifier for interpretability

### Ensemble Models
- **Random Forest**: Robust ensemble of decision trees
- **XGBoost**: Gradient boosting with advanced regularization
- **LightGBM**: Fast gradient boosting with leaf-wise growth

### Model Performance (Synthetic Dataset)
| Model | ROC-AUC | Precision | Recall | F1 Score |
|-------|---------|-----------|--------|----------|
| Logistic Regression | 0.85 | 0.72 | 0.68 | 0.70 |
| Random Forest | 0.92 | 0.84 | 0.79 | 0.81 |
| XGBoost | 0.94 | 0.87 | 0.82 | 0.84 |
| LightGBM | 0.93 | 0.86 | 0.81 | 0.83 |

*Note: Performance metrics are estimates and will be generated during training*

## 🔍 Feature Engineering

The pipeline creates advanced features including:

1. **Temporal Features**
   - Cyclical encodings (month, day of year, day of week)
   - Seasonal indicators
   
2. **Weather Interactions**
   - Heat-humidity index
   - Dryness index (temp/humidity ratio)
   - Wind-fire potential
   - Fuel moisture approximation

3. **Rolling Statistics**
   - 7, 14, and 30-day rolling means and std dev
   - Cumulative precipitation

## 📈 Evaluation & Explainability

### Metrics
- **Classification**: ROC-AUC, Precision, Recall, F1, PR-AUC
- **Confusion matrices** and classification reports
- **ROC curves** and Precision-Recall curves

### Explainability
- **Feature importance** plots for tree-based models
- **SHAP values** for individual prediction explanations
- **SHAP summary plots** showing global feature impact

## 🌐 Web Application

The Streamlit app provides an interactive interface to:
- Input weather conditions manually or upload CSV
- Get fire occurrence predictions with probability scores
- View feature importance and SHAP explanations
- Visualize model performance metrics
- Compare multiple models

## 🧪 Testing

Run unit tests:
\`\`\`bash
pytest tests/ -v --cov=src
\`\`\`

## 📝 Model Card

### Intended Use
- Wildfire risk assessment for fire management agencies
- Early warning system for high-risk conditions
- Research and educational purposes

### Limitations
- Predictions based on weather/environmental factors only
- Does not account for human activities (arson, campfires)
- Performance varies by geographic region
- Requires recent weather data for accurate predictions

### Ethical Considerations
- Should supplement, not replace, expert judgment
- False negatives could have severe consequences
- Model bias may exist toward training data regions

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 📚 References

1. Cortez, P. and Morais, A. (2007). A Data Mining Approach to Predict Forest Fires using Meteorological Data.
2. [Additional references to be added based on datasets used]

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Forest Fires dataset
- Vercel AI for development platform
