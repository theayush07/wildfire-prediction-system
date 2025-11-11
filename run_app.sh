#!/bin/bash

# Script to run the Streamlit app

echo "Starting Wildfire Prediction Web App..."
echo "========================================"
echo ""

# Check if required directories exist
if [ ! -d "models" ]; then
    echo "⚠️  Warning: models/ directory not found"
    echo "Please train models first: python src/models.py"
    echo ""
fi

if [ ! -d "data/processed" ]; then
    echo "⚠️  Warning: data/processed/ directory not found"
    echo "Please run data ingestion first: python scripts/data_ingestion.py"
    echo ""
fi

# Run Streamlit
streamlit run app.py
