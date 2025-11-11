@echo off
REM Script to run the Streamlit app on Windows

echo Starting Wildfire Prediction Web App...
echo ========================================
echo.

REM Check if required directories exist
if not exist "models\" (
    echo Warning: models\ directory not found
    echo Please train models first: python src/models.py
    echo.
)

if not exist "data\processed\" (
    echo Warning: data\processed\ directory not found
    echo Please run data ingestion first: python scripts\data_ingestion.py
    echo.
)

REM Run Streamlit
streamlit run app.py
