# Air-Quality-Index-AQI-Analysis-Forecasting-Indian-Cities

## Objective
To analyze historical air quality data across major Indian cities and build a forecasting model to predict future pollution levels. The project aims to uncover patterns and trends in air pollution (PM2.5, PM10, NO2, SO2, etc.) and provide actionable insights for policymakers.

## 📁 Project Structure

```
AQI_Project/
├── datasets/
│   └── city_day.csv                          # Raw air quality data
├── cleaned_data/
│   └── cleaned_city_day.csv                  # Processed and cleaned data
├── notebooks/
│   ├── 1_cleaning.ipynb                      # Data cleaning and preprocessing
│   ├── 2_eda.ipynb                           # Exploratory data analysis
│   ├── 3_arima_forecast.ipynb                # ARIMA model development
│   ├── 4_prophet_forecast.ipynb              # Prophet model development
│   ├── 5_model_comparison.ipynb              # Model evaluation and comparison
│   └── 6_final_forecast.ipynb                # Final forecasting results
├── forecasts/
│   ├── arima_forecast_all_cities.csv        # ARIMA predictions
│   ├── prophet_forecast_all_cities.csv      # Prophet predictions
│   ├── final_forecast.csv                    # Best model predictions
│   └── model_evaluation_all_cities.csv      # Model performance metrics
├── dashboard/
│   └── app.py                                # Streamlit interactive dashboard
├── report/                                   # Final analysis reports
└── README.md                                 # This file
```

## Project Pipeline

### 1. **Data Cleaning** (`1_cleaning.ipynb`)
   - Load and inspect raw AQI data
   - Handle missing values
   - Remove outliers
   - Feature engineering and preprocessing
   - Export cleaned dataset

### 2. **Exploratory Data Analysis** (`2_eda.ipynb`)
   - Statistical summaries of AQI data
   - Temporal trends and seasonality analysis
   - Geographic patterns across cities
   - Correlation analysis
   - Visualization of key insights

### 3. **ARIMA Forecasting** (`3_arima_forecast.ipynb`)
   - Time series stationarity testing (ADF test)
   - ARIMA parameter selection and tuning
   - Model training and validation
   - Generate AQI forecasts for all cities
   - Calculate performance metrics

### 4. **Prophet Forecasting** (`4_prophet_forecast.ipynb`)
   - Prophet model implementation
   - Handling seasonality and trends
   - Parameter optimization
   - Generate forecasts with confidence intervals
   - Performance evaluation

### 5. **Model Comparison** (`5_model_comparison.ipynb`)
   - Compare ARIMA vs Prophet models
   - Analyze MAE, RMSE, and other metrics
   - Visual comparison of predictions
   - Model selection and recommendations

### 6. **Final Forecast** (`6_final_forecast.ipynb`)
   - Generate final forecasts using best model
   - Create summary statistics
   - Produce final output files

## Getting Started

### Prerequisites
- Python 3.8+
- pip or conda package manager

### Installation

1. Clone or download the project:
```bash
cd AQI_Project
```

2. Create and activate virtual environment:
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Project

#### Jupyter Notebooks
Run notebooks in sequence (1-6) for the complete analysis:
```bash
jupyter notebook notebooks/
```

#### Interactive Dashboard
Launch the Streamlit dashboard for real-time visualization:
```bash
streamlit run dashboard/app.py
```

## Key Results

- **ARIMA Model**: Time series decomposition with ARIMA(p,d,q) parameters
- **Prophet Model**: Trend and seasonality forecasting with confidence intervals
- **Model Performance**: Comparison of MAE, RMSE, and MAPE metrics
- **Forecasts**: Generated predictions for multiple cities and time horizons

### Output Files
- `cleaned_city_day.csv` - Processed data ready for modeling
- `arima_forecast_all_cities.csv` - ARIMA model predictions
- `prophet_forecast_all_cities.csv` - Prophet model predictions
- `final_forecast.csv` - Best model forecasts
- `model_evaluation_all_cities.csv` - Performance metrics for all models

## Dashboard Features

The Streamlit dashboard (`dashboard/app.py`) provides:
- Interactive city selection for AQI trends
- Real-time forecast visualization
- Historical vs predicted values comparison
- Model performance metrics display
- Data filtering and date range selection

## Technologies Used

- **Python** - Programming language
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning utilities
- **statsmodels** - ARIMA models
- **Prophet** - Time series forecasting (Facebook)
- **Matplotlib & Seaborn** - Data visualization
- **Streamlit** - Interactive dashboard framework

## Data Dictionary

The dataset includes the following air quality parameters:
- **AQI** - Air Quality Index (Target variable)
- **PM2.5, PM10** - Particulate matter concentrations
- **NO2, SO2, CO, O3** - Gaseous pollutant concentrations
- **City** - Geographic location
- **Date** - Temporal information

## Model Hyperparameters

### ARIMA
- Auto-selected (p,d,q) based on ADF test and ACF/PACF plots
- Training/validation split: 80/20

### Prophet
- Yearly seasonality enabled
- Growth model: Linear
- Changepoint prior scale: 0.05

## Performance Metrics

Models evaluated using:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)

**Last Updated**: April 2026
=======
