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

## 🔄 Project Pipeline

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
