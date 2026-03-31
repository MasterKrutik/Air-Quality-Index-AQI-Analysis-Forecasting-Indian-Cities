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
