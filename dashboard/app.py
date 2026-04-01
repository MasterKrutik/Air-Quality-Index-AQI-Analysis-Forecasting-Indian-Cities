from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


st.set_page_config(page_title="AQI Analysis Dashboard", layout="wide")
sns.set_theme(style="whitegrid")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "datasets" / "city_day.csv"
CLEANED_DATA_PATH = PROJECT_ROOT / "cleaned_data" / "cleaned_city_day.csv"
ARIMA_PATH = PROJECT_ROOT / "forecasts" / "arima_forecast_all_cities.csv"
PROPHET_PATH = PROJECT_ROOT / "forecasts" / "prophet_forecast_all_cities.csv"
EVAL_PATH = PROJECT_ROOT / "forecasts" / "model_evaluation_all_cities.csv"
COORD_PATH = PROJECT_ROOT / "datasets" / "city_coordinates.csv"

MONTH_NAMES = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
POLLUTANT_COLS = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
    "Benzene",
    "Toluene",
    "Xylene",
]
DISPLAY_POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]


@st.cache_data
def load_data():
    raw_df = pd.read_csv(RAW_DATA_PATH)
    cleaned_df = pd.read_csv(CLEANED_DATA_PATH)
    arima_df = pd.read_csv(ARIMA_PATH)
    prophet_df = pd.read_csv(PROPHET_PATH)
    eval_df = pd.read_csv(EVAL_PATH)
    coord_df = pd.read_csv(COORD_PATH)

    for frame in (raw_df, cleaned_df, arima_df, prophet_df):
        if "Date" in frame.columns:
            frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")

    cleaned_df = cleaned_df.sort_values(["City", "Date"]).reset_index(drop=True)
    arima_df = arima_df.sort_values(["City", "Date"]).reset_index(drop=True)
    prophet_df = prophet_df.sort_values(["City", "Date"]).reset_index(drop=True)

    return raw_df, cleaned_df, arima_df, prophet_df, eval_df, coord_df


def get_aqi_category(aqi_value):
    if pd.isna(aqi_value):
        return "Unknown"
    if aqi_value <= 50:
        return "Good"
    if aqi_value <= 100:
        return "Satisfactory"
    if aqi_value <= 200:
        return "Moderate"
    if aqi_value <= 300:
        return "Poor"
    if aqi_value <= 400:
        return "Very Poor"
    return "Severe"


def format_number(value, decimals=2):
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"


@st.cache_data
def build_city_summary(cleaned_df, coord_df):
    latest_city = (
        cleaned_df.sort_values("Date")
        .groupby("City")
        .tail(1)[["City", "Date", "AQI"]]
        .rename(columns={"Date": "Latest_Date", "AQI": "Latest_AQI"})
    )

    monthly = cleaned_df.assign(
        Month_Num=cleaned_df["Date"].dt.month,
        Month=cleaned_df["Date"].dt.strftime("%b"),
    )
    peak_month = (
        monthly.groupby(["City", "Month_Num", "Month"], as_index=False)["AQI"]
        .mean()
        .sort_values(["City", "AQI"], ascending=[True, False])
        .drop_duplicates("City")
        .rename(columns={"Month": "Peak_Month", "AQI": "Peak_Month_AQI"})
        [["City", "Peak_Month", "Peak_Month_AQI"]]
    )

    pollutant_means = cleaned_df.groupby("City")[DISPLAY_POLLUTANTS].mean()
    dominant_pollutant = pollutant_means.apply(
        lambda row: row.dropna().idxmax() if row.notna().any() else "Unavailable",
        axis=1,
    ).rename("Dominant_Pollutant")

    city_summary = (
        cleaned_df.groupby("City", as_index=False)
        .agg(
            Avg_AQI=("AQI", "mean"),
            Max_AQI=("AQI", "max"),
            Min_AQI=("AQI", "min"),
            Records=("AQI", "size"),
        )
        .merge(latest_city, on="City", how="left")
        .merge(peak_month, on="City", how="left")
        .merge(dominant_pollutant.reset_index(), on="City", how="left")
        .merge(coord_df, on="City", how="left")
    )
    city_summary["Risk_Level"] = city_summary["Avg_AQI"].apply(get_aqi_category)
    return city_summary.sort_values("Avg_AQI", ascending=False).reset_index(drop=True)


def filter_forecast_horizon(forecast_df, months):
    if forecast_df.empty:
        return forecast_df
    start_date = forecast_df["Date"].min()
    cutoff = start_date + pd.DateOffset(months=months) - pd.Timedelta(days=1)
    return forecast_df[forecast_df["Date"] <= cutoff].copy()


def get_city_actions(latest_aqi, peak_month, dominant_pollutant):
    actions = [
        "Strengthen vehicle emission checks and prioritize public transport corridors on high-traffic routes.",
        "Expand urban green zones near dense roads, schools, and industrial clusters to reduce exposure hotspots.",
        "Issue festival-season and winter inversion warnings with temporary firecracker, waste-burning, and diesel-generator controls.",
    ]

    if dominant_pollutant in {"PM2.5", "PM10"}:
        actions.append(
            "Focus on road dust suppression, construction-site compliance, and cleaner freight movement because particulate matter is the main pressure point."
        )
    elif dominant_pollutant in {"NO2", "NO", "NOx", "CO"}:
        actions.append(
            "Target traffic and fuel-combustion sources with cleaner fleets, congestion management, and stricter inspection drives."
        )
    elif dominant_pollutant == "SO2":
        actions.append(
            "Tighten industrial fuel standards and monitor stack emissions in power, refinery, and manufacturing belts."
        )
    elif dominant_pollutant == "O3":
        actions.append(
            "Pair ozone alerts with heat-season advisories and control precursor gases from traffic and industry during sunny periods."
        )

    if peak_month in {"Oct", "Nov", "Dec", "Jan"}:
        actions.append(
            f"Prepare a {peak_month}-focused action plan because this city's historical peak pollution period falls in the late-year winter window."
        )

    if latest_aqi >= 200:
        actions.append(
            "Activate short-term health protection steps such as mask advisories, school exposure guidance, and outdoor activity restrictions for vulnerable groups."
        )
    elif latest_aqi >= 100:
        actions.append(
            "Use daily public advisories to warn sensitive groups, reschedule strenuous outdoor events, and promote indoor air safety messaging."
        )

    return actions


raw_df, df, arima_forecast, prophet_forecast, eval_df, coord_df = load_data()
city_summary = build_city_summary(df, coord_df)

city_list = sorted(df["City"].dropna().unique())
forecast_cities = sorted(set(arima_forecast["City"].dropna()) & set(prophet_forecast["City"].dropna()))

missing_before = int(raw_df.isna().sum().sum())
missing_after = int(df.isna().sum().sum())
missing_reduction_pct = ((missing_before - missing_after) / missing_before * 100) if missing_before else 0
aqi_gaps_before = int(raw_df["AQI"].isna().sum())
aqi_gaps_after = int(df["AQI"].isna().sum())
overall_peak_month = (
    df.assign(Month=df["Date"].dt.strftime("%b"), Month_Num=df["Date"].dt.month)
    .groupby(["Month_Num", "Month"], as_index=False)["AQI"]
    .mean()
    .sort_values("AQI", ascending=False)
    .iloc[0]["Month"]
)

st.title("AQI Analysis & Forecasting Dashboard")
st.caption(
    "A project dashboard for Indian city AQI analysis covering data cleaning, EDA, forecasting, high-risk zones, and policy recommendations."
)

with st.sidebar:
    st.header("Controls")
    default_city_index = city_list.index("Delhi") if "Delhi" in city_list else 0
    selected_city = st.selectbox("Select city", city_list, index=default_city_index)
    forecast_model = st.radio("Forecast model", ["ARIMA", "PROPHET", "COMPARE BOTH"])
    forecast_months = st.slider("Forecast horizon (months)", 6, 12, 12)
    top_n_cities = st.slider("High-risk cities to display", 5, 15, 10)
    st.markdown("---")
    st.caption(f"Cities in cleaned data: {df['City'].nunique()}")
    st.caption(f"Cities with forecasts: {len(forecast_cities)}")
    st.caption(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    st.caption(f"Cleaned records: {len(df):,}")

selected_city_data = (
    df[df["City"] == selected_city].groupby("Date", as_index=False)["AQI"].mean().sort_values("Date")
)
selected_city_details = city_summary[city_summary["City"] == selected_city].iloc[0]
selected_monthly = (
    df[df["City"] == selected_city]
    .assign(Month_Num=lambda x: x["Date"].dt.month, Month=lambda x: x["Date"].dt.strftime("%b"))
    .groupby(["Month_Num", "Month"], as_index=False)["AQI"]
    .mean()
    .sort_values("Month_Num")
)

tabs = st.tabs(
    [
        "Executive Summary",
        "EDA & Seasonality",
        "Forecasting",
        "Risk & Recommendations",
    ]
)

with tabs[0]:
    st.subheader("Project Coverage")
    st.markdown(
        """
        - Cleaned and preprocessed raw AQI data to address missing and inconsistent values.
        - Conducted EDA to study pollutant distributions, city-wise patterns, and seasonal trends.
        - Applied time-series forecasting with ARIMA and Facebook Prophet for the next 6-12 months.
        - Compared models using the saved evaluation outputs available in the repository.
        - Identified high-risk zones, peak pollution periods, and city-specific intervention priorities.
        - Converted the analysis into a dashboard with Matplotlib, Seaborn, and interactive Streamlit controls.
        """
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Cities Analyzed", f"{df['City'].nunique()}")
    metric_cols[1].metric("Daily Records", f"{len(df):,}")
    metric_cols[2].metric("Missing Values Reduced", f"{missing_reduction_pct:.1f}%")
    metric_cols[3].metric("AQI Gaps Filled", f"{aqi_gaps_before - aqi_gaps_after:,}")

    metric_cols_2 = st.columns(4)
    metric_cols_2[0].metric("Worst Avg AQI City", city_summary.iloc[0]["City"])
    metric_cols_2[1].metric("Nationwide Peak Month", overall_peak_month)
    metric_cols_2[2].metric("Selected City Risk", selected_city_details["Risk_Level"])
    metric_cols_2[3].metric("Forecast Horizon", f"{forecast_months} months")

    cleaning_summary = pd.DataFrame(
        {
            "Metric": ["Total missing cells", "Missing AQI values", "Cities covered", "Daily records"],
            "Raw": [
                f"{missing_before:,}",
                f"{aqi_gaps_before:,}",
                f"{raw_df['City'].nunique()}",
                f"{len(raw_df):,}",
            ],
            "Cleaned": [
                f"{missing_after:,}",
                f"{aqi_gaps_after:,}",
                f"{df['City'].nunique()}",
                f"{len(df):,}",
            ],
        }
    )
    st.subheader("Cleaning Snapshot")
    st.dataframe(cleaning_summary, width="stretch", hide_index=True)

    st.info(
        "Outcome: the dashboard summarizes India's pollution problem with city-wise breakdowns, 6-12 month forecasts, and action areas such as vehicle emission control, green zone promotion, and festival-season warnings."
    )

    national_monthly = (
        df.set_index("Date")
        .resample("MS")["AQI"]
        .mean()
        .reset_index()
        .rename(columns={"AQI": "Monthly_AQI"})
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(national_monthly["Date"], national_monthly["Monthly_AQI"], color="#c0392b", linewidth=2)
    ax.set_title("India-Wide Monthly Average AQI Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average AQI")
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    summary_view = city_summary[
        [
            "City",
            "Avg_AQI",
            "Latest_AQI",
            "Peak_Month",
            "Dominant_Pollutant",
            "Risk_Level",
        ]
    ].copy()
    st.subheader("City-Wise Pollution Summary")
    st.dataframe(
        summary_view.round({"Avg_AQI": 2, "Latest_AQI": 2}),
        width="stretch",
        hide_index=True,
    )

with tabs[1]:
    st.subheader(f"{selected_city}: Historical AQI and Seasonal Trends")
    city_metric_cols = st.columns(4)
    city_metric_cols[0].metric("Average AQI", format_number(selected_city_details["Avg_AQI"]))
    city_metric_cols[1].metric("Max AQI", format_number(selected_city_details["Max_AQI"]))
    city_metric_cols[2].metric("Min AQI", format_number(selected_city_details["Min_AQI"]))
    city_metric_cols[3].metric(
        "Latest AQI",
        f"{format_number(selected_city_details['Latest_AQI'])} ({get_aqi_category(selected_city_details['Latest_AQI'])})",
    )

    eda_col_1, eda_col_2 = st.columns(2)

    with eda_col_1:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(selected_city_data["Date"], selected_city_data["AQI"], color="#1f77b4", linewidth=1.8)
        ax.set_title(f"{selected_city} Historical AQI")
        ax.set_xlabel("Date")
        ax.set_ylabel("AQI")
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with eda_col_2:
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.barplot(data=selected_monthly, x="Month", y="AQI", order=MONTH_NAMES, color="#6baed6", ax=ax)
        ax.set_title(f"{selected_city} Average AQI by Month")
        ax.set_xlabel("Month")
        ax.set_ylabel("Average AQI")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    distribution_col, correlation_col = st.columns(2)

    with distribution_col:
        pollutant_distribution = (
            df[DISPLAY_POLLUTANTS]
            .melt(var_name="Pollutant", value_name="Value")
            .dropna()
        )
        if len(pollutant_distribution) > 20000:
            pollutant_distribution = pollutant_distribution.sample(20000, random_state=42)
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.boxplot(data=pollutant_distribution, x="Pollutant", y="Value", color="#9ecae1", ax=ax)
        ax.set_title("National Pollutant Distribution")
        ax.set_xlabel("")
        ax.set_ylabel("Concentration")
        ax.tick_params(axis="x", rotation=30)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with correlation_col:
        city_pollutants = df[df["City"] == selected_city][DISPLAY_POLLUTANTS].dropna(how="all")
        usable_pollutants = city_pollutants.dropna(axis=1, how="all")
        if usable_pollutants.shape[1] < 2:
            st.info("Not enough pollutant coverage is available to compute a stable correlation heatmap for this city.")
        else:
            corr_matrix = usable_pollutants.corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(corr_matrix, cmap="RdYlBu_r", annot=True, fmt=".2f", linewidths=0.5, ax=ax)
            ax.set_title(f"{selected_city} Pollutant Correlation Heatmap")
            st.pyplot(fig, width="stretch")
            plt.close(fig)

    st.subheader("Top Polluted Cities by Average AQI")
    top_polluted = city_summary.head(top_n_cities).copy()
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.barplot(data=top_polluted, x="City", y="Avg_AQI", color="#5c7cfa", ax=ax)
    ax.set_title("Highest-Risk Cities Based on Average AQI")
    ax.set_xlabel("")
    ax.set_ylabel("Average AQI")
    ax.tick_params(axis="x", rotation=35)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

with tabs[2]:
    st.subheader(f"{selected_city}: 6-12 Month AQI Forecast")

    city_arima = filter_forecast_horizon(
        arima_forecast[arima_forecast["City"] == selected_city].copy(),
        forecast_months,
    )
    city_prophet = filter_forecast_horizon(
        prophet_forecast[prophet_forecast["City"] == selected_city].copy(),
        forecast_months,
    )
    recent_history = selected_city_data[
        selected_city_data["Date"] >= selected_city_data["Date"].max() - pd.DateOffset(months=18)
    ]

    if selected_city not in forecast_cities or city_arima.empty or city_prophet.empty:
        st.warning(
            "Forecast output is not available for this city in the saved repository files. Pick one of the forecast-supported cities from the dataset to view the model section."
        )
    else:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(recent_history["Date"], recent_history["AQI"], label="Historical AQI", color="#2c3e50", linewidth=2)

        if forecast_model == "ARIMA":
            ax.plot(city_arima["Date"], city_arima["Forecast_AQI"], label="ARIMA Forecast", color="#d35400", linewidth=2)
            forecast_for_download = city_arima.copy()
            forecast_for_download["Model"] = "ARIMA"
        elif forecast_model == "PROPHET":
            ax.plot(
                city_prophet["Date"],
                city_prophet["Forecast_AQI"],
                label="Prophet Forecast",
                color="#16a085",
                linewidth=2,
            )
            forecast_for_download = city_prophet.copy()
            forecast_for_download["Model"] = "PROPHET"
        else:
            ax.plot(city_arima["Date"], city_arima["Forecast_AQI"], label="ARIMA Forecast", color="#d35400", linewidth=2)
            ax.plot(
                city_prophet["Date"],
                city_prophet["Forecast_AQI"],
                label="Prophet Forecast",
                color="#16a085",
                linewidth=2,
            )
            forecast_for_download = pd.concat(
                [
                    city_arima.assign(Model="ARIMA"),
                    city_prophet.assign(Model="PROPHET"),
                ],
                ignore_index=True,
            )

        ax.set_title(f"{selected_city} AQI Forecast for the Next {forecast_months} Months")
        ax.set_xlabel("Date")
        ax.set_ylabel("AQI")
        ax.legend()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        model_eval = eval_df[eval_df["City"] == selected_city]
        forecast_cols = st.columns(4)
        forecast_cols[0].metric("Last Observed AQI", format_number(selected_city_details["Latest_AQI"]))
        forecast_cols[1].metric("ARIMA Mean Forecast", format_number(city_arima["Forecast_AQI"].mean()))
        forecast_cols[2].metric("Prophet Mean Forecast", format_number(city_prophet["Forecast_AQI"].mean()))
        forecast_cols[3].metric(
            "End-of-Horizon AQI Gap",
            format_number(abs(city_arima["Forecast_AQI"].iloc[-1] - city_prophet["Forecast_AQI"].iloc[-1])),
        )

        if not model_eval.empty:
            st.subheader("Saved Model Evaluation")
            st.dataframe(model_eval.round(2), width="stretch", hide_index=True)

        st.info(
            "Model comparison note: the saved evaluation CSV in this repository contains ARIMA RMSE/MAE only. This makes ARIMA the best-supported validated model in the current project files, while Prophet remains available as a scenario comparison forecast."
        )

        csv_data = forecast_for_download.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Forecast CSV",
            data=csv_data,
            file_name=f"{selected_city.lower().replace(' ', '_')}_{forecast_model.lower().replace(' ', '_')}_forecast.csv",
            mime="text/csv",
        )
        st.dataframe(
            forecast_for_download.head(20).round({"Forecast_AQI": 2}),
            width="stretch",
            hide_index=True,
        )

with tabs[3]:
    st.subheader("High-Risk Zones and Peak Pollution Periods")
    high_risk = city_summary.head(top_n_cities)[
        [
            "City",
            "Avg_AQI",
            "Latest_AQI",
            "Peak_Month",
            "Dominant_Pollutant",
            "Risk_Level",
        ]
    ].copy()
    st.dataframe(
        high_risk.round({"Avg_AQI": 2, "Latest_AQI": 2}),
        width="stretch",
        hide_index=True,
    )

    heatmap_cities = city_summary.head(min(top_n_cities, 10))["City"].tolist()
    seasonal_heatmap = (
        df[df["City"].isin(heatmap_cities)]
        .assign(Month_Num=lambda x: x["Date"].dt.month, Month=lambda x: x["Date"].dt.strftime("%b"))
        .groupby(["City", "Month_Num", "Month"], as_index=False)["AQI"]
        .mean()
        .pivot(index="City", columns="Month", values="AQI")
        .reindex(columns=MONTH_NAMES)
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(seasonal_heatmap, cmap="YlOrRd", linewidths=0.5, ax=ax)
    ax.set_title("Peak Pollution Months Across High-Risk Cities")
    ax.set_xlabel("Month")
    ax.set_ylabel("")
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.subheader(f"Recommended Interventions for {selected_city}")
    st.markdown(
        f"""
        **Current status:** {selected_city} has a latest AQI of **{format_number(selected_city_details['Latest_AQI'])}**
        and an average AQI of **{format_number(selected_city_details['Avg_AQI'])}**, which falls in the
        **{selected_city_details['Risk_Level']}** band.
        """
    )

    for action in get_city_actions(
        selected_city_details["Latest_AQI"],
        selected_city_details["Peak_Month"],
        selected_city_details["Dominant_Pollutant"],
    ):
        st.markdown(f"- {action}")

    st.subheader("Public Health Recommendation")
    latest_category = get_aqi_category(selected_city_details["Latest_AQI"])
    if latest_category in {"Poor", "Very Poor", "Severe"}:
        st.error(
            "Sensitive groups should limit outdoor exposure, schools should review activity intensity, and public advisories should promote masks and indoor air precautions."
        )
    elif latest_category == "Moderate":
        st.warning(
            "People with respiratory or cardiac conditions should reduce strenuous outdoor activity and follow local air-quality alerts."
        )
    else:
        st.success(
            "Current conditions are relatively safer, but seasonal monitoring and preventive controls should remain active."
        )
