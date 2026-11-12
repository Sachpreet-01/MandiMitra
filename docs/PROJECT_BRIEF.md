# MandiMitra – Project Brief

## 1. Project Overview

MandiMitra is an India-wide, multi-commodity agricultural market intelligence platform designed to help farmers and Farmer Producer Organisations (FPOs) make better-informed market selection and selling decisions.

The application combines historical mandi price analysis, relative market performance forecasting and a net realisation calculator to provide practical decision-support insights.

## 2. Problem Statement

Farmers often have limited access to tools that help them compare agricultural markets, understand historical price behaviour, evaluate price volatility and identify markets that may offer better relative price performance.

MandiMitra addresses this problem by transforming historical agricultural market data into accessible analytical insights and short-term forecasts.

## 3. Target Users

- Small and medium-sized farmers.
- Farmer Producer Organisations (FPOs).
- Agricultural market analysts and researchers.

## 4. Geographical and Commodity Scope

**Geographical coverage:** India.

**Supported commodities:**
1. Cotton
2. Grapes
3. Onion
4. Pomegranate
5. Soybean
6. Tomato

Coverage depends on the availability and quality of historical market observations.

## 5. Application Features

### 5.1 Historical Market Explorer

- Explore historical mandi prices.
- Compare price movements across markets.
- Examine commodity-specific market behaviour.

### 5.2 Market Performance Outlook

- Estimate short-term relative market price performance.
- Compare markets within commodity-specific peer groups.
- Present quantile-based prediction intervals to communicate uncertainty.

### 5.3 Net Realisation Calculator

- Estimate potential net selling realisation.
- Account for relevant selling and transportation costs.
- Compare the financial implications of different market choices.

## 6. Data and Methodology

MandiMitra uses historical agricultural market price observations sourced from AGMARKNET / data.gov.in.

The application dataset contains:

- 11,286,611 market-price records.
- 45 vegetables.
- 2,167 distinct market names.
- 2,211 physical market identifiers.
- 546 districts across 30 states.
- Application data coverage from January 2021 to 31 October 2025.

The project uses data preprocessing, exploratory analysis, chronological model evaluation and market-level time-series comparisons.

## 7. Machine Learning

The relative market performance forecasting engine uses LightGBM Quantile Regression to estimate seven-day market performance relative to the median of comparable peer markets.

Model evaluation includes:

- Chronological holdout testing.
- Comparison against persistence and current-ratio baselines.
- Mean Absolute Error (MAE).
- Paired moving-block bootstrap confidence intervals.
- Prediction interval coverage analysis.

The project also evaluates raw-price forecasting separately and does not assume that a more complex model automatically outperforms a simple baseline.

## 8. Technology Stack

- Python
- Pandas and NumPy
- DuckDB and PyArrow
- Scikit-learn
- LightGBM
- XGBoost
- Streamlit
- Altair
- Matplotlib and Seaborn
- Git and GitHub

## 9. Limitations and Responsible Use

- The application uses historical data ending on 31 October 2025; it does not provide live mandi prices.
- Forecasts estimate relative market performance, not guaranteed future prices or profits.
- Market comparisons depend on the availability of sufficiently comparable observations.
- Prediction uncertainty and commodity-specific performance differences should be considered before making decisions.

MandiMitra is a research and portfolio prototype intended to support informed decisions, not replace local market knowledge or professional agricultural advice.

## 10. Project Repository

The repository contains the Streamlit application, processed application datasets, model artifacts, research notebooks, documentation and model validation tests.

**Live application:** https://mandimitra-p9yunvabtc39o6vutveoer.streamlit.app/

**GitHub repository:** https://github.com/Sachpreet-01/MandiMitra
