# MandiMitra 🌾

### Historical Agricultural Market Intelligence and Relative-Performance Forecasting

MandiMitra is an India-wide agricultural market intelligence and machine learning project designed to help users explore historical mandi prices, compare agricultural markets, and assess potential market performance.

The application uses **11,286,611 historical market-price records**, covering **45 vegetables, 2,167 distinct market names, 2,211 physical market identifiers, 546 districts, and 30 states across India**, from January 2021 to October 2025.

**Live Demo:** [Open MandiMitra](https://mandimitra-p9yunvabtc39o6vutveoer.streamlit.app/)

**GitHub Repository:** [Sachpreet-01/MandiMitra](https://github.com/Sachpreet-01/MandiMitra)

---

## Features

### 1. Historical Market Explorer

Explore historical modal prices across commodities, markets, and reporting dates. Examine price movements, compare a selected market against comparable peers, and investigate market-level reporting patterns.

### 2. Market Performance Outlook

Compare a market's predicted seven-day relative price performance against the median of eligible peer markets.

The forecasting system uses **LightGBM Quantile Regression** to estimate median performance and prediction uncertainty. The application displays precomputed model observations rather than running model inference for every user interaction.

### 3. Net Realisation Calculator

Estimate potential net selling realisation after accounting for costs such as transportation, commission, other expenses, and wastage. Compare the financial implications of selling at different markets using user-provided assumptions.

---

## Key Findings

### Raw-price forecasting

Across 74 held-out market-commodity series, a persistence baseline (next price = current price) achieved **7.83% sMAPE**, outperforming global XGBoost at **8.18%**.

The persistence baseline performed better on sMAPE in all 74 series.

This motivated exploring a different forecasting formulation rather than assuming that a more complex model would necessarily perform better.

### Relative market performance

The forecasting task was reframed to predict seven-day market performance relative to the corresponding peer-market median.

| Model | Test MAE |
|---|---:|
| Current Ratio Baseline | 0.103698 |
| LightGBM Quantile Median | **0.090914** |

LightGBM Quantile Median reduced test MAE by **12.33%** relative to the baseline.

A paired moving-block bootstrap estimated a 95% confidence interval of **11.90%–12.73%** for the relative MAE improvement.

MAE is measured in absolute log-ratio units, not rupees.

### Top-three market selection

On **23,626 comparable test groups**, each containing at least five markets, LightGBM Quantile Median improved mean actual top-three log-ratio performance over the current-ratio baseline.

| Measure | Result |
|---|---:|
| Baseline mean actual top-three performance | 0.618114 |
| LightGBM Quantile Median | **0.626881** |
| Mean improvement | **0.008767** |

The model's group-level outcomes were:

- Wins: 4,932 (20.88%)
- Ties: 15,471 (65.48%)
- Losses: 3,223 (13.64%)

A paired moving-block bootstrap produced a 95% confidence interval of **[0.008050, 0.009544]** for the improvement using 14-observed-date blocks. The interval remained positive with block sizes of 7, 14, and 28 observed dates.

This is a top-three selection evaluation, not a test NDCG or Spearman evaluation.

### Prediction uncertainty

Commodity-calibrated prediction intervals achieved **82.56% empirical coverage** on the held-out test set, compared with a nominal coverage target of 80%.

Coverage varies across commodities. The overall result should not be interpreted as a guarantee for every commodity.

---

## Dataset

The deployed application uses a dedicated processed dataset containing:

| Attribute | Coverage |
|---|---:|
| Market-price records | 11,286,611 |
| Vegetables | 45 |
| Distinct market names | 2,167 |
| Physical market identifiers (State + District + Market) | 2,211 |
| Districts | 546 |
| States | 30 |
| Date range | 1 Jan 2021 – 31 Oct 2025 |

**Primary data source:** AGMARKNET daily agricultural market-price records, distributed through India's Open Government Data platform. The project uses a Kaggle-hosted dataset mirror.

The application dataset is a selected subset prepared for historical exploration and deployment. It should not be confused with the full upstream dataset or the broader preprocessing and analysis pipeline.

The raw source dataset is not included in this repository. Processed application data and model artifacts required to run the application are included.

---

## Methodology

The project follows a data and modelling workflow:

1. **Data auditing:** Inspect source schemas, commodity coverage, reporting patterns, and data quality.
2. **Preprocessing:** Clean and standardise records, validate prices and dates, and prepare analytical datasets.
3. **Exploratory analysis:** Study historical price behaviour, market reporting frequency, and variation across commodities and locations.
4. **Time-series experiments:** Evaluate raw-price forecasting against persistence baselines.
5. **Relative-performance modelling:** Predict future market performance relative to peer-market medians.
6. **Evaluation:** Use chronological train, validation, and test splits, with temporal embargoes to reduce leakage.
7. **Uncertainty calibration:** Evaluate and calibrate prediction intervals on held-out data.

The relative-performance target is based on a log price ratio:

`log(Future Market Price / Future Peer-Market Median)`

This formulation focuses on how a market may perform relative to comparable markets rather than attempting to predict its absolute future price.

---

## Technology Stack

| Area | Technologies |
|---|---|
| Language | Python |
| Data processing | Pandas, NumPy, DuckDB, PyArrow |
| Machine learning | LightGBM, XGBoost, Scikit-learn |
| Visualisation | Matplotlib, Seaborn, Altair |
| Application | Streamlit |
| Development | Jupyter Notebook, VS Code |

---

## Run Locally

### Requirements

- Python 3.11
- Git

### Installation

Clone the repository:

```bash
git clone https://github.com/Sachpreet-01/MandiMitra.git
cd MandiMitra
```

Create and activate a virtual environment:

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Launch the application:

```bash
streamlit run app/streamlit_app.py
```

The application will open in your browser.

---

## Repository Structure

```text
MandiMitra/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── app/
│   │   └── market_comparison_app.parquet
│   └── inference/
│       └── mandimitra_historical_market_rankings.parquet
│
├── docs/
│   ├── PROJECT_BRIEF.md
│   └── data_audit.md
│
├── models/
│   ├── lightgbm_q10.pkl
│   ├── lightgbm_q50.pkl
│   ├── lightgbm_q90.pkl
│   ├── lightgbm_regression.pkl
│   ├── xgboost_regression.pkl
│   ├── commodity_calibration.csv
│   └── metadata.json
│
├── notebooks/
│   ├── 01_data_source_audit.ipynb
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_time_series_analysis.ipynb
│   ├── 03_forecasting_series_selection.ipynb
│   ├── 04_exploratory_data_analysis.ipynb
│   ├── 05_market_comparison.ipynb
│   ├── 06_price_forecasting.ipynb
│   └── 07_relative_market_performance.ipynb
│
├── tests/
│   └── model_artifact_smoke_test.py
│
├── requirements.txt
└── README.md
```

---

## Limitations and Responsible Use

- **Historical data:** The application dataset ends on 31 October 2025. The application is not a live mandi-price service.
- **Reporting availability:** Market comparisons use eligible markets with sufficiently comparable reporting dates. Rankings do not represent every market in India on every date.
- **Changing peer groups:** The national peer-market median can change as reporting markets vary over time. This can affect a market's relative performance independently of its own price movement.
- **Market comparability:** Differences in local market structure, variety, grade, and reporting practices may affect comparisons. Extreme relative premiums should be interpreted cautiously.
- **Model coverage:** Not every vegetable or historical market series is necessarily represented in the forecasting evaluation or prediction artifacts.
- **Prediction uncertainty:** Empirical interval coverage varies across commodities and does not guarantee that every future observation will fall within a prediction interval.
- **No profit guarantee:** Predicted relative performance is an analytical estimate, not a guarantee of higher selling prices, demand, or farmer profits. Actual net realisation depends on transport, handling, commission, quality, and other costs.

MandiMitra is a research and portfolio prototype intended for historical analysis and decision support. Its predictions should be treated as supplementary information rather than financial or agricultural advice.

---

## Acknowledgements

Data attribution: AGMARKNET and the Government of India's Open Government Data platform.

The project uses a Kaggle-hosted dataset mirror. Original data provenance and applicable reuse terms should be consulted before redistribution.