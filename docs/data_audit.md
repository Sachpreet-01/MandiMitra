# MandiMitra – Data Audit Report

## 1. Project Data Scope

**Project:** MandiMitra  
**Geographic Scope:** India  
**Initial Commodities:** Onion, Tomato, Soybean, Cotton, Grapes, Pomegranate  
**Primary Data:** Historical agricultural market prices from AGMARKNET.

MandiMitra aims to analyse historical agricultural market prices across Indian wholesale markets and develop price intelligence and forecasting capabilities.

## 2. Data Sources and Provenance

### Source 1: CEDA / AGMARKNET

**Portal:** https://agmarknet.ceda.ashoka.edu.in/  
**Role:** Additional research and potential data access source.

### Source 2: Kaggle – Daily Commodity Prices India (2001–2026)

**Dataset:** https://www.kaggle.com/datasets/khandelwalmanas/daily-commodity-prices-india

**DOI:** https://doi.org/10.34740/kaggle/dsv/15888319

**Original Data Authority:** AGMARKNET  
**Government Publisher:** Ministry of Agriculture and Farmers Welfare, Government of India  
**Primary Source:** https://www.data.gov.in/resource/variety-wise-daily-market-prices-data-commodity  
**Declared Source License:** Government Open Data License – India (GODL-India)  
**Geographic Coverage:** India

The Kaggle dataset consolidates yearly agricultural market-price files into CSV and Parquet formats. The uploader reports standardisation, duplicate removal and data validation. These claims are treated as source documentation and are independently checked during this project.

## 3. Dataset Structure

**Local Format:** Parquet  
**Number of Annual Files:** 26  
**Total Records:** 75,984,017  
**Historical Coverage:** 2001–2026 (2026 is partial)

### Columns

| Column | Description |
|---|---|
| State | State where the market is located |
| District | District where the market is located |
| Market | Wholesale market / mandi |
| Commodity | Agricultural commodity |
| Variety | Commodity variety |
| Grade | Quality grade |
| Arrival_Date | Market price recording date |
| Min_Price | Minimum price (INR/quintal) |
| Max_Price | Maximum price (INR/quintal) |
| Modal_Price | Modal price (INR/quintal) |
| Commodity_Code | Commodity identifier |

**Important limitation:** Arrival quantity is not present in the current Parquet schema.

## 4. Six-Commodity Coverage

The following source-name mapping is used:

| Project Commodity | Source Commodity Name |
|---|---|
| Onion | Onion |
| Tomato | Tomato |
| Soybean | Soyabean |
| Cotton | Cotton |
| Grapes | Grapes |
| Pomegranate | Pomegranate |

### Historical Coverage Summary

| Commodity | Records | States | Markets | First Date | Last Date |
|---|---:|---:|---:|---|---|
| Cotton | 936,032 | 24 | 1,437 | 2001-04-17 | 2026-04-20 |
| Grapes | 369,784 | 30 | 971 | 2001-01-27 | 2026-04-20 |
| Onion | 3,168,490 | 32 | 3,180 | 2001-01-10 | 2026-04-20 |
| Pomegranate | 572,365 | 25 | 854 | 2001-11-17 | 2026-04-20 |
| Soybean | 1,192,838 | 25 | 1,446 | 2001-03-12 | 2026-04-20 |
| Tomato | 2,818,577 | 32 | 3,033 | 2001-02-08 | 2026-04-20 |

**Total records across the six selected commodities:** 9,058,086.

Market counts are based on distinct source market names and may include naming inconsistencies.

## 5. Data Quality Findings

### 5.1 Duplicate Records

A duplicate audit identified 219 duplicate groups involving 438 rows.

The 2025 and 2026 files also share a date (30 December 2025). One exact cross-file duplicate was confirmed.

**Decision:** Preserve original files. Apply deduplication only in the processed dataset after checking whether duplicate records are genuinely identical.

### 5.2 Price Quality

The audit identified:
- No negative prices in the six selected commodities.
- Records where minimum price exceeds maximum price.
- Records where modal price falls outside the minimum–maximum range.
- Missing modal prices.
- Historical records where minimum and maximum prices are both zero despite a positive modal price.

These issues are more prominent in older records.

**Proposed treatment:**
- Preserve valid positive modal prices even when minimum and maximum prices are zero.
- Flag invalid price ranges.
- Flag modal prices outside the reported range.
- Exclude records with missing or invalid modal prices from forecasting experiments unless a justified treatment is established.
- Do not silently delete suspicious observations.

### 5.3 Reporting Continuity

Reporting history varies significantly between markets.

The strongest markets for the selected commodities have thousands of distinct reporting dates. However, some markets exhibit gaps exceeding several months or even a year.

**Decision:** Forecasting eligibility will depend on historical coverage and data continuity. Not every mandi will automatically receive an individual forecasting model.

## 6. Dataset Limitations

1. Arrival quantity is absent from the current dataset.
2. Market names may contain spelling or naming variations.
3. The Kaggle archive is a third-party consolidated mirror, not an untouched government download.
4. The 2026 file contains partial-year data.
5. Historical reporting gaps vary by commodity and market.
6. Source metadata and actual archive coverage have some inconsistencies.
7. Duplicate records require independent validation.

## 7. Preliminary Conclusion

The archive provides substantial multi-state historical price data for all six selected commodities.

It is suitable for proceeding to a controlled preprocessing stage, subject to continued validation of provenance, duplicates, price quality and market identifiers.

The initial project scope remains India-wide. Maharashtra may be used as a detailed regional case study, without restricting the overall platform to Maharashtra.

## 8. Next Stage

Develop a reproducible, memory-efficient preprocessing pipeline using DuckDB.

The pipeline will:
- Read original Parquet files without modifying them.
- Standardise commodity names.
- Validate dates and prices.
- Identify and handle duplicates.
- Generate quality flags.
- Export validated processed data for exploratory analysis and forecasting.


## Final Preprocessing and Processed Dataset

### 1. Preprocessing Summary

The six selected commodities were extracted from the original 26 annual Parquet files and processed using DuckDB.

The preprocessing pipeline performed the following operations:

- Selected Onion, Tomato, Soyabean, Cotton, Grapes and Pomegranate.
- Standardised the project commodity name `Soyabean` to `Soybean`.
- Removed leading and trailing whitespace from text fields.
- Converted `Arrival_Date` into a proper DATE column using date parsing.
- Identified and removed exact duplicate records from the processed dataset.
- Preserved original price values rather than automatically correcting suspicious observations.
- Created data-quality flags for missing prices, zero-price ranges, invalid price ranges and modal prices outside reported ranges.
- Created a basic forecasting eligibility flag.

The original raw Parquet files were left unchanged.

### 2. Final Processed Dataset

The processed dataset contains **9,057,867 records** across six commodities.

| Commodity | Records | States | Market Identities | Varieties |
|---|---:|---:|---:|---:|
| Cotton | 936,024 | 24 | 1,437 | 1,459 |
| Grapes | 369,772 | 30 | 971 | 980 |
| Onion | 3,168,401 | 32 | 3,180 | 3,223 |
| Pomegranate | 572,345 | 25 | 854 | 859 |
| Soybean | 1,192,827 | 25 | 1,446 | 1,457 |
| Tomato | 2,818,498 | 32 | 3,033 | 3,075 |
| **Total** | **9,057,867** | | | |

Historical coverage in the processed dataset:

- First observation: 10 January 2001
- Last observation: 20 April 2026
- Number of commodities: 6
- Geographic coverage: Multiple states across India

Market identities are distinguished using the combination of State, District and Market rather than market name alone.

### 3. Duplicate Removal

The duplicate audit identified:

| Measure | Count |
|---|---:|
| Exact duplicate groups | 219 |
| Rows in duplicate groups | 438 |
| Excess duplicate rows removed | 219 |
| Largest duplicate group size | 2 |

Exact duplicates were identified using all original source columns, including commodity, location, variety, grade, date, prices and commodity code.

Only repeated identical records were removed. The original raw files were preserved.

### 4. Final Data Quality Assessment

The final validation was performed on the saved processed Parquet dataset.

| Quality Check | Records |
|---|---:|
| Total processed records | 9,057,867 |
| Invalid dates | 0 |
| Invalid modal prices (NULL, non-finite or non-positive) | 0 |
| Invalid price ranges (Minimum Price greater than Maximum Price) | 5,059 |
| Modal prices outside reported non-zero ranges | 3,198 |
| Zero-range records with positive modal prices | 219,113 |
| Forecast-eligible records | 9,057,867 |

#### Interpretation of Data Quality Findings

**Invalid price ranges:** 5,059 records have a minimum price greater than their maximum price. These observations are retained and flagged rather than silently corrected.

**Modal prices outside reported ranges:** 3,198 records have modal prices below the reported minimum or above the reported maximum, where both range boundaries are positive. These discrepancies may indicate source-data inconsistencies and require further investigation.

**Zero-range records:** 219,113 records have both minimum and maximum prices equal to zero while the modal price is positive. These observations retain a potentially useful modal price, but their reported price ranges are unavailable.

These quality categories are not mutually exclusive. A record can trigger more than one flag, so their counts must not be added together to calculate a total number of problematic records.

### 5. Forecasting Eligibility

A record is currently considered eligible for basic modal-price forecasting when:

- Its arrival date is valid.
- Its modal price is present.
- Its modal price is finite.
- Its modal price is greater than zero.

Using this definition, all 9,057,867 processed records are currently forecast-eligible.

However, this is a preliminary eligibility criterion, not a guarantee of data accuracy. Price-range violations and suspicious observations will be investigated during time-series analysis and model preparation.

No automatic interpolation or correction of suspicious prices has been performed.

### 6. Processed Output

The cleaned dataset was exported successfully in compressed Parquet format.

**Output file:** `data/processed/mandi_prices_cleaned.parquet`

| Property | Value |
|---|---|
| Format | Apache Parquet |
| Compression | Zstandard (ZSTD) |
| File size | 94.52 MB |
| Records | 9,057,867 |
| Commodities | 6 |

The saved file was queried again to verify its record count, commodity count, date coverage and forecasting eligibility.

### 7. Remaining Limitations

The following limitations remain:

1. The selected dataset does not contain arrival quantities. Therefore, arrival-volume and supply-based analysis cannot be performed using this dataset alone.
2. Market reporting is not necessarily continuous. Weekends, holidays, seasonal trading patterns and reporting interruptions can create gaps.
3. Market names and reporting practices may vary across years and locations.
4. Some observations contain suspicious price ranges or modal-price inconsistencies.
5. The 2026 dataset is partial, with observations ending on 20 April 2026 in the supplied archive.

These limitations will be considered during market selection, time-series construction and forecasting evaluation.

### 8. Conclusion

The initial preprocessing stage has produced a reusable dataset containing approximately 9.06 million records across six agricultural commodities and multiple Indian states.

Exact duplicates have been removed, commodity names standardised, dates parsed, and price-quality flags added. The processed dataset has been successfully saved and independently validated.

The next stage will focus on time-series analysis, including daily market-price aggregation, reporting continuity, seasonal behaviour and identification of suitable markets for forecasting.


## Time-Series Coverage and Reporting Continuity

### 1. Objective

A historical price dataset is not automatically suitable for time-series forecasting. Forecasting requires sufficiently long, reasonably continuous and recent observations. This audit examines reporting frequency, historical coverage, reporting gaps and data recency across the six selected commodities.

The analysis uses the cleaned dataset, `data/processed/mandi_prices_cleaned.parquet`, containing 9,057,867 records across Cotton, Grapes, Onion, Pomegranate, Soybean and Tomato.

A price series is defined by the combination of:

- Project commodity
- State
- District
- Market
- Variety
- Grade

This distinction is important because different varieties and grades can have materially different prices within the same market.

### 2. Daily Reporting Patterns

Most market-days contain one observation. However, some market-days contain multiple observations, particularly for Cotton and Grapes.

| Commodity | Market-day observations | Average | Median | 95th percentile | Maximum |
|---|---:|---:|---:|---:|---:|
| Cotton | 850,144 | 1.10 | 1 | 2 | 7 |
| Grapes | 332,029 | 1.11 | 1 | 2 | 9 |
| Onion | 3,021,329 | 1.05 | 1 | 1 | 12 |
| Pomegranate | 557,537 | 1.03 | 1 | 1 | 6 |
| Soybean | 1,116,993 | 1.07 | 1 | 1 | 5 |
| Tomato | 2,785,000 | 1.01 | 1 | 1 | 6 |

Inspection of market-days with multiple observations showed that these records frequently represented different varieties or grades rather than duplicate entries.

For example, Grapes prices at Azadpur on 25 December 2006 ranged from ₹2,100 to ₹4,500 across the recorded variety-grade combinations. This demonstrates why unrelated varieties and grades should not be indiscriminately combined into one price series.

### 3. Variety and Grade Analysis

| Commodity | Unique varieties | Unique grades | Unique variety-grade pairs |
|---|---:|---:|---:|
| Cotton | 109 | 11 | 326 |
| Grapes | 10 | 13 | 62 |
| Onion | 26 | 14 | 178 |
| Pomegranate | 3 | 12 | 23 |
| Soybean | 7 | 11 | 40 |
| Tomato | 6 | 14 | 62 |

Grade-level comparisons also showed that prices frequently differed between grades, particularly for Grapes, Onion, Pomegranate and Tomato.

Consequently, the initial forecasting design retains variety and grade as part of the series identity. Their usefulness will be evaluated during later modelling rather than assuming that they are irrelevant.

### 4. Historical Series Coverage

The dataset contains many market-variety-grade combinations, but their reporting histories vary substantially.

| Commodity | Total series | Median reporting days | Series with at least 365 reporting days | Series with at least 1,000 reporting days |
|---|---:|---:|---:|---:|
| Cotton | 4,767 | 12 | 659 | 288 |
| Grapes | 2,560 | 13 | 292 | 93 |
| Onion | 7,488 | 30 | 1,794 | 997 |
| Pomegranate | 1,762 | 22 | 380 | 209 |
| Soybean | 3,934 | 24 | 827 | 347 |
| Tomato | 6,458 | 36 | 1,609 | 903 |

The distribution is highly uneven. Onion and Tomato have larger pools of long-running series, while Grapes has fewer long-history series.

A high observation count alone does not establish suitability for forecasting. Reporting gaps and the date of the latest observation must also be considered.

### 5. Reporting Gaps and Continuity

Reporting-gap analysis was conducted for series with at least 365 reporting days.

Although the median interval between consecutive reporting dates was generally one day, many series contained substantial interruptions. Some long-running series had gaps exceeding one year.

This means that a series can have a large historical observation count while still being unsuitable for uninterrupted daily forecasting.

Missing reporting dates are not interpreted as zero prices. Automatic interpolation across long gaps is also avoided because it could introduce artificial price movements and misleading continuity.

### 6. Data Recency

The latest date in the processed dataset is **20 April 2026**.

The recency analysis found substantial variation between commodities. Among series with at least 365 reporting days:

- Onion and Tomato had the largest pools of recently reporting series.
- Pomegranate had comparatively better median recency among its eligible series.
- Cotton and Soybean had no eligible long-history series reporting within 90 days of the dataset end.
- Many long-running series had stopped reporting well before the final dataset date.

The dataset is therefore suitable for historical analysis and retrospective forecasting experiments, but it cannot be presented as a source of live market prices as of November 2026.

### 7. November 2025 Reporting Disruption

A pronounced reporting reduction was identified across all six commodities in November 2025.

The comparison below shows the reduction in average active markets between October and November 2025.

| Commodity | October average active markets | November average active markets | Reduction |
|---|---:|---:|---:|
| Cotton | 68.10 | 45.00 | 33.92% |
| Grapes | 26.84 | 16.20 | 39.64% |
| Onion | 417.61 | 158.50 | 62.05% |
| Pomegranate | 104.42 | 44.17 | 57.70% |
| Soybean | 134.84 | 84.00 | 37.70% |
| Tomato | 417.16 | 154.71 | 62.91% |

Reporting days also fell sharply:

| Commodity | October 2025 | November 2025 | December 2025 |
|---|---:|---:|---:|
| Cotton | 31 | 5 | 31 |
| Grapes | 31 | 5 | 29 |
| Onion | 31 | 8 | 31 |
| Pomegranate | 31 | 6 | 29 |
| Soybean | 31 | 5 | 29 |
| Tomato | 31 | 7 | 31 |

The reduction occurred across multiple commodities and markets. For Grapes, Onion, Pomegranate and Tomato, numerous individual market series also showed interruptions of approximately five months between November 2025 and April 2026.

This is evidence of a widespread reporting discontinuity in the dataset. However, the available data does not establish its cause. It should not be described as a confirmed failure of a particular government reporting system or as proof that markets stopped trading.

### 8. Market-Type Considerations

Some records are associated with Uzhavar Sandhais, which are farmers' markets. These should not automatically be treated as equivalent to conventional wholesale agricultural markets.

Market-type classification will therefore be considered during forecasting-series selection. Uncertain classifications should be reviewed rather than inferred solely from market names.

### 9. Implications for Forecasting

The findings establish several requirements for the forecasting pipeline:

1. Retain commodity, state, district, market, variety and grade when identifying individual price series.
2. Evaluate historical observation count, latest reporting date and largest reporting gap together.
3. Avoid treating missing observations as zero prices.
4. Do not automatically interpolate across long reporting interruptions.
5. Evaluate market type before including a series in wholesale-market forecasting.
6. Use chronological training, validation and test periods rather than random train-test splitting.
7. Report forecast performance and reliability separately for each commodity and, where appropriate, each market series.
8. Clearly communicate the dataset's latest available date and limitations.

### 10. Conclusion

The dataset provides substantial historical coverage across six agricultural commodities and thousands of Indian markets. However, its suitability for forecasting varies significantly between individual market-variety-grade series.

The November 2025 reporting disruption, long inactive periods and uneven data recency make series-level screening essential.

The next stage will identify suitable forecasting series using explicit historical coverage, continuity, recency and market-type criteria. Only series that pass these checks will proceed to forecasting experiments.


## 7. Forecasting Series Selection and Vegetable Expansion

**Notebook:** `notebooks/03_forecasting_series_selection.ipynb`

### Objective

Evaluate the historical coverage, reporting recency and forecasting suitability of agricultural commodity market series. Expand the initial six-commodity scope to include a broader range of vegetables.

### Vegetable Inventory

The original 26 annual Parquet files were scanned to identify vegetable commodities.

- 389 distinct commodity names were found in the source files.
- 46 commodities were provisionally classified as vegetables.
- 8 commodities were excluded from the vegetable scope.
- 2 commodities were marked ambiguous for classification review.

The vegetable classification is based on commodity names and should be treated as a project-level classification rather than an official agricultural taxonomy.

### Historical Coverage

Several vegetables have substantial historical coverage, including:

| Commodity | Series with 365+ reporting days |
|---|---:|
| Potato | 1,806 |
| Onion | 1,794 |
| Tomato | 1,609 |
| Brinjal | 1,389 |
| Green Chilli | 1,190 |
| Bhindi | 1,043 |
| Cauliflower | 1,031 |
| Cabbage | 938 |
| Bitter gourd | 819 |
| Pumpkin | 804 |
| Bottle gourd | 792 |
| Carrot | 748 |
| Ginger (Green) | 653 |

A series represents a unique combination of commodity, state, district, market, variety and grade.

### Reporting Recency and Continuity

The analysis identified a major reporting decline around November 2025.

Although many series contain observations after November 6, 2025, very few combine at least 365 historical reporting days with at least 30 post-November reporting days.

The corrected screening found:
- Potato: 1 such established series.
- Onion, Tomato, Brinjal, Green Chilli and the other screened vegetables: 0 under this particular threshold.

No screened vegetable had a series with 90 or more post-November reporting days.

These findings indicate that historical coverage alone does not establish forecasting suitability.

### Project Decisions

1. Expand historical market intelligence to the identified vegetable commodities.
2. Preserve market, variety and grade as distinct analytical dimensions.
3. Select forecasting series individually based on history, continuity and validation performance.
4. Treat forecasting as experimental until adequate data continuity and freshness are established.
5. The latest available observation in the dataset is April 20, 2026. The dataset is not suitable for presenting live November 2026 market prices.

### Conclusion

The dataset supports broad historical agricultural market analysis across multiple vegetables. Forecasting is possible as a historical experiment, but reliable operational forecasting requires more recent and sufficiently continuous observations.