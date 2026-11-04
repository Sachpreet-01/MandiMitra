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