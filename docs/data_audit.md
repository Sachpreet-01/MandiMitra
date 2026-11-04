# MandiMitra – Data Source Audit

## Source 1: CEDA / AGMARKNET

Portal: https://agmarknet.ceda.ashoka.edu.in/

Geography: Maharashtra

# MandiMitra – Data Source Audit (National Scope)

## Source 2: Kaggle / AGMARKNET Parquet Mirror
Geography: All India

| Commodity | Available? | Target Files | Granularity | 2025 Row Count | Status |
| --------- | ---------- | ------------ | ----------- | -------------- | ------ |
| Onion     | Yes        | 2024-2026    | Daily       | 220,991        | Loaded |

## Verification Discoveries
1. **File Format:** High-performance compressed Parquet files (`.parquet`).
2. **Key Data Columns:** `['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade', 'Arrival_Date', 'Min_Price', 'Max_Price', 'Modal_Price', 'Commodity_Code']`.
3. **Data Quality:** The data structural schemas map perfectly with our planned platform architectures.

## Additional observations

1. Are arrival quantities available?
2. Are individual mandi records available?
3. Can different varieties be distinguished?
4. What file format is available?
5. Any missing commodities or reporting gaps?

## Notes

Record anything unusual or unavailable.
