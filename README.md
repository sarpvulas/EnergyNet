#!/bin/bash

README_PATH="README.md"

cat << 'EOF' >> $README_PATH

---

## 📊 Data Description

### 1. `house*.csv` Files

Each `Generated Data - N` folder contains time-series energy data for multiple houses. The format is as follows:

| Column Name                            | Description                                                                 |
|---------------------------------------|-----------------------------------------------------------------------------|
| `datetime`                            | Timestamp of the measurement (hourly, ISO format with timezone)            |
| `solar_cells_house`                   | Number of solar cells installed in the house                               |
| `solar_generation_watt_by_cell_house` | Power generated **per cell** in watts                                      |
| `solar_hourly_watt_generation_house`  | Total power generation by the house in that hour (cells × cell output)     |
| `hourly_load_kw_house`                | Energy consumption by the house in kilowatts for that hour                |
| `hourly_load_watt_house`              | Same load in watts (1 kW = 1000 W)                                         |
| `Battery_capacity_kw`                 | Total battery capacity of the house in kilowatts                          |
| `Battery_charge_kw`                   | Current battery charge at the timestamp in kilowatts                      |
| `Excess_energy_watt`                  | Net excess energy in watts (can be negative if underpowered)              |
| `Electricity_price_watt`              | Price of electricity per watt at that time (for buying/selling decisions) |

> 🧠 Use case: This data is used to simulate individual house generation, consumption, battery storage, and trading behavior over time.

---

### 2. `monthly-balances.csv`

This file exists inside each folder and gives a **summary of net energy balances per house over the month**.

- **Rows**: Each row corresponds to a house (e.g. row index 0 → house0, 1 → house1, etc.)
- **Columns**: 49 columns, assumed to represent monthly energy balance values per time bin (e.g. each column = daily/hourly summary slot)

> 💡 These balances can be used to evaluate energy deficits, surpluses, or optimization results at the end of a simulation period.

---

\`\`\`
data/
└── icc_combined/
    ├── Generated Data - 1/
    │   ├── house1.csv
    │   ├── house2.csv
    │   └── monthly-balances.csv
    ├── Generated Data - 2/
    │   ├── house1.csv
    │   └── ...
    └── ...
\`\`\`

EOF

echo "✅ Data description added to $README_PATH."
