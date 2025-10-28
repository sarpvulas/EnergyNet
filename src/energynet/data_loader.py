import os
import re
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Tuple
from .config import BASE_DIR

# -- Constants --
REQUIRED_COLUMNS = {
    "house": [
        "datetime", "solar_cells_house", "solar_generation_watt_by_cell_house",
        "solar_hourly_watt_generation_house", "hourly_load_kw_house",
        "hourly_load_watt_house", "Battery_capacity_kw", "Battery_charge_kw",
        "Excess_energy_watt", "Electricity_price_watt"
    ]
}

def get_project_directory() -> str:
    """
    Get the parent directory of the 'energynet' package.
    """
    utility_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(utility_dir)
    return project_dir


def get_data_directory() -> str:
    """
    Get the path to the 'data/icc_combined' directory.
    """
    return os.path.join(get_project_directory(), "data", "icc_combined")


def load_csv_with_datetime(file_path: str, datetime_col: str = "datetime") -> pd.DataFrame:
    """
    Load CSV file and parse datetime column if present.
    """
    return pd.read_csv(file_path, parse_dates=[datetime_col])


def validate_data_format(data: Dict[str, pd.DataFrame]) -> None:
    """
    Ensure each house file contains the required columns.
    """
    for house_name, df in data.items():
        if re.match(r"folder\d+_house\d+", house_name):
            missing_cols = [col for col in REQUIRED_COLUMNS["house"] if col not in df.columns]
            if missing_cols:
                raise ValueError(f"{house_name} is missing columns: {missing_cols}")


def get_all_house_names(data: Dict[str, pd.DataFrame]) -> List[str]:
    """
    Get all house DataFrame keys from the loaded data.
    """
    return [name for name in data if re.match(r"folder\d+_house\d+", name)]


def read_data_from_generated_folder(folder_number: int) -> Dict[str, pd.DataFrame]:
    """
    Read CSVs from a specific Generated Data folder.
    """
    folder_name = f"Generated Data - {folder_number}"
    folder_path = os.path.join(BASE_DIR, folder_name)

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder '{folder_path}' does not exist.")

    data: Dict[str, pd.DataFrame] = {}

    # Read all house*.csv files
    for file_name in os.listdir(folder_path):
        if re.match(r"house\d+\.csv", file_name):
            file_path = os.path.join(folder_path, file_name)
            try:
                df_name = f"folder{folder_number}_" + file_name.replace(".csv", "")
                data[df_name] = load_csv_with_datetime(file_path)
                print(f"Loaded: {file_name}")
            except Exception as e:
                print(f"Error reading '{file_name}': {e}")

    # Read monthly-balances.csv
    monthly_balances_path = os.path.join(folder_path, "monthly-balances.csv")
    if os.path.exists(monthly_balances_path):
        try:
            mb_df = pd.read_csv(monthly_balances_path)
            mb_df.set_index(mb_df.columns[0], inplace=True)
            mb_df.index.name = "house_id"
            data["monthly_balances"] = mb_df
            print("Loaded: monthly-balances.csv")
        except Exception as e:
            print(f"Error reading 'monthly-balances.csv': {e}")
    else:
        raise FileNotFoundError("monthly-balances.csv is missing.")

    if not data:
        raise ValueError("No CSV files found in the target folder.")

    validate_data_format(data)

    print(f"Data loaded successfully from '{folder_name}'!")
    return data


def read_all_generated_data(verbose: bool = True, fraction: float = 1.0) -> Tuple[Dict[int, Dict[str, pd.DataFrame]], Dict[int, pd.DataFrame]]:
    """
    Read a fraction of all 'Generated Data - *' folders.

    Args:
        verbose (bool): Whether to show progress bar and messages.
        fraction (float): Fraction of folders to load (0.0 to 1.0).

    Returns:
        Tuple[dict, dict]: all_data, monthly_balances
    """
    all_data: Dict[int, Dict[str, pd.DataFrame]] = {}
    monthly_balances: Dict[int, pd.DataFrame] = {}

    folder_names = sorted(os.listdir(BASE_DIR))
    folder_matches = [re.match(r"Generated Data - (\d+)", name) for name in folder_names]
    folder_numbers = [int(m.group(1)) for m in folder_matches if m]

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1")

    num_to_load = max(1, int(len(folder_numbers) * fraction))
    folder_numbers = folder_numbers[:num_to_load]

    iterator = tqdm(folder_numbers, desc="Loading Folders", disable=not verbose)

    for folder_number in iterator:
        try:
            if verbose:
                print(f"\nReading folder: Generated Data - {folder_number}")
            folder_data = read_data_from_generated_folder(folder_number)
            # Only keep house data (ignore 'monthly_balances')
            all_data[folder_number] = {
                k: v for k, v in folder_data.items() if re.match(r"folder\d+_house\d+", k)
            }
            if "monthly_balances" in folder_data:
                monthly_balances[folder_number] = folder_data["monthly_balances"]
        except Exception as e:
            print(f"Failed to load folder {folder_number}: {e}")

    if not all_data:
        raise RuntimeError("No generated data folders were successfully loaded.")

    return all_data, monthly_balances


if __name__ == "__main__":
    all_data, monthly_balances = read_all_generated_data(verbose=True, fraction=0.1)

    print(f"\n✅ Loaded {len(all_data)} folders.")
    print(f"📊 Monthly balances available for {len(monthly_balances)} folders.")

    # Show sample house key from folder 1
    if 1 in all_data:
        sample_key = next((k for k in all_data[1] if "house1" in k), None)
        if sample_key:
            print(f"\n📁 Sample from Generated Data - 1, {sample_key}.csv:")
            print(all_data[1][sample_key].head())

    if 1 in monthly_balances:
        print("\n📊 Monthly balances (folder 1):")
        print(monthly_balances[1].head())
