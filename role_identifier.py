from typing import Dict, List
import pandas as pd
import re

def classify_house_roles(all_data: Dict[int, Dict[str, pd.DataFrame]]) -> Dict[str, List[str]]:
    """
    Classifies each house as a prosumer or consumer based on solar panel presence.

    Args:
        all_data (dict): Nested dictionary of folder data from data_loader.

    Returns:
        dict: Dictionary with keys 'prosumers' and 'consumers' listing unique house identifiers.
    """
    prosumers = set()
    consumers = set()

    for folder_number, folder_data in all_data.items():
        for house_name, df in folder_data.items():
            # Only process entries that look like 'folderN_houseM'
            if not re.match(r"folder\d+_house\d+", house_name):
                continue

            if df["solar_cells_house"].gt(0).any():
                prosumers.add(house_name)
            else:
                consumers.add(house_name)

    # Ensure no overlap — consumer if always 0, prosumer if ever > 0
    prosumers = prosumers - consumers

    return {
        "prosumers": sorted(prosumers),
        "consumers": sorted(consumers)
    }
