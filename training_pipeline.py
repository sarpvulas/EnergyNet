import pandas as pd
from typing import Dict, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from data_loader import read_all_generated_data
from role_identifier import classify_house_roles
from models.EnergyReturnLSTM import EnergyReturnLSTM
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from tabulate import tabulate

BATTERY_CRITICAL_THRESHOLD = 0.2  # 20% of battery capacity
LOOKAHEAD_HOURS = 12

def estimate_required_return(df: pd.DataFrame, verbose: bool = True) -> List[Dict]:
    results = []
    df = df.sort_values("datetime").reset_index(drop=True)

    for i in range(len(df) - LOOKAHEAD_HOURS):
        current = df.iloc[i]
        battery_capacity = current["Battery_capacity_kw"] * 1000  # convert to watt
        battery_charge = current["Battery_charge_kw"] * 1000

        future_window = df.iloc[i+1:i+1+LOOKAHEAD_HOURS]
        net_generation = (
            future_window["solar_hourly_watt_generation_house"]
            - future_window["hourly_load_watt_house"]
        )

        running_battery = battery_charge
        t_required = None
        e_required = 0

        for h, net in enumerate(net_generation):
            running_battery += net
            running_battery = max(0, min(running_battery, battery_capacity))

            if running_battery / battery_capacity < BATTERY_CRITICAL_THRESHOLD:
                t_required = h + 1  # hours ahead
                e_required = (BATTERY_CRITICAL_THRESHOLD * battery_capacity) - running_battery
                break

        if t_required is None:
            continue

        results.append({
            "timestamp": current["datetime"],
            "battery_capacity": battery_capacity,
            "battery_charge": battery_charge,
            "grid_price": current["Electricity_price_watt"],
            "solar_forecast": df.iloc[i+1:i+1+LOOKAHEAD_HOURS]["solar_hourly_watt_generation_house"].sum(),
            "load_forecast": df.iloc[i+1:i+1+LOOKAHEAD_HOURS]["hourly_load_watt_house"].sum(),
            "T_required_return": t_required,
            "E_required_return": e_required
        })

    if verbose:
        print(f"🔍 Extracted {len(results)} examples from DataFrame")

    return results

def train_lstm_model(df: pd.DataFrame, verbose: bool = True) -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if verbose:
        print(f"✅ Using device: {device}")

    X = df[[
        "battery_capacity",
        "battery_charge",
        "grid_price",
        "solar_forecast",
        "load_forecast"
    ]].values
    y = df[["E_required_return", "T_required_return"]].values

    if verbose:
        print("\n📋 Feature Matrix (X) and Target (y) Samples:")
        preview_df = pd.DataFrame(X, columns=[
            "battery_capacity",
            "battery_charge",
            "grid_price",
            "solar_forecast",
            "load_forecast"
        ])
        preview_df[["E_required_return", "T_required_return"]] = y
        print(tabulate(preview_df.head(10), headers="keys", tablefmt="fancy_grid"))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = EnergyReturnLSTM(input_size=5, hidden_size=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.L1Loss()

    model.train()
    for epoch in range(20):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output_energy, output_time = model(batch_X)
            loss_energy = criterion(output_energy, batch_y[:, 0])
            loss_time = criterion(output_time, batch_y[:, 1])
            loss = loss_energy + loss_time
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if verbose:
            print(f"Epoch {epoch+1}/20 - Loss: {total_loss:.4f}")

    model.eval()
    with torch.no_grad():
        pred_energy, pred_time = model(X_test_tensor)
        predictions = torch.stack([pred_energy, pred_time], dim=1).cpu().numpy()

    y_test_np = y_test_tensor.cpu().numpy()
    energy_mae = mean_absolute_error(y_test_np[:, 0], predictions[:, 0])
    energy_r2 = r2_score(y_test_np[:, 0], predictions[:, 0])
    time_mae = mean_absolute_error(y_test_np[:, 1], predictions[:, 1])
    time_r2 = r2_score(y_test_np[:, 1], predictions[:, 1])

    if verbose:
        print(f"\n⚡ Energy Model - MAE: {energy_mae:.2f} watts | R²: {energy_r2:.3f}")
        print(f"⏳ Time Model   - MAE: {time_mae:.2f} hours | R²: {time_r2:.3f}")

def main(verbose: bool = True):
    all_data, _ = read_all_generated_data(verbose=verbose, fraction=0.05)

    house_roles = classify_house_roles(all_data)
    prosumers = house_roles["prosumers"]

    if verbose:
        print(f"\n📫 Prosumers: {prosumers}")

    all_examples = []
    for folder_data in all_data.values():
        for house_name in prosumers:
            if house_name in folder_data:
                df = folder_data[house_name]
                house_examples = estimate_required_return(df, verbose=verbose)
                all_examples.extend(house_examples)

    if verbose:
        print(f"\n✅ Extracted {len(all_examples)} training examples.")

    training_df = pd.DataFrame(all_examples)
    if verbose:
        print(training_df.head())

    train_lstm_model(training_df, verbose=verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--silent", action="store_true", help="Run training without verbose output")
    args = parser.parse_args()

    main(verbose=not args.silent)