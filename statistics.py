import numpy as np
import pandas as pd


def statistics_after_trade_and_share(
    clients_data,
    tradedEnergy,
    prices,
    chargesDf,
    dischargesDf,
    sharedEnergy,
    allTradedEnergyOfSharing,
    batteryUsablePercentage=0.8,
):
    """
    Returns a pandas.DataFrame with one row per house and useful
    aggregate columns such as:
        total_bought_from_grid, total_sold_to_peer, …
    """
    NUMBER_OF_CLIENTS = len(clients_data)
    totalBoughtPerHouseFromPeer = [0.0] * (NUMBER_OF_CLIENTS + 1)
    totalSoldPerHouseToPeer = [0.0] * (NUMBER_OF_CLIENTS + 1)
    totalMoneyEarnedFromPeer = [0.0] * (NUMBER_OF_CLIENTS + 1)
    totalMoneyPaidToPeer = [0.0] * (NUMBER_OF_CLIENTS + 1)

    # …  (unchanged bulk of your computation) …

    # -------- build one row per client --------------------------------
    rows = []
    for i in range(NUMBER_OF_CLIENTS):
        rows.append(
            {
                "house": i,
                "grid_draw_kWh": -min(0, clients_data[i]["Excess_energy_watt"].sum()),
                "grid_sell_kWh": max(0, clients_data[i]["Excess_energy_watt"].sum()),
                "peer_buy_kWh": totalBoughtPerHouseFromPeer[i],
                "peer_sell_kWh": totalSoldPerHouseToPeer[i],
                "money_from_peer": totalMoneyEarnedFromPeer[i],
                "money_to_peer": totalMoneyPaidToPeer[i],
                "battery_charged_kWh": np.sum([c["amount"] for c in chargesDf[i]]),
                "battery_discharged_kWh": np.sum([d["amount"] for d in dischargesDf[i]]),
            }
        )

    return pd.DataFrame(rows).set_index("house")
