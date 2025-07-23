import numpy as np
import pandas as pd

batteryUsablePercentage = 0

def get_statistics(clients_data, to_grid_prices):
    NUMBER_OF_CLIENTS = len(clients_data)
    total_grid_energy_bought = 0
    total_money_paid_to_grid = 0
    total_wasted_energy = 0
    total_earnining_by_prosumers = 0
    prosumersConsumers = []
    index = []
    # figure, axis = plt.subplots(25,1,  sharex=True, figsize = (18,18))
    for i in range(NUMBER_OF_CLIENTS):
        index.append('client_' + str(i))
        total_wasted_energy += np.sum(np.maximum(0,  clients_data[i]['Excess_energy_watt']))
        total_grid_energy_bought += np.sum(-1 * np.minimum(0,  clients_data[i]['Excess_energy_watt']))
        total_money_paid_to_grid += np.sum(-1 * np.minimum(0,  clients_data[i]['Excess_energy_watt']) * clients_data[i]['Electricity_price_watt'])

        prosumersConsumers.append({'prosumer': len(clients_data[i][clients_data[i]['Excess_energy_watt']  > 0 ]),
                                'consumer': len(clients_data[i]) - len(clients_data[i][clients_data[i]['Excess_energy_watt'] > 0]),
                                'total_bought_from_grid': np.sum(-1 * np.minimum(0,  clients_data[i]['Excess_energy_watt'])),
                                'total_paid_to_grid': np.sum(-1 * np.minimum(0,  clients_data[i]['Excess_energy_watt']) * clients_data[i]['Electricity_price_watt']),
                                'total_wasted': np.sum(np.maximum(0,  clients_data[i]['Excess_energy_watt'])),
                                'money_from_grid': np.sum(np.array(to_grid_prices)  * (np.maximum(0, np.maximum(0,  clients_data[i]['Excess_energy_watt']) )))
                                })
    clientsPd = pd.DataFrame(prosumersConsumers, index=index)
    return clientsPd


def statitics_after_tradeAndShareWithBatteriesP2P(clients_data, tradedEnergy, prices, chargesDf, dischargesDf, sharedEnergy, allTradedEnergyOfSharing):
  #
  NUMBER_OF_CLIENTS = len(clients_data)
  totalBoughtPerHouseFromPeer = [0 for i in range(NUMBER_OF_CLIENTS + 1)]
  totalSoldPerHouseToPeer = [0 for i in range(NUMBER_OF_CLIENTS + 1)]
  totalMoneyEarnedFromPeer = [0 for i in range(NUMBER_OF_CLIENTS + 1)]
  totalMoneyPaidToPeer = [0 for i in range(NUMBER_OF_CLIENTS + 1)]


  totalEnergyDischarged = [np.sum([discharge['amount'] for discharge in  dischargesDf[i]]) for i in range(NUMBER_OF_CLIENTS)]
  totalEnergyCharged = [np.sum([charge['amount'] for charge in  chargesDf[i]]) for i in range(NUMBER_OF_CLIENTS)]

  boughtPerHouseFromPeer = []
  soldPerHouseToPeer = []
  moneyEarnedFromPeer = []
  moneyPaidToPeer = []

  for i in range(len(tradedEnergy)):
    tr = tradedEnergy[i]
    bTotal = [0 for i in range(NUMBER_OF_CLIENTS + 1)]
    sTotal = [0 for i in range(NUMBER_OF_CLIENTS + 1)]
    smTotal = [0 for i in range(NUMBER_OF_CLIENTS + 1)]
    bmTotal = [0 for i in range(NUMBER_OF_CLIENTS + 1)]
    for t in tr:
      totalBoughtPerHouseFromPeer[t['buyer']] += t['amount']
      totalMoneyPaidToPeer[t['buyer']] += prices[i]['price'] * t['amount']
      totalSoldPerHouseToPeer[t['seller']] += t['amount']
      totalMoneyEarnedFromPeer[t['seller']] += prices[i]['price'] * t['amount']

      bTotal[t['buyer']] += t['amount']
      sTotal[t['seller']] += t['amount']
      smTotal[t['seller']] +=  prices[i]['price'] * t['amount']
      bmTotal[t['buyer']] +=  prices[i]['price'] * t['amount']

    boughtPerHouseFromPeer.append(bTotal)
    moneyPaidToPeer.append(bmTotal)
    soldPerHouseToPeer.append(sTotal)
    moneyEarnedFromPeer.append(smTotal)

  sharedPerHouseFromPeer = []
  moneyEarnedFromSharingPerHouse = []
  moneyPaidToContract = []
  sharedByHouse = []
  totalSharedPerHouseFromPeer = [0 for i in range(NUMBER_OF_CLIENTS)]
  totalMoneyEarnedFromSharing = [0 for i in range(NUMBER_OF_CLIENTS)]
  totalSharedByHouse = [0 for i in range(NUMBER_OF_CLIENTS)]
  totalSharedAndSoldByHouse = [0 for i in range(NUMBER_OF_CLIENTS)]
  for i in range(len(allTradedEnergyOfSharing)): # trading my own energy at other's battery
    se = allTradedEnergyOfSharing[i]
    sTotal = [0 for i in range(NUMBER_OF_CLIENTS )]
    for s in se:
      if s != None:
         sTotal[s['seller']] += s['amount']
         totalMoneyEarnedFromSharing[s['seller']] += prices[i]['price'] * s['amount']
         totalSharedAndSoldByHouse[s['seller']] +=  s['amount']

    moneyEarnedFromSharingPerHouse.append(sTotal)

  for i in range(len(sharedEnergy)):
    se = sharedEnergy[i]
    sTotal = [0 for i in range(NUMBER_OF_CLIENTS )]
    mesTotal = [0 for i in range(NUMBER_OF_CLIENTS)]

    sbTotal = [0 for i in range(NUMBER_OF_CLIENTS)]
    for s in se:
      if s != None:
        totalSharedPerHouseFromPeer[s['buyer']] += s['amount']
        totalSharedByHouse[s['seller']] += s['amount'] / ( 1 - batteryUsablePercentage)
        # totalMoneyEarnedFromSharing[s['seller']] += prices[i]['price'] * s['amount'] / ( 1 - batteryUsablePercentage)
        sTotal[s['buyer']] += s['amount']
        mesTotal[s['seller']] += prices[i]['price'] * s['amount'] / ( 1 - batteryUsablePercentage)
        sbTotal[s['seller']] += s['amount'] / ( 1 - batteryUsablePercentage)
    sharedPerHouseFromPeer.append(sTotal)

    sharedByHouse.append(sbTotal)


  total_grid_energy_bought = 0
  total_money_paid_to_grid = 0
  total_wasted_energy = 0
  total_earnining_by_prosumers = 0
  prosumersConsumers = []
  index = []
  for i in range(NUMBER_OF_CLIENTS):
    index.append('client_' + str(i))
    total_wasted_energy += np.sum(np.maximum(0,  clients_data[i]['Excess_energy_watt'] - [charge['amount'] for charge in  chargesDf[i]] ))
    total_grid_energy_bought += 0 # np.sum(-1 * np.minimum(0,  clients_data[i]['Excess_energy_watt'] + [discharge['amount'] for discharge in  dischargesDf[i]] - np.array(boughtPerHouseFromPeer)[:,i]))
    total_money_paid_to_grid +=  np.sum(-1 * np.minimum(0,  clients_data[i]['Excess_energy_watt']) * clients_data[i]['Electricity_price_watt']) - np.sum(np.array(boughtPerHouseFromPeer)[:,i] * clients_data[i]['Electricity_price_watt'])

    prosumersConsumers.append({'prosumer': len(clients_data[i][clients_data[i]['Excess_energy_watt'] - [charge['amount'] for charge in  chargesDf[i]]  >= 0 ]),
                              'consumer': len(clients_data[i]) - len(clients_data[i][clients_data[i]['Excess_energy_watt'] - [charge['amount'] for charge in  chargesDf[i]]  >= 0]),
                              'total_bought_from_grid': np.sum(-1 * np.minimum(0,  clients_data[i]['Excess_energy_watt'] + [discharge['amount'] for discharge in  dischargesDf[i]])) - totalBoughtPerHouseFromPeer[i] - totalSharedPerHouseFromPeer[i],
                              'total_paid_to_grid': np.sum(-1 * np.minimum(0,  clients_data[i]['Excess_energy_watt'] + [discharge['amount'] for discharge in  dischargesDf[i]])  * clients_data[i]['Electricity_price_watt']  - np.array(boughtPerHouseFromPeer)[:,i] * clients_data[i]['Electricity_price_watt'] - np.array(sharedPerHouseFromPeer)[:,i] * clients_data[i]['Electricity_price_watt']) ,
                              'total_wasted': np.maximum(0, np.sum(np.maximum(0,  clients_data[i]['Excess_energy_watt'] - [charge['amount'] for charge in  chargesDf[i]])) - totalSoldPerHouseToPeer[i] - totalSharedByHouse[i]),
                              'money_from_grid': np.sum(np.array(to_grid_prices)  * (np.maximum(0, np.maximum(0,  clients_data[i]['Excess_energy_watt'] - [charge['amount'] for charge in  chargesDf[i]]) - np.array(soldPerHouseToPeer)[:,i]) - np.array(sharedByHouse)[:,i])),
                              'total_bought_from_peer': totalBoughtPerHouseFromPeer[i],
                              'total_sold_to_peer': totalSoldPerHouseToPeer[i],
                              'total_money_from_peer': totalMoneyEarnedFromPeer[i],
                              'total_money_to_peer':  totalMoneyPaidToPeer[i],
                              'total_shared_from_peer':  totalSharedPerHouseFromPeer[i],
                              "total_shared_by_house": totalSharedByHouse[i],
                              "total_earned_from_sharing":totalMoneyEarnedFromSharing[i],
                              'final_charge':  clients_data[i]['Battery_charge_kw'][len(clients_data[i]) - 1] / clients_data[i]['Battery_capacity_kw'][0] * 100,
                              'totalSharedAndSoldByHouse':totalSharedAndSoldByHouse[i]
                              })

  clientsPd = pd.DataFrame(prosumersConsumers, index=index)

  return clientsPd, totalEnergyDischarged, totalEnergyCharged, sharedPerHouseFromPeer, sharedByHouse