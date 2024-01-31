import pandas as pd

def reorder(flight_network):
    flight_network.sort_values(by=["DepartureDateTime"], inplace = True, kind= "stable")
    flight_network = flight_network.reset_index()
    flight_network = flight_network.iloc[:,1:]
    return flight_network


flight_network = pd.read_csv("INV-ZZ-20231208_041852.csv")
flight_network = reorder(flight_network)
flight_network.to_csv("flight_net_dataset.csv")