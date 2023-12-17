from pnr_reaccomodation import *
from classical_pathfinding import * 

def main(*disruptions, INVENTORY_FILE="INV-ZZ-20231208_041852.csv", PNR_FILE = "PNRB-ZZ-20231208_062017.csv", PASSENGER_LIST = "PNRP-ZZ-20231208_111136.csv", TOKEN = 'DEV-6ddf205adb6761bc0018a65f2496245457fe977f'):
    flight_network = pd.read_csv(INVENTORY_FILE)
    PNR_list = pd.read_csv(PNR_FILE)
    passenger_details = pd.read_csv(PASSENGER_LIST)
    disruptions = list(disruptions)
    
    # Create Object and run solve function
    solver = pathfind_recursion(flight_network, disruptions, scoring_criteria= scoring_criteria_Flights, toggle = scoring_criteria_Flights_toggle, verbose= 0, stopovers= 2)
    solutions, alphas, sources, destinations = solver.solve()
    
    # Create Object and run solve function for PNRs
    PNR_list = impacted_PNR(scoring_criteria_PNRs, PNR_list, passenger_details, scoring_criteria_PNRs_toggle, flight_network, disruptions)
    impacted_pax, PNRs, matrix_solved = PNR_list.solve()
    
    for disrupt in disruptions:
        print(f"Solving for disruption of {disrupt} inventory ID flight")
        paths = solutions[disrupt]
        alpha = alphas[disrupt]
        Passengers_flight = PNRs[disrupt]
        scores = matrix_solved[disrupt]
        abs_alpha = []
        
        PNR = []
        row_index_list = Passengers_flight.index.tolist()
        for i in range(len(Passengers_flight)):
            PNR.append(Passenger(int(Passengers_flight['PAX_CNT'].iloc[i]), row_index_list[i]))
        
        
        for i in range(len(paths)):
            abs_alpha.append(paths[i][-1])
            paths[i] = paths[i][:-1]
            for j in range(len(paths[i])):
                paths[i][j] = Flight(flight_network[flight_network["InventoryId"]==paths[i][j]][['FC_AvailableInventory', 'BC_AvailableInventory', 'PC_AvailableInventory', 'EC_AvailableInventory']], paths[i][j], flight_network[flight_network["InventoryId"]==paths[i][j]]['DepartureAirport'].iloc[0], flight_network[flight_network["InventoryId"]==paths[i][j]]['ArrivalAirport'].iloc[0])
        
        sampleset =  reaccomodation(PNR, paths, scores, alpha, sources[disrupt], destinations[disrupt], impacted_pax[disrupt], disrupt, TOKEN)
        
if __name__ == '__main__':
    main("INV-ZZ-3174758")