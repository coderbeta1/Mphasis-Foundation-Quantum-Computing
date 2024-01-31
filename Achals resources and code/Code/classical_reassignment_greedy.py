import pandas as pd
from datetime import datetime, timedelta
import time

def find_impacted_passengers(disrupted_flight_data, PNR_list, passenger_details, scoring_criteria):
    PNR_list = PNR_list[PNR_list['FLT_NUM'] == disrupted_flight_data["FlightNumber"]]
    dep_date = datetime.strptime(disrupted_flight_data['DepartureDateTime'], "%Y-%m-%d %H:%M:%S")
    dep_date = dep_date.strftime("%#m/%#d/%#Y %#H:%M")
    PNR_list = PNR_list[PNR_list['DEP_DTML'] == dep_date]

    passenger_details = passenger_details.groupby('RECLOC')
    passenger_affected = passenger_details.get_group(PNR_list.iloc[0,:]['RECLOC'])

    for i in range(1,len(PNR_list)):
        passenger_affected = pd.concat([passenger_affected,passenger_details.get_group(PNR_list.iloc[i,:]['RECLOC'])])
    scores, class_passenger, pax_count = calc_score(passenger_affected, PNR_list, scoring_criteria)
    passenger_affected["Scores"] = scores
    passenger_affected["BookingClass"] = class_passenger
    passenger_affected["PAX_CNT"] = pax_count
    passenger_affected.sort_values(by= ["Scores"], inplace=True, ascending= False)
    passenger_affected = passenger_affected.reset_index().iloc[:,1:]

    return passenger_affected, PNR_list
def calc_score(passenger_details, PNR_list, scoring_criteria):
    scores = []
    class_passenger = []
    pax_count = []

    for i in range(len(passenger_details)):
        current_score = 0
        current_passenger = passenger_details.iloc[i,:]
        if type(current_passenger['SSR_CODE_CD1']) == str: current_score += scoring_criteria['SSR']
        current_score += (PNR_list.at[PNR_list['RECLOC'].eq(current_passenger['RECLOC']).idxmax(), "PAX_CNT"]) * scoring_criteria["Per_Pax"]
        if type(current_passenger['TierLevel'] == str):
            if current_passenger['TierLevel'] == "Gold": current_score += scoring_criteria["Loyalty_Gold"]
            if current_passenger['TierLevel'] == "Silver": current_score += scoring_criteria["Loyalty_Silver"]
            if current_passenger['TierLevel'] == "Platinum": current_score += scoring_criteria["Loyalty_Platinum"]
            if current_passenger['TierLevel'] == "Presidential Platinum": current_score += scoring_criteria["Loyalty_PPlatinum"]
        current_score += (scoring_criteria[PNR_list.at[PNR_list['RECLOC'].eq(current_passenger['RECLOC']).idxmax(), "COS_CD"]])
        scores.append(current_score)
        class_passenger.append(PNR_list.at[PNR_list['RECLOC'].eq(current_passenger['RECLOC']).idxmax(), "COS_CD"])
        pax_count.append(PNR_list.at[PNR_list['RECLOC'].eq(current_passenger['RECLOC']).idxmax(), "PAX_CNT"])
    return scores, class_passenger, pax_count
def find_possible_paths(flight_network, disrupted_flight_data):
    if type(disrupted_flight_data['PathFinding']) == str:
        paths = disrupted_flight_data['PathFinding'].split("//")[:-1]

        solutions = []
        for i in paths:
            current_path = i.split("/")
            solutions.append(flight_network[flight_network['InventoryId'].isin(current_path)])
        return solutions
    else: return []

def assign_class(counts, pax_count):
    assigned = -1
    for i in range(len(counts)):
        if counts[i] >= pax_count :
            assigned = i
            counts[i] -= pax_count            
            break
    return assigned, counts

def reallocate_passengers(path_details, passenger_affected):
    passenger_affected = passenger_affected.groupby('RECLOC').first()
    if (len(path_details) == 0):
        return [], list(passenger_affected.index), len(list(passenger_affected.index)) 

    first_seats = []
    eco_seats = []
    prem_seats = []
    buss_seats = []

    wrong_class_count = 0
    assignment = {}
    unassigned = []

    for sol in path_details:
        first_seats_curr = []
        eco_seats_curr = []
        prem_seats_curr = []
        buss_seats_curr = []
        for i in range(len(sol)):
            current_flight = sol.iloc[i,:]
            first_seats_curr.append(current_flight['FC_AvailableInventory'])
            eco_seats_curr.append(current_flight['EC_AvailableInventory'])
            prem_seats_curr.append(current_flight['PC_AvailableInventory'])
            buss_seats_curr.append(current_flight['BC_AvailableInventory'])
        first_seats.append(min(first_seats_curr))
        eco_seats.append(min(eco_seats_curr))
        prem_seats.append(min(prem_seats_curr))
        buss_seats.append(min(buss_seats_curr))

    for passenger in range(len(passenger_affected)):
        current_passenger = passenger_affected.iloc[passenger,:]
        pass_class = current_passenger['BookingClass']
        pax_count = current_passenger['PAX_CNT']

        if pass_class == 'FirstClass':
            assigned, first_seats = assign_class(first_seats, pax_count)
            if assigned != -1: 
                assignment[current_passenger.name] = ('FirstClass' ,path_details[assigned])
                continue
            else:
                assigned, buss_seats = assign_class(buss_seats, pax_count)
            
            if assigned != -1: 
                assignment[current_passenger.name] = ('BusinessClass', path_details[assigned])
                wrong_class_count += 1
                continue
            else:
                assigned, prem_seats = assign_class(prem_seats, pax_count)
            
            if assigned != -1: 
                assignment[current_passenger.name] = ('PremiumEconomyClass', path_details[assigned])
                wrong_class_count += 1
                continue
            else:
                assigned, eco_seats = assign_class(eco_seats, pax_count)
            
            if assigned != -1: 
                assignment[current_passenger.name] = ('EconomyClass', path_details[assigned])
                wrong_class_count += 1
                continue
            else:
                unassigned.append(current_passenger.name)
        elif pass_class == 'BusinessClass':
            assigned, buss_seats = assign_class(buss_seats, pax_count)
            if assigned != -1: 
                assignment[current_passenger.name] = ('BusinessClass' ,path_details[assigned])
                continue
            else:
                assigned, prem_seats = assign_class(prem_seats, pax_count)
            
            if assigned != -1: 
                assignment[current_passenger.name] = ('PremiumEconomyClass', path_details[assigned])
                wrong_class_count += 1
                continue
            else:
                assigned, eco_seats = assign_class(eco_seats, pax_count)
            
            if assigned != -1: 
                assignment[current_passenger.name] = ('EconomyClass', path_details[assigned])
                wrong_class_count += 1
                continue
            else:
                assigned, first_seats = assign_class(first_seats, pax_count)
            
            if assigned != -1: 
                assignment[current_passenger.name] = ('FirstClass', path_details[assigned])
                wrong_class_count += 1
                continue
            else:
                unassigned.append(current_passenger.name)
        elif pass_class == 'PremiumEconomyClass':
            assigned, prem_seats = assign_class(prem_seats, pax_count)
            if assigned != -1: 
                assignment[current_passenger.name] = ('PremiumEconomyClass' ,path_details[assigned])
                continue
            else:
                assigned, eco_seats = assign_class(eco_seats, pax_count)
            
            if assigned != -1: 
                assignment[current_passenger.name] = ('EconomyClass', path_details[assigned])
                wrong_class_count += 1
                continue
            else:
                assigned, buss_seats = assign_class(buss_seats, pax_count)
            
            if assigned != -1: 
                assignment[current_passenger.name] = ('BusinessClass', path_details[assigned])
                wrong_class_count += 1
                continue
            else:
                assigned, first_seats = assign_class(first_seats, pax_count)
            
            if assigned != -1: 
                assignment[current_passenger.name] = ('FirstClass', path_details[assigned])
                wrong_class_count += 1
                continue
            else:
                unassigned.append(current_passenger.name)
        elif pass_class == 'EconomyClass':
            assigned, eco_seats = assign_class(eco_seats, pax_count)
            if assigned != -1: 
                assignment[current_passenger.name] = ('EconomyClass' ,path_details[assigned])
                continue
            else:
                assigned, prem_seats = assign_class(prem_seats, pax_count)
            
            if assigned != -1: 
                assignment[current_passenger.name] = ('PremiumEconomyClass', path_details[assigned])
                wrong_class_count += 1
                continue
            else:
                assigned, buss_seats = assign_class(buss_seats, pax_count)
            
            if assigned != -1: 
                assignment[current_passenger.name] = ('BusinessClass', path_details[assigned])
                wrong_class_count += 1
                continue
            else:
                assigned, first_seats = assign_class(first_seats, pax_count)
            
            if assigned != -1: 
                assignment[current_passenger.name] = ('FirstClass', path_details[assigned])
                wrong_class_count += 1
                continue
            else:
                unassigned.append(current_passenger.name)

    return assignment, unassigned, wrong_class_count

def main(scoring_criteria, PNR_list, passenger_details, flight_network, disrupted_row = 123):

    disrupted_flight_data = flight_network[flight_network["InventoryId"] == "INV-ZZ-1206499"].iloc[0,:]
    passenger_affected, PNR_list = find_impacted_passengers(disrupted_flight_data, PNR_list, passenger_details, scoring_criteria)
    path_details = find_possible_paths(flight_network, disrupted_flight_data)

    print(passenger_affected)

    start_time = time.perf_counter()

    assignment, unassigned, wrong_class_count = reallocate_passengers(path_details, passenger_affected)
    

    end_time = time.perf_counter()

    print("DISRUPTION:", disrupted_flight_data['InventoryId'], "UNASSIGNED PNRs:", len(unassigned), "WRONG CLASS COUNT:", wrong_class_count, "RUNTIME:", end_time-start_time)

    return end_time - start_time

scoring_criteria = {"SSR": 200,
                    "Per_Pax": 50,
                    "Loyalty_Silver": 1500,
                    "Loyalty_Gold": 1600,
                    "Loyalty_Platinum": 1800,
                    "Loyalty_PPlatinum": 2000,
                    "Booking_Group": 500,
                    "Paid_Service": 200,
                    "Downline_Connection": 100,
                    "EconomyClass": 500,
                    "BusinessClass": 600,
                    "PremiumEconomyClass": 550,
                    "FirstClass": 650,
                    "Cabin_F": 1750,
                    "Cabin_J": 1750,
                    "Cabin_Y": 1750,
                    }

PNR_list = pd.read_csv("PNRB-ZZ-20231208_062017.csv")
passenger_details = pd.read_csv("PNRP-ZZ-20231208_111136.csv")
flight_network = pd.read_csv("PathsFound.csv")
flight_network = flight_network.iloc[:,1:]

main(scoring_criteria, PNR_list, passenger_details, flight_network, 32)

# times = []

# start_time = time.perf_counter()

# for i in range(5):
#     times.append(main(scoring_criteria, PNR_list, passenger_details, flight_network, i))
# end_time = time.perf_counter()

# print("\n\nTotal Runtime:", end_time-start_time)
# print("Average Runtime:", sum(times)/len(times))