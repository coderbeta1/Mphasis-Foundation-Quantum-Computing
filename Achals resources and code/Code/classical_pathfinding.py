import pandas as pd
from datetime import datetime, timedelta
import time
import Nearby_Airports

class pathfind_recursion:
    def __init__(self, flight_network, disrupted_flights, scoring_criteria, toggle, search_radius = 150, max_delay = 72, stopovers = 2, verbose = False):
        self.flight_network = flight_network
        self.disrupted_flights = disrupted_flights
        self.scoring_criteria = scoring_criteria
        self.toggle = toggle
        self.max_delay = max_delay
        self.stopovers = stopovers
        self.possible_destinations = list(self.flight_network['ArrivalAirport'].unique())
        self.all_destinations = None
        self.search_radius = search_radius
        self.verbose = verbose

        self.flight_network.sort_values(by=["DepartureDateTime"], inplace = True, kind= "stable")
        self.flight_network = self.flight_network.reset_index().iloc[:,1:]
        return
    
    def statistics_printer(self, disrupted_flight_data = {}, flights = 0):
        print("\nSIMULATION STATISTICS: \n")
        print("Number of Unique Airports Count: ", len(self.flight_network.groupby("DepartureAirport")))
        print("Total Number of Flights: ", len(self.flight_network))
        print("Time Span of Data Set: ", self.flight_network["DepartureDate"][0], "-", self.flight_network["DepartureDate"][len(self.flight_network)-1])
        if flights:
            print("\nFlight INV ID of Disruption: ", disrupted_flight_data["InventoryId"])
            print("Source Node of Disruption: ", disrupted_flight_data["DepartureAirport"])
            print("Destination Node of Disruption: ", disrupted_flight_data["ArrivalAirport"])
            print("Departure Time of Disruption: ", str(disrupted_flight_data["DepartureDateTime"]))
            print("Arrival Time of Disruption: ", str(disrupted_flight_data["ArrivalDateTime"]))
        print()
        return 
    
    def reduce_network(self, flight_network, disrupted_flight_data):
        
        flight_network = flight_network.iloc[disrupted_flight_data.name:,:]
        date = datetime.strptime((disrupted_flight_data["DepartureDateTime"]), "%Y-%m-%d %H:%M:%S")
        dateLimit = (date + timedelta(hours=self.max_delay)).strftime("%Y-%m-%d %H:%M:%S")
        flight_network = flight_network[flight_network['DepartureDateTime']<=dateLimit]
        flight_network = flight_network.reset_index()
        flight_network = flight_network.iloc[:,1:]

        # Remove Disrupted Row
        flight_network = flight_network.iloc[1:,:]
        
        return flight_network

    def calc_score(self, path_detailed, disrupted_flight_data):
        score = 0
        arrival_delay = (datetime.strptime(path_detailed[-1]['ArrivalDateTime'], "%Y-%m-%d %H:%M:%S") - datetime.strptime(disrupted_flight_data['ArrivalDateTime'], "%Y-%m-%d %H:%M:%S")).total_seconds()/3600
        if arrival_delay < 6 and self.toggle['Arr_LT_6hrs']:
            score += self.scoring_criteria['Arr_LT_6hrs']
        elif arrival_delay < 12 and self.toggle['Arr_LT_12hrs']:
            score += self.scoring_criteria['Arr_LT_12hrs']
        elif arrival_delay < 24 and self.toggle['Arr_LT_24hrs']:
            score += self.scoring_criteria['Arr_LT_24hrs']
        elif arrival_delay < 48 and self.toggle['Arr_LT_48hrs']:
            score += self.scoring_criteria['Arr_LT_48hrs']
        
        departure_delay = (datetime.strptime(path_detailed[0]['DepartureDateTime'], "%Y-%m-%d %H:%M:%S") - datetime.strptime(disrupted_flight_data['DepartureDateTime'], "%Y-%m-%d %H:%M:%S")).total_seconds()/3600
        if departure_delay < 6 and self.toggle['SPF_LT_6hrs']:
            score += self.scoring_criteria['SPF_LT_6hrs']
        elif departure_delay < 12 and self.toggle['SPF_LT_12hrs']:
            score += self.scoring_criteria['SPF_LT_12hrs']
        elif departure_delay < 24 and self.toggle['SPF_LT_24hrs']:
            score += self.scoring_criteria['SPF_LT_24hrs']
        elif departure_delay < 48 and self.toggle['SPF_LT_48hrs']:
            score += self.scoring_criteria['SPF_LT_48hrs']

        equipment_check = 1
        for flight in path_detailed:
            if flight['AircraftType'] != disrupted_flight_data['AircraftType']:
                equipment_check = 0
                break
        if equipment_check and self.toggle['Equipment']: score += self.scoring_criteria['Equipment']
        if len(path_detailed) != 1 and self.toggle['IsStopover']: score += self.scoring_criteria['IsStopover']

        if path_detailed[-1]['ArrivalAirport'] == disrupted_flight_data['ArrivalAirport'] and self.toggle['Same_Citipairs']: score += self.scoring_criteria['Same_Citipairs']
        if path_detailed[-1]['ArrivalAirport'] != disrupted_flight_data['ArrivalAirport'] and self.toggle['Different_Citipairs']: score += self.scoring_criteria['Different_Citipairs']

        return score

    def recurse(self, groups, current_node, destination, path, path_detailed, solution, stopovers, current_datetime, disrupted_flight_data):
        if(current_node == destination):
            if len(path_detailed) == 0: return
            score = self.calc_score(path_detailed, disrupted_flight_data)
            solution.append(path.copy())
            solution[-1].append(score)
            return
        if(stopovers <= -1):
            return

        try:
            possibilities = groups.get_group(current_node)
            possibilities = possibilities[possibilities['DepartureDateTime'] > current_datetime.strftime("%Y-%m-%d %H:%M:%S")]
        except:
            possibilities = pd.DataFrame()

        for i in range(len(possibilities)):
            current_flight = possibilities.iloc[i,:]
            if(current_flight['InventoryId'] in path):
                continue
            if current_flight['InventoryId'] in self.disrupted_flights:
                continue
            if ((datetime.strptime(current_flight['DepartureDateTime'], "%Y-%m-%d %H:%M:%S") > current_datetime + timedelta(minutes=60))):
                if (len(path) == 0) or ((len(path) >= 1) and ((datetime.strptime(current_flight['DepartureDateTime'], "%Y-%m-%d %H:%M:%S") < current_datetime + timedelta(hours=12)))):
                    path.append(current_flight['InventoryId'])
                    path_detailed.append(current_flight)
                    self.recurse(groups, current_flight['ArrivalAirport'], destination, path, path_detailed, solution, stopovers-1, datetime.strptime(current_flight['ArrivalDateTime'], "%Y-%m-%d %H:%M:%S"), disrupted_flight_data)
                    path.pop()        
                    path_detailed.pop()               
            else:
                continue
            
        return
    
    def solver(self, disruption):    
        start_time = time.perf_counter()

        disrupted_flight_data = self.flight_network[self.flight_network["InventoryId"] == disruption].iloc[0, :]
        reduced_network = self.reduce_network(self.flight_network.copy(), disrupted_flight_data)
        
        groups = reduced_network.groupby("DepartureAirport")

        source = disrupted_flight_data['DepartureAirport']
        destination = Nearby_Airports.nearby_airport(disrupted_flight_data['ArrivalAirport'], self.search_radius)
        destination.append(disrupted_flight_data['ArrivalAirport'])
        current_datetime = datetime.strptime(disrupted_flight_data['DepartureDateTime'], "%Y-%m-%d %H:%M:%S") + timedelta(hours=-1)

        true_destinations = []

        solutions = []
        for dest in destination:    
            if dest not in self.possible_destinations: continue    
            true_destinations.append(dest)
            path = []
            path_detailed = []
            self.recurse(groups, source, dest, path, path_detailed, solutions, self.stopovers, current_datetime, disrupted_flight_data)
        solutions.sort(key=lambda x:int(x[-1]), reverse= True)

        final = []
        alphas = []

        for i in solutions:
            if i[-1] >= self.scoring_criteria['A_Grade'] and self.toggle['A_Grade']:
                alphas.append(i[-1])
                final.append(i)
            elif i[-1] >= self.scoring_criteria['B_Grade'] and self.toggle['B_Grade']:
                alphas.append(i[-1])
                final.append(i)
            elif i[-1] >= self.scoring_criteria['C_Grade'] and self.toggle['C_Grade']:
                alphas.append(i[-1])
                final.append(i)
            elif self.toggle['D_Grade']:
                alphas.append(i[-1])
                final.append(i)

        div_value = max(alphas)
        alphas = [alphas[x]/(div_value * ((len(solutions[x])-1) ** 2)) for x in range(len(alphas))]

        self.all_destinations = true_destinations

        end_time = time.perf_counter()
        if self.verbose: print("DISRUPTION:", disruption, "RUNTIME: ", end_time - start_time, "SOLUTIONs: ", len(final))

        return final, alphas, [source], self.all_destinations

    def multisolve(self):
        solution = {}
        alphas = {}
        sources = {}
        destinations = {}
        for disruption in self.disrupted_flights:
            solution[disruption], alphas[disruption], sources[disruption], destinations[disruption] = self.solver(disruption)
        return solution, alphas, sources, destinations
    
    def solve(self):
        if self.verbose: self.statistics_printer()
        if(len(self.disrupted_flights) == 1):
            a, b, c, d = self.solver(self.disrupted_flights[0])
            return {str(self.disrupted_flights[0]): a}, {str(self.disrupted_flights[0]): b}, {str(self.disrupted_flights[0]): c}, {str(self.disrupted_flights[0]): d}
        else:
            return self.multisolve()

class impacted_PNR:
    def __init__(self, scoring_criteria, PNR_list, passenger_details, toggle, flight_network, disruption, verbose = 0):
        self.scoring_criteria = scoring_criteria
        self.PNR_list = PNR_list
        self.passenger_details = passenger_details
        self.flight_network = flight_network
        self.toggle = toggle
        self.disruption = disruption
        self.verbose = verbose

        self.flight_network.sort_values(by=["DepartureDateTime"], inplace = True, kind= "stable")
        self.flight_network = self.flight_network.reset_index().iloc[:,1:]
        pass

    def find_impacted_passengers(self, PNR_list, passenger_details, disrupted_flight_data):
        PNR_list = PNR_list[PNR_list['FLT_NUM'] == disrupted_flight_data["FlightNumber"]]
        dep_date = datetime.strptime(disrupted_flight_data['DepartureDateTime'], "%Y-%m-%d %H:%M:%S")
        dep_date = dep_date.strftime("%#m/%#d/%#Y %#H:%M")
        PNR_list = PNR_list[PNR_list['DEP_DTML'] == dep_date]
        detailed_PNR = self.PNR_list[self.PNR_list['RECLOC'].isin(list(PNR_list['RECLOC']))]

        passenger_details = passenger_details.groupby('RECLOC')
        passenger_affected = passenger_details.get_group(PNR_list.iloc[0,:]['RECLOC']).copy()
        current_PNR = PNR_list.iloc[0,:]
        pax = current_PNR['PAX_CNT']
        booking_class = current_PNR['COS_CD']
        passenger_affected['COS_CD'] = [booking_class] * len(passenger_affected)
        passenger_affected['PAX_CNT'] = [pax] * len(passenger_affected)
        score = self.calc_score(passenger_affected, current_PNR)
        passenger_affected["Scores"] = [max(score)] * len(score)    

        for i in range(1,len(PNR_list)):
            passengers = passenger_details.get_group(PNR_list.iloc[i,:]['RECLOC']).copy()
            current_PNR = PNR_list.iloc[i,:]
            pax = current_PNR['PAX_CNT']
            booking_class = current_PNR['COS_CD']
            passengers['COS_CD'] = [booking_class] * len(passengers)
            passengers['PAX_CNT'] = [pax] * len(passengers)
            score = self.calc_score(passengers, current_PNR)
            passengers["Scores"] = [max(score)] * len(score)

            passenger_affected = pd.concat([passenger_affected, passengers])
            
        passenger_affected.sort_values(by= ["Scores"], inplace=True, ascending= False)
        passenger_affected = passenger_affected.reset_index().iloc[:,1:]

        return passenger_affected
    
    def calc_score(self, passengers, PNR):
        scores = []
        for i in range(len(passengers)):
            current_score = 0
            current_passenger = passengers.iloc[i,:]
            if type(current_passenger['SSR_CODE_CD1']) == str and self.toggle['SSR']: current_score += self.scoring_criteria['SSR']
            if self.toggle['Per_Pax']: current_score += PNR['PAX_CNT'] * self.scoring_criteria["Per_Pax"]
            if type(current_passenger['TierLevel'] == str):
                if current_passenger['TierLevel'] == "Gold" and self.toggle['Loyalty_Gold']: current_score += self.scoring_criteria["Loyalty_Gold"]
                if current_passenger['TierLevel'] == "Silver" and self.toggle['Loyalty_Silver']: current_score += self.scoring_criteria["Loyalty_Silver"]
                if current_passenger['TierLevel'] == "Platinum" and self.toggle['Loyalty_Platinum']: current_score += self.scoring_criteria["Loyalty_Platinum"]
                if current_passenger['TierLevel'] == "Presidential Platinum" and self.toggle['Loyalty_PPlatinum']: current_score += self.scoring_criteria["Loyalty_PPlatinum"]
            if self.toggle['EconomyClass'] and PNR['COS_CD'] == 'EconomyClass': current_score += (self.scoring_criteria[PNR['COS_CD']])
            if self.toggle['BusinessClass'] and PNR['COS_CD'] == 'BusinessClass': current_score += (self.scoring_criteria[PNR['COS_CD']])
            if self.toggle['PremiumEconomyClass'] and PNR['COS_CD'] == 'PremiumEconomyClass': current_score += (self.scoring_criteria[PNR['COS_CD']])
            if self.toggle['FirstClass'] and PNR['COS_CD'] == 'FirstClass': current_score += (self.scoring_criteria[PNR['COS_CD']])
            
            scores.append(current_score)
        return scores
    
    def solver(self, disruption):
        flight_disruption = self.flight_network[self.flight_network["InventoryId"] == disruption].iloc[0,:]
        passengers = self.find_impacted_passengers(self.PNR_list.copy(), self.passenger_details.copy(), flight_disruption)
        PNR_list = passengers.groupby('RECLOC').first()
        PNR_list.sort_values(by= ["Scores"], inplace=True, ascending= False)
        return passengers, PNR_list
    
    def solve(self):
        if len(self.disruption) == 1:
            solution_pax, solution_pnr = self.solver(self.disruption[0])
            mater = self.matrix_finder(solution_pnr)
            return {self.disruption[0]:solution_pax}, {self.disruption[0]:solution_pnr}, {self.disruption[0]:mater}
        else:
            solution_pax = {}
            solution_pnr = {}
            matrix_find = {}
            for disrupt in self.disruption:
                solution_pax[disrupt], solution_pnr[disrupt] = self.solver(disrupt)
                matrix_find[disrupt] = self.matrix_finder(solution_pnr[disrupt])
            return solution_pax, solution_pnr, matrix_find

    def matrix_finder(self, solution_pnr):
        bossman = {}

        for pnr in range(len(solution_pnr)):
            current_pnr = solution_pnr.iloc[pnr,:].copy()
            answer = {"FirstClass":0,
                      "BusinessClass":0,
                      "PremiumEconomyClass":0,
                      "EconomyClass":0,
                      }
            all_class = list(answer.keys())
            class_curr = current_pnr['COS_CD']
            curr_score = current_pnr['Scores']
            answer[current_pnr['COS_CD']] = curr_score
            
            if class_curr == 'FirstClass' and self.toggle['Downgrade_allow']:
                answer[all_class[1]] = curr_score - 100
            if class_curr == 'EconomyClass' and self.toggle['Upgrade_allow']:
                answer[all_class[2]] = curr_score - 100
            if class_curr == 'PremiumEconomyClass' and self.toggle['Upgrade_allow']:
                answer[all_class[1]] = curr_score - 100
            if class_curr == 'PremiumEconomyClass' and self.toggle['Downgrade_allow']:
                answer[all_class[3]] = curr_score - 100
            if class_curr == 'BusinessClass' and self.toggle['Upgrade_allow']:
                answer[all_class[0]] = curr_score - 100
            if class_curr == 'BusinessClass' and self.toggle['Downgrade_allow']:
                answer[all_class[2]] = curr_score - 100

            final = {}
            final["FC"] = answer["FirstClass"]
            final["BC"] = answer["BusinessClass"]
            final["PC"] = answer["PremiumEconomyClass"]
            final["EC"] = answer["EconomyClass"]
            bossman[current_pnr.name] = final
        
        return bossman

# Read Input Flight Data
flight_network = pd.read_csv("INV-ZZ-20231208_041852.csv")
PNR_list = pd.read_csv("PNRB-ZZ-20231208_062017.csv")
passenger_details = pd.read_csv("PNRP-ZZ-20231208_111136.csv")
disruption = ["INV-ZZ-1206499"]

scoring_criteria_PNRs = {"SSR": 200,
                        "Per_Pax": 50,
                        "Loyalty_Silver": 1500,
                        "Loyalty_Gold": 1600,
                        "Loyalty_Platinum": 1800,
                        "Loyalty_PPlatinum": 2000,
                        "Booking_Group": 500,
                        "Paid_Service": 200,
                        "Downline_Connection": 100,
                        "EconomyClass": 1500,
                        "BusinessClass": 1850,
                        "PremiumEconomyClass": 1650,
                        "FirstClass": 2000,
                        }
scoring_criteria_PNRs_toggle = { "SSR": 1,
                                "Per_Pax": 1,
                                "Loyalty_Silver": 1,
                                "Loyalty_Gold": 1,
                                "Loyalty_Platinum": 1,
                                "Loyalty_PPlatinum": 1,
                                "Booking_Group": 1,
                                "Paid_Service": 1,
                                "Downline_Connection": 1,
                                "EconomyClass": 1,
                                "BusinessClass": 1,
                                "PremiumEconomyClass": 1,
                                "FirstClass": 1,
                                "Downgrade_allow": 1,
                                "Upgrade_allow": 1,
                                }

scoring_criteria_Flights = {"Arr_LT_6hrs": 70,
                            "Arr_LT_12hrs": 50,
                            "Arr_LT_24hrs": 40,
                            "Arr_LT_48hrs": 30,
                            "Equipment": 50,
                            "Same_Citipairs": 40,
                            "DiffSame_Citipairs": 30,
                            "Different_Citipairs": 20,
                            "SPF_LT_6hrs": 70,
                            "SPF_LT_12hrs": 50,
                            "SPF_LT_24hrs": 40,
                            "SPF_LT_48hrs": 30,
                            "IsStopover": -20,
                            "A_Grade": 200,
                            "B_Grade": 180,
                            "C_Grade": 150,
                            "D_Grade": 150,
                            }
scoring_criteria_Flights_toggle = {"Arr_LT_6hrs": 1,
                                "Arr_LT_12hrs": 1,
                                "Arr_LT_24hrs": 1,
                                "Arr_LT_48hrs": 1,
                                "Equipment": 1,
                                "Same_Citipairs": 1,
                                "DiffSame_Citipairs": 1,
                                "Different_Citipairs": 1,
                                "SPF_LT_6hrs": 1,
                                "SPF_LT_12hrs": 1,
                                "SPF_LT_24hrs": 1,
                                "SPF_LT_48hrs": 1,
                                "IsStopover": 1,
                                "A_Grade": 1,
                                "B_Grade": 1,
                                "C_Grade": 1,
                                "D_Grade": 1,
                                }

# Create Object and run solve function
solver = pathfind_recursion(flight_network, disruption, scoring_criteria= scoring_criteria_Flights, toggle = scoring_criteria_Flights_toggle, verbose= 1, stopovers= 2)
solutions, alphas, sources, destinations = solver.solve()

# Create Object and run solve function for PNRs
PNR_list = impacted_PNR(scoring_criteria_PNRs, PNR_list, passenger_details, scoring_criteria_PNRs_toggle, flight_network, disruption)
impacted_pax, PNRs, matrix_solved = PNR_list.solve()

def calculate_score(data, index = 0 ):
    score_value = 0
    for i in range(len(data)):
        current_row = data.iloc[i,:]
        score_value += int(matrix_solved[disruption[index]][current_row['PNR ID']][current_row['Class']])

    return score_value

