import pandas as pd
import time

def statistics_printer(flight_network, disrupted_flight_data):
    print("\nSIMULATION STATISTICS: \n")
    print("Airports Count: ", max(flight_network.iloc[:,1]))
    print("Time Span: ", flight_network.iloc[-1,-1])
    print("\nFlight ID of Disruption: ", disrupted_flight_data["Flight_ID"])
    print("Source Node of Disruption: ", disrupted_flight_data["Source_Airport"])
    print("Destination Node of Disruption: ", disrupted_flight_data["Dest_Airport"])
    print("\nDeparture Time of Disruption: ", str(disrupted_flight_data["Dep_Time"]) + " // " + str(disrupted_flight_data["Dep_Date"]))
    print("Arrival Time of Disruption: ", str(disrupted_flight_data["Arr_Time"]) + " // " + str(disrupted_flight_data["Arr_Date"]))
    print()
    return 
def reduce_network(flight_network: pd.DataFrame, disrupted_row, disrupted_flight_data):
    flight_network = flight_network.iloc[disrupted_row:,:]
    flight_network = flight_network[flight_network['Dep_Date']<=disrupted_flight_data['Dep_Date']+3]
    flight_network = flight_network.reset_index()
    flight_network = flight_network.iloc[:,1:]

    # Remove Disrupted Row
    flight_network = flight_network.iloc[1:,:]
    
    return flight_network
def calc_score(path, disrupted_flight_data):
    delay_amount = 0
    current_time = disrupted_flight_data['Dep_Time']
    current_date = disrupted_flight_data['Dep_Date']

    for flight in path:
        delay_amount += (flight['Dep_Time'] + 24*(flight['Dep_Date'] - current_date)) - current_time
        current_time = flight['Arr_Time']
        current_date = flight['Arr_Date']
    return delay_amount
def recurse(groups, current_node, destination, path_detailed, path, solution, solution_detailed, stopovers, current_time, current_date, disrupted_flight_data):
    if(current_node == destination):
        score = calc_score(path_detailed, disrupted_flight_data)
        solution.append(path.copy())
        solution[-1].append("Score:      "+str(score))
        solution_detailed.append(path_detailed.copy())
        solution_detailed[-1].append("Score:      "+str(score))
        return
    if(stopovers <= -1):
        return

    possibilities = groups.get_group(current_node)
    possibilities = possibilities[possibilities['Dep_Date']>=current_date]

    for i in range(len(possibilities)):
        current_flight = possibilities.iloc[i,:]
        if(current_flight['Dest_Airport'] in path):
            continue
        if ((current_flight['Dep_Time']>current_time and current_flight['Dep_Date']==current_date) or \
            (current_flight['Dep_Date']>current_date)):
            path_detailed.append(current_flight)
            path.append(current_flight['Flight_ID'])
            recurse(groups, current_flight['Dest_Airport'], destination, path_detailed, path, solution, solution_detailed, stopovers-1, current_flight['Arr_Time'], current_flight['Arr_Date'], disrupted_flight_data)
            path.pop()        
            path_detailed.pop()
        else:
            continue
         
    return
def solve(flight_network: pd.DataFrame, disrupted_flight_data, stopovers = 1):      
    groups = flight_network.groupby("Source_Airport")
    
    source = disrupted_flight_data['Source_Airport']
    destination = disrupted_flight_data['Dest_Airport']
    current_time = disrupted_flight_data['Dep_Time']
    current_date = disrupted_flight_data['Dep_Date']

    solutions = []
    solutions_detailed = []
    path_detailed = []
    path = []
    recurse(groups, source, destination, path_detailed, path, solutions, solutions_detailed, stopovers, current_time, current_date, disrupted_flight_data)
    solutions.sort(key=lambda x:int(x[-1][-5:]))

    solution_string = ""
    for i in solutions:
        for j in i[:-1]:
            solution_string += str(j)
            solution_string += "/"
        solution_string += "/"
    
    return solution_string, solutions_detailed

def main(flight_network, disrupted_flight_row):
    # Create Flight Disruption
    row_count = len(flight_network)
    disrupted_flight_data = dict(flight_network.iloc[disrupted_flight_row, :])

    start_time = time.perf_counter()

    # statistics_printer(flight_network= flight_network, disrupted_flight_data= disrupted_flight_data)
    flight_network = reduce_network(flight_network, disrupted_flight_row, disrupted_flight_data)
    solutions, solutions_detailed = solve(flight_network, disrupted_flight_data, stopovers= 1)
    
    stop_time  = time.perf_counter()

    print("Row Number: ", disrupted_flight_row,"Solutions: ", len(solutions_detailed), "RUNTIME: ", stop_time-start_time)
    return

# Read Input Flight Data
flight_network = pd.read_csv("flight_network.csv")
flight_network = flight_network.iloc[:, 1:]

for i in range(10):
    main(flight_network, i)