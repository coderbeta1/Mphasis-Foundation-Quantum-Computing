from numpy import random
import pandas as pd

def create_random_flights(max_flight, min_flight, num_airport, num_days):
    flight_id = 13412
    flights_IDs = []
    flights_source = []
    flights_destination = []
    flights_departure_time = []
    flights_departure_date = []
    flights_arrival_time = []
    flights_arrival_date = []
    
    date = 1
    for day in range(num_days):
        num_flights = random.randint(min_flight,max_flight)
        for flight in range(num_flights):
            current_flight_id = flight_id
            current_flights_source = random.randint(1, num_airport+1)
            current_flights_destination = random.randint(1, num_airport+1)
            if(current_flights_source == current_flights_destination): current_flights_destination-=1
            if(current_flights_destination == 0): current_flights_destination = 6
            current_flight_dep_time = random.randint(1, 25)
            current_flight_arr_time = random.randint(1, 25)
            if(current_flight_dep_time == current_flight_arr_time): current_flight_arr_time+=3
            if(current_flight_arr_time >24): current_flight_arr_time = 3
            if current_flight_dep_time>current_flight_arr_time:
                current_flight_arr_date = date + 1
            else: 
                current_flight_arr_date = date
            current_flight_dep_date = date

            flights_IDs.append(current_flight_id)
            flights_source.append(current_flights_source)
            flights_destination.append(current_flights_destination)
            flights_departure_time.append(current_flight_dep_time)
            flights_arrival_time.append(current_flight_arr_time)
            flights_arrival_date.append(current_flight_arr_date)
            flights_departure_date.append(current_flight_dep_date)

            flight_id += 1
        date += 1
    
    data = {
        "Flight_ID": flights_IDs,
        "Source_Airport": flights_source,
        "Dest_Airport": flights_destination, 
        "Dep_Time": flights_departure_time, 
        "Arr_Time": flights_arrival_time,
        "Dep_Date": flights_departure_date,
        "Arr_Date": flights_arrival_date
    }

    flight_network = pd.DataFrame(data=data)
    flight_network.sort_values(by=["Dep_Date", "Dep_Time", "Source_Airport"],inplace=True, kind="stable")

    return flight_network

max_number_of_flights_per_day = 1000
min_number_of_flights_per_day = 999
number_of_airports = 30
# departure_time = random
# arrival_time = random
number_of_days = 100

flight_network = create_random_flights(max_number_of_flights_per_day, min_number_of_flights_per_day, number_of_airports, number_of_days)

flight_network.to_csv("flight_network.csv")