import pandas as pd
from numpy import random
import numpy as np
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

# Read Input Flight Data
flight_network = pd.read_csv("flight_network.csv")
flight_network = flight_network.iloc[:, 1:]

# Create Flight Disruption
row_count = len(flight_network)
disrupted_flight_row = 153
disrupted_flight_data = dict(flight_network.iloc[disrupted_flight_row, :])

'''
Reduce Current Network
'''
def reduce_network(flight_network: pd.DataFrame, disrupted_row):
    flight_network = flight_network.iloc[disrupted_row:,:]
    flight_network = flight_network[flight_network['Dep_Date']<=disrupted_flight_data['Dep_Date']+3]
    flight_network = flight_network.reset_index()
    flight_network = flight_network.iloc[:,1:]

    # Remove Disrupted Row
    flight_network = flight_network.iloc[1:,:]
    
    return flight_network

'''
Create Ant Colony Algorithm
'''
class AntColonySolver:
    def __init__(self,
                #  cost_fn=0,                         
                 sol_count = 1,

                 time=0,                  # run for a fixed amount of time
                 min_time=0,              # minimum runtime
                 timeout=0,               # maximum time in seconds to run for
                 stop_factor=2,           # how many times to redouble effort after new new best path
                #  min_round_trips=10,      # minimum number of round trips before stopping
                #  max_round_trips=0,       # maximum number of round trips before stopping                 
                 min_ants=0,              # Total number of ants to use
                 max_ants=0,              # Total number of ants to use
                 
                 ant_count=16,            # this is the bottom of the near-optimal range for numpy performance
                 ant_speed=1,             # how many steps do ants travel per epoch

                 distance_power=1,        # power to which distance affects pheromones                 
                 pheromone_power=1.25,    # power to which differences in pheromones are noticed
                 decay_rate=0,            # how fast do pheromones decay
                 reward_power=0,          # relative pheromone reward based on best_path_length/path_length 
                 best_path_smell=2,       # queen multiplier for pheromones upon finding a new best path                  
                 start_smell=0,           # amount of starting pheromones [0 defaults to `10**self.distance_power`]

                 verbose=False,

    ) -> None:

        # self.cost_fn         = cost_fn
        self.time            = int(time)
        self.min_time        = int(min_time)
        self.timeout         = int(timeout)
        self.stop_factor     = float(stop_factor)
        # self.min_round_trips = int(min_round_trips)
        # self.max_round_trips = int(max_round_trips)
        self.min_ants        = int(min_ants)
        self.max_ants        = int(max_ants)
    
        self.ant_count       = int(ant_count)
        self.ant_speed       = int(ant_speed)
        
        self.distance_power  = float(distance_power)     
        self.pheromone_power = float(pheromone_power)
        self.decay_rate      = float(decay_rate)
        self.reward_power    = float(reward_power)
        self.best_path_smell = float(best_path_smell)
        self.start_smell     = float(start_smell or 10**self.distance_power)
        
        self.sol_count       = sol_count

        self.verbose         = int(verbose)
        self._initalized     = False

        self.rejection_list  = []

        # if self.min_round_trips and self.max_round_trips: self.min_round_trips = min(self.min_round_trips, self.max_round_trips)
        if self.min_ants and self.max_ants:               self.min_ants        = min(self.min_ants, self.max_ants)
       
    def solve_initialize(
            self,
            flight_network,
    ) -> None:
        
        pheromone_network= flight_network.copy()
        pheromone_network['Pheromone'] = 1
        self.pheromone_network = pheromone_network

        self._initalized = True   
        return     

    def solve(self,
              flight_network,
              disrupted_flight_data,
              restart=False,
    ) -> list[tuple[int,int]]:
        
        # If it is (first run) or (want to reset) Initialize 
        if restart or not self._initalized:
            self.solve_initialize(flight_network)
        
        solution_list = []

        while self.sol_count > 0:
            '''
            Define the Ant(Agent) Population and Initialize
            '''
            ants = {
                "distance":     np.zeros((self.ant_count,)).astype('int32'),
                "path":         [ [ disrupted_flight_data['Source_Airport'] ]   for n in range(self.ant_count) ],
                "path_cost":    np.zeros((self.ant_count,)).astype('int32'),
                "time":         [ disrupted_flight_data['Dep_Time']   for n in range(self.ant_count) ],
                "date":         [ disrupted_flight_data['Dep_Date']   for n in range(self.ant_count) ],
                "path_details": [ []   for n in range(self.ant_count) ],
                "check_list": [ ""  for n in range(self.ant_count) ],
            }

            best_path       = None
            best_path_details = None
            best_check_list = None
            best_path_cost  = np.inf
            best_epochs     = []
            epoch           = 0
            time_start      = time.perf_counter()

            while True:
                epoch += 1
                '''
                Simulated Walk of the Ants
                '''
                for ant in range(self.ant_count):
                    while True:
                        '''
                        Solving for Next Best Node
                        '''
                        next_flight = self.next_node(ants, ant)
        
                        if type(next_flight) == int:
                            ants['path'][ant] = [disrupted_flight_data['Source_Airport']]
                            ants['path_cost'][ant] = 1000000
                            ants['check_list'][ant] = ""
                            break
                        else:
                            ants['distance'][ant]  = (next_flight['Dep_Time'] + 24*(next_flight['Dep_Date'] - ants['date'][ant])) - ants['time'][ant]
                            ants['path_cost'][ant] = ants['path_cost'][ant] + ants['distance'][ant]    
                            ants['path'][ant].append(next_flight['Dest_Airport'])
                            ants['path_details'][ant].append(next_flight)
                            ants['check_list'][ant] += str(next_flight['Flight_ID'])
                            ants['time'][ant] = next_flight['Arr_Time']
                            ants['date'][ant] = next_flight['Arr_Date']

                        if next_flight['Dest_Airport'] == disrupted_flight_data['Dest_Airport']:
                            break
                
                    if (ants['path_cost'][ant] < best_path_cost) and (ants['path'][ant][-1] == disrupted_flight_data['Dest_Airport']) and (ants['check_list'][ant] not in self.rejection_list):
                            best_path_cost = ants['path_cost'][ant]
                            best_path      = ants['path'][ant]
                            best_check_list = ants['check_list'][ant]
                            best_path_details = ants['path_details'][ant]
                            best_epochs   += [ epoch ]
                
                if self.verbose:                # To Print Statistics if needed 
                    print({
                        "Path":        best_path,
                        "Path_cost":   best_path_cost,
                        "Epoch":       epoch,
                        "Clock":       time.perf_counter() - time_start,
                    })
                
                '''
                Leave Pheromone Trail
                
                1. Doing this only after ants reach destination improves initial exploration
                '''

                for ant in range(self.ant_count):
                    if(ants['check_list'][ant] in self.rejection_list):
                        for node in range(len(ants['path_details'][ant])):
                            self.pheromone_network.loc[ants['path_details'][ant][node].name, "Pheromone"] = (1-self.decay_rate)*self.pheromone_network.loc[ants['path_details'][ant][node].name,"Pheromone"] + (1/1000000)
                    for node in range(len(ants['path_details'][ant])):
                        self.pheromone_network.loc[ants['path_details'][ant][node].name, "Pheromone"] = ((1-self.decay_rate)*self.pheromone_network.loc[ants['path_details'][ant][node].name,"Pheromone"] + (1/ants['path_cost'][ant]))
                    
                    # Reset the Ant Values
                    ants["distance"][ant]     = 0
                    ants["path"][ant]         = [disrupted_flight_data['Source_Airport']]
                    ants["path_cost"][ant]    = 0
                    ants["time"][ant]         = disrupted_flight_data['Dep_Time']
                    ants["date"][ant]         = disrupted_flight_data['Dep_Date']
                    ants['path_details'][ant] = []
                    ants['check_list'][ant]   = ""

                '''
                Check Conditions of Termination

                1. Atleast One Solution
                2. Runtime Finished
                3. Max Trips Reached
                4. Ants Used more than Max
                5. Good enough solution for current Epoch
                '''
                
                # Always wait for at least 1 solutions (note: 2+ solutions are not guaranteed)
                # if not len(best_epochs): continue 
                
                # Running for too long (Time Based Constraint)
                if self.time or self.min_time or self.timeout:
                    clock = time.perf_counter() - time_start
                    if self.time:
                        if clock > self.time: break
                        else:                 continue
                    if self.min_time and clock < self.min_time: continue
                    if self.timeout  and clock > self.timeout:  break
                
                # # First epoch only has start smell - question: how many epochs are required for a reasonable result?
                # if self.min_round_trips and self.round_trips <  self.min_round_trips: continue        
                # if self.max_round_trips and self.round_trips >= self.max_round_trips: break

                # This factor is most closely tied to computational power                
                # if self.min_ants and self.ants_used <  self.min_ants: continue        
                # if self.max_ants and self.ants_used >= self.max_ants: break            
                
                # Lets keep redoubling our efforts until we can't find anything more
                # if self.stop_factor and epoch > (best_epochs[-1] * self.stop_factor): break
                                    
                # Nothing else is stopping us: Queen orders the ants to continue!      
                continue
            
            self.sol_count -= 1
            if best_check_list != "": self.rejection_list.append(best_check_list)
            if best_path != None: solution_list.append(best_path_details)
            else:
                for i in range(self.sol_count):
                    solution_list.append("No Solution Possible")
                break

        '''
        Run Statistics and Analysis
        '''
        self.epochs_used = epoch
        if(len(solution_list) == 1): return best_path_details
        else: return solution_list

    def next_node_choices(self, ants, ant, current_node, flight_network):
        possible_flights = []

        for i in range(len(flight_network)):
            current_flight = flight_network.iloc[i,:]
            if (current_flight['Source_Airport'] == current_node) and (current_flight['Dest_Airport'] not in ants['path'][ant] and ((current_flight['Dep_Time'] > ants['time'][ant] and current_flight['Dep_Date'] == ants['date'][ant]) or (current_flight['Dep_Date'] > ants['date'][ant]))):
                possible_flights.append(current_flight)
        
        return possible_flights

    def next_node(self, ants, ant):
        this_node = ants['path'][ant][-1]
        current_choices = self.next_node_choices(ants, ant, this_node, self.pheromone_network)
        
        weights = []
        weights_sum = 0

        for flight in current_choices:
            '''
            OPTIMIZATION FUNCTION (to maximize)
            '''
            reward = (flight['Pheromone'] ** self.pheromone_power) * ((flight['Dep_Time'] + 24*(flight['Dep_Date'] - ants['date'][ant])) - ants['time'][ant]) 

            '''
            OUTLIER CHECKING
            '''
            # if(time_stamp>max_flight_delay or 0):
            #     reward = 0
            
            weights.append(reward)
            weights_sum += reward

        if len(weights)>0: weights=[a/weights_sum for a in weights]
        
        # Pick a random path in proportion to the weight of the pheromone
        if len(current_choices)!=0: next_flight= np.random.choice(a=range(len(current_choices)), size=1, p=weights)
        else: return -1

        return current_choices[next_flight[0]]

'''
Create a Simulation Runner
'''
def AntColonyRunner(flight_network, disrupted_flight_data, number_of_solutions, verbose=False, plot=False, label={}, algorithm=AntColonySolver):
    
    start_time = time.perf_counter()
    
    solver           = algorithm(verbose=verbose, timeout=5, decay_rate= 0.05, ant_count= 64, sol_count = number_of_solutions)
    path_details     = solver.solve(flight_network, disrupted_flight_data)
    
    stop_time  = time.perf_counter()
    
    # print("N={:<3d} | {:5.0f} -> {:4.0f} | {:4.0f}s | ants: {:5d} | trips: {:4d} | "
    #       .format(len(cities), path_distance(cities), path_distance(result), (stop_time - start_time), solver.ants_used, solver.round_trips)
    #       + " ".join([ f"{k}={v}" for k,v in kwargs.items() ])
    # )

    print("CODE RUNTIME: ", stop_time-start_time)

    # if plot:
    #     show_path(result)


    return path_details

statistics_printer(flight_network=flight_network, disrupted_flight_data= disrupted_flight_data)
flight_network = reduce_network(flight_network, disrupted_flight_row)
path = AntColonyRunner(flight_network, disrupted_flight_data, verbose= 0, number_of_solutions= 10)

# print("\nThe Best Path for given Problem is:\n", path)
counter = 0
for i in path:
    if type(i)!=str:
        counter+=1
print("Number of Solutions Found: ", counter)
print(path)
