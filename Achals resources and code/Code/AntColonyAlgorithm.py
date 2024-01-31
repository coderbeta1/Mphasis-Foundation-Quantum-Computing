import numpy as np
import random
import time

# VARIABLES 
max_flight_delay = 6
disrupted_time = None

# OBJECTIVE FUNCTION

# CONSTRAINTS
'''
1. If (flight_delay > 6):   Stop/Add infinity weight
'''

'''
Initialize a random network of Flights
'''
def create_flights(number_of_airports: int) -> np.ndarray:
    network = np.zeros((number_of_airports,number_of_airports))
    for airport1 in range(len(network)):
        for airport2 in range(len(network)):
            connection = np.random.choice(a= (0,1), size= 1, p= (0.3, 0.7))
            if(connection == 1):
                network[airport1][airport2] = 1     
    return network
def create_flight_network(number_of_airports: int, time_span: int) -> dict:
    time_network={}
    for i in range(1,time_span+1):
        current_network= create_flights(number_of_airports=number_of_airports)
        time_network[i]= current_network
    return time_network
def choose_source_destination(number_of_airports: int):
    source_airport = random.randint(0, number_of_airports)
    destination_airport = random.randint(0, number_of_airports)

    if (source_airport==destination_airport):
        destination_airport= (destination_airport+1)%number_of_airports
    return source_airport%number_of_airports, destination_airport%number_of_airports

number_of_airports = random.randint(100,150)
time_span = random.randint(20,50)
flight_network = create_flight_network(number_of_airports= number_of_airports, time_span=time_span)
source_airport, destination_airport = choose_source_destination(number_of_airports=number_of_airports)

print("SIMULATION STATISTICS: \n")
print("Airports Count: ", number_of_airports)
print("Time Span: ", time_span)
print("Source Node: ", source_airport)
print("Destination Node: ", destination_airport)
print()

# x= input()

'''
Create a random disruption in the network of Flights
'''

def create_flight_disruption(flight_network: dict, source_airport: int, destination_airport: int) -> dict:
    number_of_airports = flight_network[1].shape[0]
    time_span = len(flight_network)

    disrupted_time= random.randint(1,int(time_span/2))

    print("Disrupted Time: ", disrupted_time)
    # x=input()

    # Apply Disruption
    for i in range(1,disrupted_time):
        flight_network[i]= np.zeros((number_of_airports,number_of_airports))
    flight_network[disrupted_time][source_airport][destination_airport]=0

    return flight_network, disrupted_time

flight_network, disrupted_time=create_flight_disruption(flight_network=flight_network, source_airport=source_airport, destination_airport=destination_airport)

'''
Create Ant Colony for Optimization
'''

class AntColonySolver:
    def __init__(self,
                #  cost_fn=0,                         

                 time=0,                  # run for a fixed amount of time
                 min_time=0,              # minimum runtime
                 timeout=0,               # maximum time in seconds to run for
                 stop_factor=2,           # how many times to redouble effort after new new best path
                #  min_round_trips=10,      # minimum number of round trips before stopping
                #  max_round_trips=0,       # maximum number of round trips before stopping                 
                 min_ants=0,              # Total number of ants to use
                 max_ants=0,              # Total number of ants to use
                 
                 ant_count=2,            # this is the bottom of the near-optimal range for numpy performance
                 ant_speed=1,             # how many steps do ants travel per epoch

                 distance_power=1,        # power to which distance affects pheromones                 
                 pheromone_power=1.25,    # power to which differences in pheromones are noticed
                 decay_power=0,           # how fast do pheromones decay
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
        self.decay_power     = float(decay_power)
        self.reward_power    = float(reward_power)
        self.best_path_smell = float(best_path_smell)
        self.start_smell     = float(start_smell or 10**self.distance_power)
        
        self.verbose         = int(verbose)
        self._initalized     = False

        # if self.min_round_trips and self.max_round_trips: self.min_round_trips = min(self.min_round_trips, self.max_round_trips)
        if self.min_ants and self.max_ants:               self.min_ants        = min(self.min_ants, self.max_ants)

        
    def solve_initialize(
            self,
            problem_path: dict,
    ) -> None:
        


        # ### Cache of distances between nodes
        # self.distances = {
        #     source: {
        #         dest: self.cost_fn(source, dest)
        #         for dest in problem_path
        #     }
        #     for source in problem_path
        # }

        # ### Cache of distance costs between nodes - division in a tight loop is expensive
        # self.distance_cost = {
        #     source: {
        #         dest: 1 / (1 + self.distances[source][dest]) ** self.distance_power
        #         for dest in problem_path
        #     }
        #     for source in problem_path
        # }

        # ### This stores the pheromone trail that slowly builds up
        # self.pheromones = {
        #     source: {
        #         # Encourage the ants to start exploring in all directions and furthest nodes
        #         dest: self.start_smell
        #         for dest in problem_path
        #     }
        #     for source in problem_path
        # }
        
        ### Sanitise input parameters
        # if self.ant_count <= 0:
        #     self.ant_count = len(problem_path)
        # if self.ant_speed <= 0:
        #     self.ant_speed = np.median(list(chain(*[ d.values() for d in self.distances.values() ]))) // 5
        # self.ant_speed = int(max(1,self.ant_speed))
        
        ### Heuristic Exports
        # self.ants_used   = 0
        # self.epochs_used = 0
        # self.round_trips = 0

        self._initalized = True   
        return     

    def solve(self,
              flight_network: dict,
              pheromone_network: dict,
              source_airport: int,
              destination_airport: int,
              disruption_time: int,
              max_time_in_network: int,
              restart=False,
    ) -> list[tuple[int,int]]:
        
        # If it is (first run) or (want to reset) Initialize 
        if restart or not self._initalized:
            self.solve_initialize(flight_network)

        '''
        Define the Ant(Agent) Population and Initialize
        '''
        ants = {
            "distance":    np.zeros((self.ant_count,)).astype('int32'),
            "path":        [ [source_airport]   for n in range(self.ant_count) ],
            "remaining":   [ set(list(range(flight_network[1].shape[0])))-{source_airport} for n in range(self.ant_count) ],
            "path_cost":   np.zeros((self.ant_count,)).astype('int32'),
            # "round_trips": np.zeros((self.ant_count,)).astype('int32'),
        }

        best_path       = None
        best_path_cost  = np.inf
        best_epochs     = []
        epoch           = 0
        time_start      = time.perf_counter()

        while True:
            epoch += 1
            # print("epoch: ", epoch)
            # current_pheromone_network = pheromone_network.copy()

            '''
            Simulated Walk of the Ants
            '''
            for ant in range(self.ant_count):
                timer=0
                while((not ants['remaining'][ant]) or (ants['path'][ant][-1] == destination_airport) or (max_time_in_network>disruption_time+timer)):
                    '''
                    Solving for Next Best Node
                    '''
                    # this_node = ants['path'][ant][-1]
                    next_node = self.next_node(ants, ant, epoch, flight_network, disruption_time)
                    # if next_node==-1:

                    # ants['distance'][i]  = self.distances[ this_node ][ next_node ]
                    ants['distance'][ant]  = 1 # Change this to the optimization function
                    if next_node!=-1:  ants['remaining'][ant] = ants['remaining'][ant] - {next_node}
                    ants['path_cost'][ant] = ants['path_cost'][ant] + ants['distance'][ant]
                    ants['path'][ant].append(next_node)
                    if next_node==-1:
                        break
                    if(not ants['remaining'][ant]):
                        break
                    if next_node==destination_airport:
                        break

                if (ants['path_cost'][ant] < best_path_cost) and (next_node == destination_airport):
                        best_path_cost = ants['path_cost'][ant]
                        best_path      = ants['path'][ant]
                        best_epochs   += [ epoch ]
                        
            if self.verbose:
                print({
                    "path_cost":   best_path_cost,
                    # "ants_used":   self.ants_used,
                    "epoch":       epoch,
                    # "round_trips": ants['round_trips'][ant] + 1,
                    "clock":       time.perf_counter() - time_start,
                })
            '''
            Leave Pheromone Trail
            
            1. Doing this only after ants arrive home improves initial exploration
            '''

            for ant in range(self.ant_count):
                timer=0
                for node in range(len(ants['path'][ant])-1):
                    if node!=-1: pheromone_network[disruption_time+timer][ants['path'][ant][node]][ants['path'][ant][node+1]] += (1-0.5)*pheromone_network[disruption_time+timer][ants['path'][ant][node]][ants['path'][ant][node+1]]+(1/ants['path_cost'][ant])
                
                # Reset the Ant Values
                ants["distance"][ant]     = 0
                ants["path"][ant]         = [source_airport]
                ants["remaining"][ant]    = set(list(range(flight_network[1].shape[0])))-{source_airport}
                ants["path_cost"][ant]    = 0       

            '''
            Check Conditions of Termination

            1. Atleast One Solution
            2. Runtime Finished
            3. Max Trips Reached
            4. Ants Used more than Max
            5. Good enough solution for current Epoch
            '''
            
            # Always wait for at least 1 solutions (note: 2+ solutions are not guaranteed)
            if not len(best_epochs): continue 
            
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
            if True: continue
            
        '''
        Run Statistics and Analysis
        '''
        self.epochs_used = epoch
        # self.round_trips = np.max(ants["round_trips"])
        return best_path

    def next_node(self, ants, index, time_stamp, possible_paths, disruption_time):
        this_node = ants['path'][index][-1]
        current_choices = possible_paths[index+disruption_time][this_node]
        x=[]
        # print("Here", current_choices, this_node)
        for a in range(len(current_choices)):
            # print(current_choices[a])
            if(current_choices[a]==1 and a!=this_node and (a not in ants['path'][index])):
                x.append(a)
        # current_choices= [a for a in range(len(current_choices)) if (int(current_choices[a])==1 and a!=this_node)]
        current_choices = x

        weights = [] 
        weights_sum = 0
        # print(current_choices, this_node)

        for next_node in current_choices:
            if next_node == this_node: continue

            '''
            OPTIMIZATION FUNCTION (to maximize)
            '''
            # reward = (
            #         self.pheromones[this_node][next_node] ** self.pheromone_power
            #         * self.distance_cost[this_node][next_node]  # Prefer shorter paths
            # )
            reward = 1 

            '''
            OUTLIER CHECKING
            '''
            # if(time_stamp>max_flight_delay or 0):
            #     reward = 0
            # if(not ants['remaining'][index]):
            #     reward=0
            
            weights.append(reward)
            weights_sum += reward
        # if weights_sum==0: print(weights)
        weights=[a/weights_sum for a in weights]

        # print(weights)
        # print("Current Choices: ",current_choices)

        
        # Pick a random path in proportion to the weight of the pheromone
        if len(current_choices)!=0: next_node= np.random.choice((current_choices), size=1, p=weights)
        else: return -1
        # print("Next: ",next_node)
        return next_node[0]


def AntColonyRunner(flight_network:dict, verbose=False, plot=False, label={}, algorithm=AntColonySolver):

    pheromone_network= flight_network.copy()

    solver     = algorithm(verbose=verbose, timeout=1)
    start_time = time.perf_counter()
    result     = solver.solve(flight_network,pheromone_network, source_airport, destination_airport, disrupted_time, time_span)
    stop_time  = time.perf_counter()
    
    # print("N={:<3d} | {:5.0f} -> {:4.0f} | {:4.0f}s | ants: {:5d} | trips: {:4d} | "
    #       .format(len(cities), path_distance(cities), path_distance(result), (stop_time - start_time), solver.ants_used, solver.round_trips)
    #       + " ".join([ f"{k}={v}" for k,v in kwargs.items() ])
    # )

    print("Code Runtime: ", stop_time-start_time)
    print("Shortest Path: ", result)
    # if plot:
    #     show_path(result)

    return result

# distance = 5

AntColonyRunner(flight_network, True)
# print(np.zeros((10,)).astype('int32'))
# x=set(list(range(10)))
# print(x.)
# x[5]=1
# x=[a for a in range(x.shape[0]) if x[a]==0]
# print(x)

# for i in range(10):
#     print(AntColonyRunner(cities=1))