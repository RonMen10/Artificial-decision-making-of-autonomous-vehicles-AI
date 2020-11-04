import random

""" 
    Variables of the environment:
"""
# Simulation Frequency Hz
SIMULATION_FREQUENCY = 15

# inter arrival time of ego vehicles
time_difference = 40

# number of other cars in roundabout north
num_other_cars_north = 10

# number of cars in roundabout south
num_other_cars_south = 10

# number of ego vehicles
#num_ego_vehicles = 20
num_ego_vehicles = 10

# roundabout radius
raradius = 30

# roundabout centers
center_south = [0, 0]
center_north = [0, -210]


""" 
    Variables of the agents:
"""

# Starting position
START_POS = ("start", "east", 0)  
X = random.randint(0, 50)       
Y = random.randint(-35, 35)        

# Actions of the vehicles
ACTIONS = {0: 'IDLE',
           1: 'FASTER',
           2: 'SLOWER',
           3: 'STOP'}

# communication radius
com_radius = 70

# Vehicles Max Velocity
#MAX_VELOCITY = 11
MAX_VELOCITY = 11
""" Maximum reachable velocity [m/s] """

# Max Waiting Time of the Agent to enter the roundabout
MAX_WAITING_TIME = 30 # [s]

# Mutation probability and relevant distribution variance
mutation_probability = 0.5

# risk mutation bounds
upper_risk_bound = 1.
lower_risk_bound = 0.01

# CAF variable
CAF_FACTOR = 100

# Crash penalty
CRASH_PENALTY = 100

# Termination criterion: Number of entries in the roundabout
test_terminal = 200
training_terminal = 1000

# Path for storing results
path = '/home/swarmlab/Test code/cdm_project-1/highwayEnv/Tests'
