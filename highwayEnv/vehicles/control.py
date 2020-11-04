from __future__ import division, print_function
import numpy as np
import pandas as pd
import copy
import math
import random
import matplotlib.pyplot as plt
from highwayEnv.Vars_and_Methods import methods, variables
from highwayEnv.environment import roundabout
from highwayEnv.environment import archive


"""
    MAIN VEHICLE CONTROL FOR ALL CARS (WE COULD MAKE A COPY OF THIS TO SEPARATE CONTROL OF AGENTS AND OTHER CARS):
"""


class Vehicle(object):
    """
        A moving vehicle on a road, and its dynamics.

        The vehicle is represented by a dynamical system: a modified bicycle model.
        It's state is propagated depending on its steering and acceleration actions.
    """
    COLLISIONS_ENABLED = True
    """ Enable collision detection between vehicles """

    LENGTH = 5.0
    """ Vehicle length [m] """
    WIDTH = 2.0
    """ Vehicle width [m] """
    DEFAULT_VELOCITIES = [23, 25]
    """ Range for random initial velocities [m/s] """
    MAX_VELOCITY = variables.MAX_VELOCITY
    """ Maximum reachable velocity [m/s] """
    
    def __init__(self, road, position, heading=0, velocity=0):
        self.road = road
        self.position = np.array(position).astype('float')
        self.heading = heading
        self.velocity = velocity
        self.lane_index = self.road.network.get_closest_lane_index(self.position) if self.road else np.nan
        self.lane = self.road.network.get_lane(self.lane_index) if self.road else None
        self.action = {'steering': 0, 'acceleration': 0}
        self.log = []

    @classmethod
    def make_on_lane(cls, road, lane_index, longitudinal, velocity=0):
        """
            Create a vehicle on a given lane at a longitudinal position.
        
        :param road: the road where the vehicle is driving
        :param lane_index: index of the lane where the vehicle is located
        :param longitudinal: longitudinal position along the lane
        :param velocity: initial velocity in [m/s]
        :return: A vehicle with at the specified position
        """

        lane = road.network.get_lane(lane_index)
        if velocity is None:
            velocity = lane.speed_limit
        return cls(road, lane.position(longitudinal, 0), lane.heading_at(longitudinal), velocity)

    """ @classmethod
    def create_random(cls, road, velocity=None, spacing=1): """
    """
            Create a random vehicle on the road.

            The lane and /or velocity are chosen randomly, while longitudinal position is chosen behind the last
            vehicle in the road with density based on the number of lanes.

        :param road: the road where the vehicle is driving
        :param velocity: initial velocity in [m/s]. If None, will be chosen randomly
        :param spacing: ratio of spacing to the front vehicle, 1 being the default
        :return: A vehicle with random position and/or velocity
        """
    """ if velocity is None:
            velocity = road.np_random.uniform(Vehicle.DEFAULT_VELOCITIES[0], Vehicle.DEFAULT_VELOCITIES[1])
        default_spacing = 1.5*velocity
        _from = road.np_random.choice(list(road.network.graph.keys()))
        _to = road.np_random.choice(list(road.network.graph[_from].keys()))
        _id = road.np_random.choice(len(road.network.graph[_from][_to]))
        offset = spacing * default_spacing * np.exp(-5 / 30 * len(road.network.graph[_from][_to]))
        x0 = np.max([v.position[0] for v in road.vehicles]) if len(road.vehicles) else 3*offset
        x0 += offset * road.np_random.uniform(0.9, 1.1)
        v = cls(road, road.network.get_lane((_from, _to, _id)).position(x0, 0), 0, velocity)
        return v """

    def act(self, action=None):
        """
            Store an action to be repeated.

        :param action: the input action
        """
        if action:
            self.action = action

    def step(self, dt):
        """
            Propagate the vehicle state given its actions.

            Integrate a modified bicycle model with a 1st-order response on the steering wheel dynamics.
            If the vehicle is crashed, the actions are overridden with erratic steering and braking until complete stop.
            The vehicle's current lane is updated.

        :param dt: timestep of integration of the model [s]
        """
        self.action['steering'] = float(self.action['steering'])
        self.action['acceleration'] = float(self.action['acceleration'])
        if self.velocity > self.MAX_VELOCITY:
            self.action['acceleration'] = min(self.action['acceleration'], 1.0 * (self.MAX_VELOCITY - self.velocity))
        elif self.velocity < -self.MAX_VELOCITY:
            self.action['acceleration'] = max(self.action['acceleration'], 1.0 * (self.MAX_VELOCITY - self.velocity))

        v = self.velocity * np.array([np.cos(self.heading), np.sin(self.heading)])
        self.position += v * dt
        self.heading += self.velocity * np.tan(self.action['steering']) / self.LENGTH * dt
        self.velocity += self.action['acceleration'] * dt

        if self.road:
            self.lane_index = self.road.network.get_closest_lane_index(self.position)
            self.lane = self.road.network.get_lane(self.lane_index)

    def lane_distance_to(self, vehicle):
        """
            Compute the signed distance to another vehicle along current lane.

        :param vehicle: the other vehicle
        :return: the distance to the other vehicle [m]
        """
        if not vehicle:
            return np.nan
        return self.lane.local_coordinates(vehicle.position)[0] - self.lane.local_coordinates(self.position)[0]

    @property
    def direction(self):
        return np.array([np.cos(self.heading), np.sin(self.heading)])

    def front_distance_to(self, other):
        return self.direction.dot(other.position - self.position)
        
    def distance_to(self, other):
        euclidian = math.sqrt(math.pow(other.position[0] - self.position[0], 2) + math.pow(other.position[1] - self.position[1], 2))
        return euclidian

    def direction_to(self, other):
        return other.position - self.position

    def angle_to(self, other):
        if (np.cos(self.heading) * (other.position - self.position)[1]
                - np.sin(self.heading) * (other.position - self.position)[0] <= 0):
            angle = math.acos(self.direction.dot(other.position - self.position) / np.linalg.norm((other.position - self.position)))
            if (0 <= angle <= (math.pi / 2)):
                return angle
        return None 

    def to_dict(self, origin_vehicle=None):
        d = {
            'presence': 1,
            'x': self.position[0],
            'y': self.position[1],
            'vx': self.velocity * self.direction[0],
            'vy': self.velocity * self.direction[1],
            'cos_h': self.direction[0],
            'sin_h': self.direction[1]
        }
        if origin_vehicle:
            origin_dict = origin_vehicle.to_dict()
            for key in ['x', 'y', 'vx', 'vy']:
                d[key] -= origin_dict[key]
        return d

    def dump(self):
        """
            Update the internal log of the vehicle, containing:
                - its kinematics;
                - some metrics relative to its neighbour vehicles.
        """
        data = {
            'x': self.position[0],
            'y': self.position[1],
            'psi': self.heading,
            'vx': self.velocity * np.cos(self.heading),
            'vy': self.velocity * np.sin(self.heading),
            'v': self.velocity,
            'acceleration': self.action['acceleration'],
            'steering': self.action['steering']}

        if self.road:
            for lane_index in self.road.network.side_lanes(self.lane_index):
                lane_coords = self.road.network.get_lane(lane_index).local_coordinates(self.position)
                data.update({
                    'dy_lane_{}'.format(lane_index): lane_coords[1],
                    'psi_lane_{}'.format(lane_index): self.road.network.get_lane(lane_index).heading_at(lane_coords[0])
                })
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
            if front_vehicle:
                data.update({
                    'front_v': front_vehicle.velocity,
                    'front_distance': self.lane_distance_to(front_vehicle)
                })
            if rear_vehicle:
                data.update({
                    'rear_v': rear_vehicle.velocity,
                    'rear_distance': rear_vehicle.lane_distance_to(self)
                })

        self.log.append(data)

    def get_log(self):
        """
            Cast the internal log as a DataFrame.

        :return: the DataFrame of the Vehicle's log.
        """
        return pd.DataFrame(self.log)

    def __str__(self):
        return "{} #{}: {}".format(self.__class__.__name__, id(self) % 1000, self.position)

    def __repr__(self):
        return self.__str__() 
""" 
----------------------------------------OUR AGENTS CONTROL: ---------------------------------------------------------------
"""


class AgentVehicle(Vehicle):
    """
        A vehicle piloted by two low-level controller, allowing high-level actions
        such as cruise control and lane changes.

        - The longitudinal controller is a velocity controller;
        - The lateral controller is a heading controller cascaded with a lateral position controller.

        A controlled vehicle with a specified discrete range of allowed target velocities.
    """

    TAU_A = 0.6  # [s]
    TAU_DS = 0.2  # [s]
    PURSUIT_TAU = 1.5 * TAU_DS  # [s]
    KP_A = 1 / TAU_A
    KP_HEADING = 1 / TAU_DS
    KP_LATERAL = 1 / 0.5  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    MAX_ACC = 4 # [m/s2]

    DELTA_VELOCITY = variables.MAX_VELOCITY  # [m/s]

    SPEED_COUNT = 3  # []
    SPEED_MIN = 20  # [m/s]
    SPEED_MAX = 30  # [m/s]
    WEIGHT_RISK = None
    WEIGHT_TIME = None

    # Variables for learning
    risk_threshold = 0.0
    waiting_time = variables.MAX_WAITING_TIME
    max_waiting_time = variables.MAX_WAITING_TIME
    crashed = False
    stopped = False
    standing = False
    solution_selected = -1
    archive_updated = False
    exchange_done = False
    stats_updated = False
    hypervolume = 0.0
    non_dominated_time = []
    non_dominated_risk = []
    archive_index = -1

    def __init__(self,
                 road,
                 position,
                 heading=0,
                 velocity=0,
                 target_lane_index=None,
                 target_velocity=None,
                 route=None,
                 id=None,
                 timer=None,
                 waiting_time=20.0,
                 crashed=False):
        super(AgentVehicle, self).__init__(road, position, heading, velocity)
        self.target_lane_index = target_lane_index or self.lane_index
        self.target_velocity = target_velocity or self.velocity
        self.route = route
        #self.velocity_index = self.speed_to_index(self.target_velocity)
        #self.target_velocity = self.index_to_speed(self.velocity_index)
        self.id = id
        self.timer = timer or (np.sum(self.position) * np.pi) % 1
        self.risk_threshold = random.randint(50, 100) / 100 

    @classmethod
    def create_from(cls, vehicle):
        """
            Create a new vehicle from an existing one.
            The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, velocity=vehicle.velocity,
                target_lane_index=vehicle.target_lane_index, target_velocity=vehicle.target_velocity,
                route=vehicle.route)
        return v

    def plan_route_to(self, destination):
        """
            Plan a route to a destination in the road network

        :param destination: a node in the road network
        """
        path = self.road.network.shortest_path(self.lane_index[1], destination)
        if path:
            self.route = [self.lane_index] + [(path[i], path[i + 1], None) for i in range(len(path) - 1)]
        else:
            self.route = [self.lane_index]
        return self

    def act(self, action=None):
        """
            Perform a high-level action to change the desired lane or velocity.

            - If a high-level action is provided, update the target velocity and lane;
            - then, perform longitudinal and lateral control.

        :param action: a high-level action
        """
        self.follow_road()
        if action == "FASTER":
            self.target_velocity = self.DELTA_VELOCITY
            #self.velocity_index = self.speed_to_index(self.velocity) + 1
        elif action == "SLOWER":
            self.target_velocity -= self.DELTA_VELOCITY
            #self.velocity_index = self.speed_to_index(self.velocity) - 1
        elif action == "STOP":
            self.velocity = 0
            self.target_velocity = 0

        
        action = {'steering': self.steering_control(self.target_lane_index),
                  'acceleration': self.velocity_control(self.target_velocity)}
        super(AgentVehicle, self).act(action)
        
    def follow_road(self):
        """
           At the end of a lane, automatically switch to a next one.
        """
        if self.road.network.get_lane(self.target_lane_index).after_end(self.position):
            self.target_lane_index = self.road.network.next_lane(self.target_lane_index,
                                                                 route=self.route,
                                                                 position=self.position,
                                                                 np_random=self.road.np_random)

    def steering_control(self, target_lane_index):
        """
            Steer the vehicle to follow the center of an given lane.

        1. Lateral position is controlled by a proportional controller yielding a lateral velocity command
        2. Lateral velocity command is converted to a heading reference
        3. Heading is controlled by a proportional controller yielding a heading rate command
        4. Heading rate command is converted to a steering angle

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        target_lane = self.road.network.get_lane(target_lane_index)
        lane_coords = target_lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.velocity * self.PURSUIT_TAU
        lane_future_heading = target_lane.heading_at(lane_next_coords)

        # Lateral position control
        lateral_velocity_command = - self.KP_LATERAL * lane_coords[1]
        # Lateral velocity to heading
        heading_command = np.arcsin(np.clip(lateral_velocity_command/methods.not_zero(self.velocity), -1, 1))
        heading_ref = lane_future_heading + np.clip(heading_command, -np.pi/4, np.pi/4)
        # Heading control
        heading_rate_command = self.KP_HEADING * methods.wrap_to_pi(heading_ref - self.heading)
        # Heading rate to steering angle
        steering_angle = np.arctan(self.LENGTH / methods.not_zero(self.velocity) * heading_rate_command)
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        return steering_angle

    def velocity_control(self, target_velocity):
        """
            Control the velocity of the vehicle.

            Using a simple proportional controller with an upper limit.

        :param target_velocity: the desired velocity
        :return: an acceleration command [m/s2]
        """
        acc = self.KP_A * (target_velocity - self.velocity)
        if acc > self.MAX_ACC:
            acc = self.MAX_ACC
        return acc


    def predict_trajectory_constant_velocity(self, times):
        """
            Predict the future positions of the vehicle along its planned route, under constant velocity
        :param times: timesteps of prediction
        :return: positions, headings
        """
        coordinates = self.lane.local_coordinates(self.position)
        route = self.route or [self.lane_index]
        return zip(*[self.road.network.position_heading_along_route(route, coordinates[0] + self.velocity * t, 0)
                     for t in times])

    @classmethod
    def index_to_speed(cls, index):
        """
            Convert an index among allowed speeds to its corresponding speed
        :param index: the speed index []
        :return: the corresponding speed [m/s]
        """
        if cls.SPEED_COUNT > 1:
            return cls.SPEED_MIN + index * (cls.SPEED_MAX - cls.SPEED_MIN) / (cls.SPEED_COUNT - 1)
        else:
            return cls.SPEED_MIN

    @classmethod
    def speed_to_index(cls, speed):
        """
            Find the index of the closest speed allowed to a given speed.
        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - cls.SPEED_MIN) / (cls.SPEED_MAX - cls.SPEED_MIN)
        return np.int(np.clip(np.round(x * (cls.SPEED_COUNT - 1)), 0, cls.SPEED_COUNT - 1))

    def speed_index(self):
        """
            The index of current velocity
        """
        return self.speed_to_index(self.velocity)

    def predict_trajectory(self, actions, action_duration, trajectory_timestep, dt):
        """
            Predict the future trajectory of the vehicle given a sequence of actions.

        :param actions: a sequence of future actions.
        :param action_duration: the duration of each action.
        :param trajectory_timestep: the duration between each save of the vehicle state.
        :param dt: the timestep of the simulation
        :return: the sequence of future states
        """
        states = []
        v = copy.deepcopy(self)
        t = 0
        for action in actions:
            v.act(action)  # High-level decision
            for _ in range(int(action_duration / dt)):
                t += 1
                v.act()  # Low-level control action
                v.step(dt)
                if (t % int(trajectory_timestep / dt)) == 0:
                    states.append(copy.deepcopy(v))
        return states

    def projected_position(self):
        """ Get position of ego_vehicle projected into round about."""
        if self.lane_index == ("nxr", "sen", 0):
            position_of_interest = [variables.center_north[0] + self.position[0], 
                                   variables.center_north[1] + math.sqrt((variables.raradius + 4)**2 - self.position[0]**2)]
        elif self.lane_index == ("nes", "ne", 0):
            position_of_interest = [variables.center_south[0] + self.position[0], 
                                   variables.center_south[1] - math.sqrt((variables.raradius + 4)**2 - self.position[0]**2)]
        else: 
            return None
        return position_of_interest

    def distances_to_relevant_vehicles(self, other_vehicles, position_of_interest):
        """ Filter other_vehicles based on whether they are in the relevant field of fiew of the agent and if
            so calculate their distance to the projected_position of the agent in the round about."""
        vehicle_distance_list = []
        for vehicle in other_vehicles:
            if self.angle_to(vehicle) != None and self.distance_to(vehicle) <= variables.com_radius:
                distance = np.linalg.norm(position_of_interest - vehicle.position)
                alpha = 2 * np.arcsin(distance / (2 * (variables.raradius + 4)))
                vehicle_distance_list.append((vehicle, alpha * (variables.raradius + 4)))
                vehicle_distance_list.sort(key=lambda t: t[1])
        return vehicle_distance_list

    def distances_between_relevant_vehicles(self, position_of_interest, vehicle_distance_list):
        """ Calculate the distances between the other vehicles from the distances to the agent."""
        relevant_distances = []
        for i in range(0, len(vehicle_distance_list)):
                if i == 0:
                    distance = np.linalg.norm(vehicle_distance_list[i][0].position - position_of_interest)
                    alpha = 2 * np.arcsin(distance / (2 * (variables.raradius + 4)))
                    length_between = alpha * (variables.raradius + 4)
                    relevant_distances.append(length_between)     
                else:
                    distance = np.linalg.norm(vehicle_distance_list[i][0].position - vehicle_distance_list[i-1][0].position)
                    alpha = 2 * np.arcsin(distance / (2 * (variables.raradius + 4)))
                    length_between = alpha * (variables.raradius+4)
                    relevant_distances.append(length_between)
        return relevant_distances
        
    def waiting_time_and_risk(self, other_vehicles):
        """ Calculate waiting time and a risk assuming the agent enters the roundabout right after every new vehicle."""
        position_of_interest = self.projected_position()
        if position_of_interest is not None:
            vehicle_distance_list = self.distances_to_relevant_vehicles(other_vehicles, position_of_interest)
            relevant_distances = self.distances_between_relevant_vehicles(position_of_interest, vehicle_distance_list)
            # calculate the risk and the time based on s = t*v,
            # simplifying for s by just using distances from other cars to ego vehicle.
            risk=[]
            waiting_time=[]
            for i in range(0, len(relevant_distances)):
                risk.append(3 / relevant_distances[i])
                if i == 0:
                    waiting_time.append(0 + self.timer)
                else:
                    waiting_time.append((vehicle_distance_list[i-1][1]) / vehicle_distance_list[i-1][0].velocity + self.timer) 
            return waiting_time, risk
        else:
            return None, None

    def mutate_threshold(self, sigma, mutation_probability):
        """Change the agent risk threshold by mutation."""
        if  random.random() <= mutation_probability:
            mutated_threshold = self.risk_threshold - random.gauss(0, sigma)
            if mutated_threshold < variables.lower_risk_bound:
                mutated_threshold = variables.lower_risk_bound
            elif mutated_threshold > variables.upper_risk_bound:   
                mutated_threshold = variables.upper_risk_bound 
            self.risk_threshold = mutated_threshold

    def get_PO_solutions(self, time, risk):
        """Identify the pareto optimal solutions for the agent out of all possible ones.
        """
        dominated_risk=[]
        dominated_time=[]
        for i in range(0,len(time)):
            for j in range(0, len(time)):
                if (time[i]<=time[j] and risk[i]<risk[j]) or (time[i]<time[j] and risk[i]<=risk[j]):
                    if time[j] not in dominated_time:
                        dominated_risk.append(risk[j])
                        dominated_time.append(time[j])

        non_dominated_risk=self.diff_list(risk, dominated_risk)
        non_dominated_time=self.diff_list(time, dominated_time)

        return non_dominated_time, dominated_time, non_dominated_risk, dominated_risk

    def diff_list(self, li1, li2):
        """Get the difference between two lists.""" 
        li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2] 
        return li_dif 
    
    def visualize (self, non_dominated_time, dominated_time, non_dominated_risk, dominated_risk, roundabout):
        """Create a plot of the different options for the agent based on risk and time."""
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.4)
        hyperv = roundabout.archive.hypervolume(non_dominated_risk, non_dominated_time)
        plt.title("Risk/Waiting time decision options for Agent " + self.id)
        plt.xlabel("Waiting time \n \n Selected solution: " + str(self.solution_selected + 1) 
                    + "\n threshold: " + str(self.risk_threshold)
                    + "\n hypervolume: " + str(hyperv))
        plt.ylabel("Risk")
        #plt.ylim(0,2)
        plt.xlim(-0.3, 10)
        plt.axhline(y=self.risk_threshold)
        ax.scatter(non_dominated_time, non_dominated_risk, c='g')
        ax.scatter(dominated_time, dominated_risk, c='b')
        plt.show()

    def select_solution(self, non_dominated_time, non_dominated_risk):
        """Pick solution for agent out of the non-dominated possible options."""
        for index in range(len(non_dominated_risk)):
            if non_dominated_risk[index] <= self.risk_threshold:
                self.solution_selected = index
                break
        if self.solution_selected != -1:
            self.waiting_time = non_dominated_time[self.solution_selected]
            


    def first_arrival(self, roundabout):
        """Routine when the agent approaches a roundabout in the training phase."""
        self.act(variables.ACTIONS[3])
        self.stopped = True
        self.standing = True
        self.crashed = False
        self.archive_updated = False
        self.timer = 0
        self.waiting_time = variables.MAX_WAITING_TIME
        self.solution_selected = -1
        time, risk = self.waiting_time_and_risk(roundabout.other_vehicles)
        non_dominated_time, dominated_time, non_dominated_risk, dominated_risk = self.get_PO_solutions(time, risk)
        self.non_dominated_risk = non_dominated_risk
        self.non_dominated_time = non_dominated_time
        self.hypervolume = roundabout.archive.hypervolume(non_dominated_risk, non_dominated_time)
        roundabout.archive.search_archive(self, roundabout.risk_tol, roundabout.threshold_tol, roundabout.hv_tol)
        self.select_solution(non_dominated_time, non_dominated_risk)
        # self.visualize(non_dominated_time, dominated_time, non_dominated_risk, dominated_risk) 

    def first_arrival_test(self, roundabout):
        """Routine when the agent is already waiting at the roundabout entrance in the test phase."""
        self.act(variables.ACTIONS[3])
        self.stopped = True
        self.standing = True
        self.crashed = False
        self.stats_updated = False
        self.timer = 0
        self.waiting_time = variables.MAX_WAITING_TIME
        self.solution_selected = -1
        time, risk = self.waiting_time_and_risk(roundabout.other_vehicles)
        non_dominated_time, dominated_time, non_dominated_risk, dominated_risk = self.get_PO_solutions(time, risk)
        self.non_dominated_risk = non_dominated_risk
        self.non_dominated_time = non_dominated_time
        self.hypervolume = roundabout.archive.hypervolume(non_dominated_risk, non_dominated_time)
        roundabout.archive.search_test_archive(self, roundabout.risk_tol, roundabout.hv_tol, roundabout.distances_to_solutions)
        self.select_solution(non_dominated_time, non_dominated_risk)

    def still_standing(self, roundabout):
        """Routine when the agent is already waiting at the roundabout entrance in the training phase."""
        self.act(variables.ACTIONS[0]) 

        if self.timer >= self.max_waiting_time and self.solution_selected == -1: 
            self.mutate_threshold(roundabout.sigma, 1.)
            self.first_arrival(roundabout)
            
        elif self.timer >= self.waiting_time and self.solution_selected != -1:
            self.act(variables.ACTIONS[1])
            self.standing = False
     
        elif self.timer >= max(self.non_dominated_time[-1], self.non_dominated_time[0] + 1) and self.solution_selected == -1:
            self.waiting_time = max(self.non_dominated_time[-1], self.non_dominated_time[0] + 1) - self.non_dominated_time[0] + 5
            roundabout.archive.update_archive(self, roundabout.risk_tol, roundabout.threshold_tol, roundabout.hv_tol, self.archive_index,
                                        self.hypervolume, self.non_dominated_risk, 'waiting')
            self.archive_updated = False
            time, risk = self.waiting_time_and_risk(roundabout.other_vehicles)
            non_dominated_time, dominated_time, non_dominated_risk, dominated_risk = self.get_PO_solutions(time,risk)
            self.hypervolume = roundabout.archive.hypervolume(non_dominated_risk, non_dominated_time)
            self.non_dominated_risk = non_dominated_risk
            self.non_dominated_time = non_dominated_time
            roundabout.archive.search_archive(self, roundabout.risk_tol, roundabout.threshold_tol, roundabout.hv_tol)
            self.select_solution(non_dominated_time, non_dominated_risk)
            
            
    def still_standing_test(self, roundabout):
        """Routine when the agent is already waiting at the roundabout entrance in the test phase."""
        self.act(variables.ACTIONS[0]) 
        if self.timer >= self.waiting_time and self.solution_selected != -1:
            self.act(variables.ACTIONS[1])
            self.standing = False
                
        elif self.timer >= max(self.non_dominated_time[-1], self.non_dominated_time[0] + 1) and self.solution_selected == -1:
            time, risk = self.waiting_time_and_risk(roundabout.other_vehicles)
            non_dominated_time, dominated_time, non_dominated_risk, dominated_risk = self.get_PO_solutions(time,risk)
            self.hypervolume = roundabout.archive.hypervolume(non_dominated_risk, non_dominated_time)
            self.non_dominated_time = non_dominated_time
            self.non_dominated_risk = non_dominated_risk
            roundabout.archive.search_test_archive(self, roundabout.risk_tol, roundabout.hv_tol, roundabout.distances_to_solutions)
            self.select_solution(non_dominated_time, non_dominated_risk)

    def roundabout_entrance(self, roundabout):
        """Select routines for agent based on whether it just arrived at the roundabout or is already standing there for the training phase."""
        if (self.position[1] < -168 and self.stopped == False) or (self.position[1] > -44 and self.stopped == False):
            self.first_arrival(roundabout)
        elif (self.position[1] < -168 or self.position[1] > -44) and self.stopped == True and self.standing == True:
            self.still_standing(roundabout)

    def roundabout_entrance_test(self, roundabout):
        """Select routines for agent based on whether it just arrived at the roundabout or is already standing there for the test phase."""
        if (self.position[1] < -168 and self.stopped == False) or (self.position[1] > -44 and self.stopped == False):
            self.first_arrival_test(roundabout)
        elif (self.position[1] < -168 or self.position[1] > -44) and self.stopped == True and self.standing == True:
            self.still_standing_test(roundabout)
            

    def step(self, dt):
        """
            Step the simulation.

            Increases a timer used for decision policies, and step the vehicle dynamics.

        :param dt: timestep
        """
        self.timer += dt
        super(AgentVehicle, self).step(dt)

    def check_collision(self, other):
        """
            Check for collision with another vehicle.

        :param other: the other vehicle
        """
    
        if not self.COLLISIONS_ENABLED or not other.COLLISIONS_ENABLED or self.crashed or other is self:
            return

        # Fast spherical pre-check
        if np.linalg.norm(other.position - self.position) > self.LENGTH:
            return

        # Accurate rectangular check
        if methods.rotated_rectangles_intersect((self.position, 0.9*self.LENGTH, 0.9*self.WIDTH, self.heading),
                                              (other.position, 0.9*other.LENGTH, 0.9*other.WIDTH, other.heading)) and self.crashed == False:
            """ self.velocity = min(self.velocity, other.velocity)"""
            self.crashed = True 
            #print(self.crashCounter)




""" 
--------------------------------------------VEHICLES FOR THE ROUNDABOUT: ---------------------------------------------------------------------------------------------
"""
class simpleVehicle(Vehicle):
    """
        A vehicle using both a longitudinal and a lateral decision policies.

        - Longitudinal: the IDM model computes an acceleration given the preceding vehicle's distance and velocity.
        
        """
    TAU_A = 0.6  # [s]
    TAU_DS = 0.2  # [s]
    PURSUIT_TAU = 1.5*TAU_DS  # [s]
    KP_A = 1 / TAU_A
    KP_HEADING = 1 / TAU_DS
    KP_LATERAL = 1 / 0.5  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]

    DELTA_VELOCITY = 5  # [m/s]

    # Longitudinal policy parameters
    ACC_MAX = 6.0  # [m/s2]
    COMFORT_ACC_MAX = 3.0  # [m/s2]
    COMFORT_ACC_MIN = -5.0  # [m/s2]
    DISTANCE_WANTED = 1.0  # [m]
    TIME_WANTED = 0.0  # [s]
    DELTA = 4.0  # []


    def __init__(self,
                 road,
                 position,
                 heading=0,
                 velocity=0,
                 target_lane_index=None,
                 target_velocity=None,
                 route=None,
                 enable_lane_change=False,
                 timer=None):
        super(simpleVehicle, self).__init__(road, position, heading, velocity)
        self.target_lane_index = target_lane_index or self.lane_index
        self.target_velocity = target_velocity or self.velocity
        self.route = route
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position)*np.pi) % 1


    def act(self, action=None):
        """
            Perform a high-level action to change the desired lane or velocity.

            - If a high-level action is provided, update the target velocity and lane;
            - then, perform longitudinal and lateral control.

        :param action: a high-level action
        """

        if action == "FASTER":
            self.target_velocity += self.DELTA_VELOCITY
        elif action == "SLOWER":
            self.target_velocity -= self.DELTA_VELOCITY
        action = {}
        action = {'steering': self.steering_control(self.target_lane_index),
                  'acceleration': self.velocity_control(self.target_velocity)}
        super(simpleVehicle, self).act(action)

    def steering_control(self, target_lane_index):
        """
            Steer the vehicle to follow the center of an given lane.

        1. Lateral position is controlled by a proportional controller yielding a lateral velocity command
        2. Lateral velocity command is converted to a heading reference
        3. Heading is controlled by a proportional controller yielding a heading rate command
        4. Heading rate command is converted to a steering angle

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        target_lane = self.road.network.get_lane(target_lane_index)
        lane_coords = target_lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.velocity * self.PURSUIT_TAU
        lane_future_heading = target_lane.heading_at(lane_next_coords)

        # Lateral position control
        lateral_velocity_command = - self.KP_LATERAL * lane_coords[1]
        # Lateral velocity to heading
        heading_command = np.arcsin(np.clip(lateral_velocity_command/methods.not_zero(self.velocity), -1, 1))
        heading_ref = lane_future_heading + np.clip(heading_command, -np.pi/4, np.pi/4)
        # Heading control
        heading_rate_command = self.KP_HEADING * methods.wrap_to_pi(heading_ref - self.heading)
        # Heading rate to steering angle
        steering_angle = np.arctan(self.LENGTH / methods.not_zero(self.velocity) * heading_rate_command)
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        return steering_angle

    def velocity_control(self, target_velocity):
        """
            Control the velocity of the vehicle.

            Using a simple proportional controller.

        :param target_velocity: the desired velocity
        :return: an acceleration command [m/s2]
        """
        return self.KP_A * (target_velocity - self.velocity)

    def setDistanceWanted (self, distance):
        self.DISTANCE_WANTED = distance

    def randomize_behavior(self):
        pass
    
    def step(self, dt):
        """
            Step the simulation.

            Increases a timer used for decision policies, and step the vehicle dynamics.

        :param dt: timestep
        """
        self.timer += dt
        super(simpleVehicle, self).step(dt)

    def acceleration(self, ego_vehicle, front_vehicle=None, rear_vehicle=None):
        """
            Compute an acceleration command with the Intelligent Driver Model.

            The acceleration is chosen so as to:
            - reach a target velocity;
            - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            IDM vehicle, which is why this method is a class method. This allows an IDM vehicle to
                            reason about other vehicles behaviors even though they may not IDMs.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        if not ego_vehicle:
            return 0
        acceleration = self.COMFORT_ACC_MAX * (
                1 - np.power(max(ego_vehicle.velocity, 0) / methods.not_zero(ego_vehicle.target_velocity), self.DELTA))

        if front_vehicle:
            if type(front_vehicle) is type(self):
                d = ego_vehicle.lane_distance_to(front_vehicle)
                #Avoid going backwards:
                if (acceleration<0):
                    acceleration -= self.COMFORT_ACC_MAX * \
                        np.power(self.desired_gap(ego_vehicle, front_vehicle) / methods.not_zero(d), 2)
        return acceleration

    def desired_gap(self, ego_vehicle, front_vehicle=None):
        """
            Compute the desired distance between a vehicle and its leading vehicle.

        :param ego_vehicle: the vehicle being controlled
        :param front_vehicle: its leading vehicle
        :return: the desired distance between the two [m]
        """
        d0 = self.DISTANCE_WANTED + ego_vehicle.LENGTH / 2 + front_vehicle.LENGTH / 2
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = ego_vehicle.velocity - front_vehicle.velocity
        d_star = d0 + ego_vehicle.velocity * tau + ego_vehicle.velocity * dv / (2 * np.sqrt(ab))
        return d_star

    def maximum_velocity(self, front_vehicle=None):
        """
            Compute the maximum allowed velocity to avoid Inevitable Collision States.

            Assume the front vehicle is going to brake at full deceleration and that
            it will be noticed after a given delay, and compute the maximum velocity
            which allows the ego-vehicle to brake enough to avoid the collision.

        :param front_vehicle: the preceding vehicle
        :return: the maximum allowed velocity, and suggested acceleration
        """
        if not front_vehicle:
            return self.target_velocity
        d0 = self.DISTANCE_WANTED
        a0 = self.COMFORT_ACC_MIN
        a1 = self.COMFORT_ACC_MIN
        tau = self.TIME_WANTED
        d = max(self.lane_distance_to(front_vehicle) - self.LENGTH / 2 - front_vehicle.LENGTH / 2 - d0, 0)
        v1_0 = front_vehicle.velocity
        delta = 4 * (a0 * a1 * tau) ** 2 + 8 * a0 * (a1 ** 2) * d + 4 * a0 * a1 * v1_0 ** 2
        v_max = -a0 * tau + np.sqrt(delta) / (2 * a1)

        # Velocity control
        self.target_velocity = min(self.maximum_velocity(front_vehicle), self.target_velocity)
        acceleration = self.velocity_control(self.target_velocity)

        return v_max, acceleration

   




"""
    --------------The next classes are not used, they define certain behaviors of cars. --------------------------------
"""
class LinearVehicle(simpleVehicle):
    """
        A Vehicle whose longitudinal and lateral controllers are linear with respect to parameters
    """
    ACCELERATION_PARAMETERS = [0.3, 0.14, 0.8]
    STEERING_PARAMETERS = [AgentVehicle.KP_HEADING, AgentVehicle.KP_HEADING * AgentVehicle.KP_LATERAL]

    ACCELERATION_RANGE = np.array([0.5*np.array(ACCELERATION_PARAMETERS), 1.5*np.array(ACCELERATION_PARAMETERS)])
    STEERING_RANGE = np.array([np.array(STEERING_PARAMETERS) - np.array([0.07, 1.5]),
                               np.array(STEERING_PARAMETERS) + np.array([0.07, 1.5])])

    TIME_WANTED = 2.0

    def __init__(self, road, position,
                 heading=0,
                 velocity=0,
                 target_lane_index=None,
                 target_velocity=None,
                 route=None,
                 enable_lane_change=True,
                 timer=None):
        super(LinearVehicle, self).__init__(road,
                                            position,
                                            heading,
                                            velocity,
                                            target_lane_index,
                                            target_velocity,
                                            route,
                                            enable_lane_change,
                                            timer)

    def randomize_behavior(self):
        ua = self.road.np_random.uniform(size=np.shape(self.ACCELERATION_PARAMETERS))
        self.ACCELERATION_PARAMETERS = self.ACCELERATION_RANGE[0] + ua*(self.ACCELERATION_RANGE[1] -
                                                                        self.ACCELERATION_RANGE[0])
        ub = self.road.np_random.uniform(size=np.shape(self.STEERING_PARAMETERS))
        self.STEERING_PARAMETERS = self.STEERING_RANGE[0] + ub*(self.STEERING_RANGE[1] - self.STEERING_RANGE[0])

    def acceleration(self, ego_vehicle, front_vehicle=None, rear_vehicle=None):
        """
            Compute an acceleration command with a Linear Model.

            The acceleration is chosen so as to:
            - reach a target velocity;
            - reach the velocity of the leading (resp following) vehicle, if it is lower (resp higher) than ego's;
            - maintain a minimum safety distance w.r.t the leading vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            Linear vehicle, which is why this method is a class method. This allows a Linear vehicle to
                            reason about other vehicles behaviors even though they may not Linear.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        return np.dot(self.ACCELERATION_PARAMETERS, self.acceleration_features(ego_vehicle, front_vehicle, rear_vehicle))

    def acceleration_features(self, ego_vehicle, front_vehicle=None, rear_vehicle=None):
        vt, dv, dp = 0, 0, 0
        if ego_vehicle:
            vt = ego_vehicle.target_velocity - ego_vehicle.velocity
            d_safe = self.DISTANCE_WANTED + np.max(ego_vehicle.velocity, 0) * self.TIME_WANTED + ego_vehicle.LENGTH
            if front_vehicle:
                d = ego_vehicle.lane_distance_to(front_vehicle)
                dv = min(front_vehicle.velocity - ego_vehicle.velocity, 0)
                dp = min(d - d_safe, 0)
        return np.array([vt, dv, dp])

    def steering_control(self, target_lane_index):
        """
            Linear controller with respect to parameters.
            Overrides the non-linear controller ControlledVehicle.steering_control()
        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        steering_angle = np.dot(np.array(self.STEERING_PARAMETERS), self.steering_features(target_lane_index))
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        return steering_angle

    def steering_features(self, target_lane_index):
        """
            A collection of features used to follow a lane
        :param target_lane_index: index of the lane to follow
        :return: a array of features
        """
        lane = self.road.network.get_lane(target_lane_index)
        lane_coords = lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.velocity * self.PURSUIT_TAU
        lane_future_heading = lane.heading_at(lane_next_coords)
        features = np.array([methods.wrap_to_pi(lane_future_heading - self.heading) *
                             self.LENGTH / methods.not_zero(self.velocity),
                             -lane_coords[1] * self.LENGTH / (methods.not_zero(self.velocity) ** 2)])
        return features

class AggressiveVehicle(LinearVehicle):
    LANE_CHANGE_MIN_ACC_GAIN = 1.0  # [m/s2]
    MERGE_ACC_GAIN = 0.8
    MERGE_VEL_RATIO = 0.75
    MERGE_TARGET_VEL = 30
    ACCELERATION_PARAMETERS = [MERGE_ACC_GAIN / ((1 - MERGE_VEL_RATIO) * MERGE_TARGET_VEL),
                               MERGE_ACC_GAIN / (MERGE_VEL_RATIO * MERGE_TARGET_VEL),
                               0.5]

class DefensiveVehicle(LinearVehicle):
    LANE_CHANGE_MIN_ACC_GAIN = 1.0  # [m/s2]
    MERGE_ACC_GAIN = 1.2
    MERGE_VEL_RATIO = 0.75
    MERGE_TARGET_VEL = 30
    ACCELERATION_PARAMETERS = [MERGE_ACC_GAIN / ((1 - MERGE_VEL_RATIO) * MERGE_TARGET_VEL),
                               MERGE_ACC_GAIN / (MERGE_VEL_RATIO * MERGE_TARGET_VEL),
                               2.0]



""" 
                OBSTACLE CLASS, IN CASE WE WANT TO PUT OBSTACLES IN THE ROAD (NOT NEEDED)
"""
class Obstacle(Vehicle):
    """
        A motionless obstacle at a given position.
    """

    def __init__(self, road, position, heading=0):
        super(Obstacle, self).__init__(road, position, velocity=0, heading=heading)
        self.target_velocity = 0
        self.LENGTH = self.WIDTH
