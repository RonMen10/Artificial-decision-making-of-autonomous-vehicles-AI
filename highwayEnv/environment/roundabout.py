from __future__ import division, print_function, absolute_import
import copy
import gym
import numpy as np
import random
import pandas as pd
import datetime
import pickle
import matplotlib.pyplot as plt
from gym import spaces
from gym.utils import seeding
from highwayEnv.Vars_and_Methods import methods, variables
from highwayEnv.environment.observation import observation_factory
from highwayEnv.vehicles.control import AgentVehicle, simpleVehicle
from gym.envs.registration import register
from highwayEnv.road.lane import LineType, StraightLane, CircularLane, SineLane, AbstractLane
from highwayEnv.road.road import Road, RoadNetwork
from highwayEnv.analytics import build_boxplot, build_scatterplot
from highwayEnv.environment.archive import Archive


np.random.seed(1)
random.seed(1)
theList = [0, 1, 2]


class RoundaboutEnv(gym.Env):
    """
        A generic environment for a vehicle driving on a roundabout.
        The action space is fixed.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    ACTIONS = variables.ACTIONS
    """
        A mapping of action indexes to action labels
    """
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}
    """
        A mapping of action labels to action indexes
    """

    SIMULATION_FREQUENCY = variables.SIMULATION_FREQUENCY
    """
        The frequency at which the system dynamics are simulated [Hz]
    """

    PERCEPTION_DISTANCE = 5.0 * AgentVehicle.SPEED_MAX
    """
        The maximum distance of any vehicle present in the observation [m]
    """
    TIME_DIFFERENCE = variables.time_difference
    """ 
        Inter arrival time of agent vehicles 
    """

    # For plots
    plot_counter = 0

    DEFAULT_CONFIG = {
        "observation": {
            "type": "Kinematics"
        },
        "policy_frequency": 1,  # [Hz]
        "other_vehicles_type": "highwayEnv.vehicles.control.simpleVehicle",
        "incoming_vehicle_destination": None,
        "screen_width": 800,
        "screen_height": 750,
        "centering_position": [0.9, 0.1]
    }



    def __init__(self, config=None):
        # Variables to configure for the Paralelization
        self.sigma = 0
        self.risk_tol = 0
        self.threshold_tol = 0
        self.hv_tol = 0
        self.total_attempts = 0

        # Configuration
        self.config = config
        if not self.config:
            self.config = self.DEFAULT_CONFIG.copy()

        # Seeding
        self.np_random = None
        self.seed(1)

        # Scene
        self.road = None
        self.other_vehicles = []
        self.ego_vehicles = []

        # Spaces
        self.observation = None
        self.define_spaces()

        # Running is done
        self.done = False

        # Rendering
        self.viewer = None
        self.automatic_rendering_callback = None
        self.should_update_rendering = True
        self.rendering_mode = 'human'
        self.offscreen = self.config.get("offscreen_rendering", True)
        self.enable_auto_render = True

        # Number of steps performed in the iteration
        self.steps = 0

        # Initialize the roundabout with initial state
        self.reset()

        # Variables for the Archive
        self.number_of_entries = []
        self.number_of_crashes = []
        self.archive = Archive()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def define_spaces(self):
        self.action_space = spaces.Discrete(len(self.ACTIONS))

        if "observation" not in self.config:
            raise ValueError("The observation configuration must be defined")
        self.observation = observation_factory(self, self.config["observation"])
        self.observation_space = self.observation.space()

    def _is_terminal(self):
        """
            Check whether the current state is a terminal state
        :return: is the state terminal
        """
        """
            The episode is over when the steps defined in the Variables.py are reached.
        """

        #print(self.total_attempts)
        if (self.total_attempts >= variables.training_terminal):
            pickle.dump(self.archive.get_archive(), open(variables.path + '/Archives/{0}_{1}_{2}_{3}.pkl'.format(self.risk_tol, self.threshold_tol, self.hv_tol, self.sigma), 'wb'))
            (pd.DataFrame(self.number_of_entries)).to_csv(variables.path + '/TrainingStatistics/convStatesOverTime_{0}_{1}_{2}_{3}.csv'.format(
                self.risk_tol, self.threshold_tol, self.hv_tol, self.sigma), index=None, header=True)
            (pd.DataFrame([self.steps])).to_csv(variables.path + '/TrainingStatistics/steps_{0}_{1}_{2}_{3}.csv'.format(
                self.risk_tol, self.threshold_tol, self.hv_tol, self.sigma), index=None, header=True)
            return True

    def reset(self):
        """
            Reset the environment to it's initial configuration
        :return: the observation of the reset state
        """
        self.other_vehicles = []
        self.ego_vehicles = []
        self._make_road()
        self._make_other_vehicles()
        self.define_spaces()
        return self.observation.observe()

    def step(self, action):
        """
            Perform an action and step the environment dynamics.

            The action is executed by the Agent vehicle, and all other vehicles on the road performs their default
            behaviour for several simulation timesteps until the next decision making step.
        :param int action: the action performed by the Agent vehicle
        :return: if is done or not (Terminal state)
        """
        self._make_ego_vehicles()
        if self.road is None or self.ego_vehicles is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")
        self.steps += 1
        for agent in self.ego_vehicles:
            # Set the route of the agents to the next roundabout.
            if agent.lane_index == ("nxr", "sen", 0) and not agent.route:
                agent.plan_route_to("nes")
            elif agent.lane_index == ("nes", "ne", 0) and not agent.route:
                agent.plan_route_to("nxr")

            # After leaving roundabout
            if agent.lane_index in [("sxn","ner", 0), ("nx","nxs", 0)] and agent.archive_updated == False:
                self.archive.update_archive(agent, self.risk_tol, self.threshold_tol, self.hv_tol, agent.archive_index,
                                        agent.hypervolume, agent.non_dominated_risk, 'normal')
                agent.mutate_threshold(self.sigma, variables.mutation_probability)
                self.total_attempts += 1
                agent.stopped = False
                agent.exchange_done = False

            # On entrance lane to roundabout
            if agent.lane_index in [("nxr", "sen", 0), ("nes", "ne", 0)]:
                agent.roundabout_entrance(self)
            else:
                agent.act(self.ACTIONS[action])

            #when leaving the roundabout (By Ronald)
            if agent.lane_index in [("nxs", "mid", 0)]:
                if len(self.archive.get_archive()['id']) > 0 and len(theList[1]) > 1:
                    id1, id2 = self.archive.idValues()
                    self.archive.clusterUpdateArchive(id1, id2)
                    self.exchangeStateShifter(id1)
                    #self.exchangeStateShifter(id2)


        # Save the archive every 100 steps.
        if (self.steps % 100 == 0):
            #build_scatterplot(self, self.ego_vehicles)
            keys=['hypervolume', 'first_risk','counter','threshold','cumWaitTime','cumCrashes','avgFitness','case']
            archive_df=pd.DataFrame({ key: self.archive.get_archive()[key] for key in keys })
            archive_df = archive_df.sort_values(by=['hypervolume','threshold'])
            self.number_of_entries.append((self.steps, len(archive_df.index)))
            self.number_of_crashes.append((self.steps, sum(self.archive.get_archive()['cumCrashes'])))

        self.road.act()
        self.road.step(1 / self.SIMULATION_FREQUENCY, self.ego_vehicles, self.other_vehicles)

        methods._automatic_rendering(self)

        self.enable_auto_render = False
        terminal = self._is_terminal()
        theList[1] = self.index_for_exchange_noTrue() #To get only the individuals that have no get information from other on this iteration
        theList[2] = self.index_for_exchange_True() #To get all the individuals in the exchange area
        return terminal

    def index_for_exchange_noTrue(self):
        agent_id = []
        self.lane_index = self.road.network.get_lane
        for agent in self.ego_vehicles:
            if agent.lane_index in [("nx", "nxs", 0), ("nxs", "mid", 0), ("mid", "nxr", 0), ("ner", "nes", 0),
                                    ("nes", "ne", 0)] and agent.exchange_done != True:
                agent_id.append(agent.id)
        return agent_id


    def index_for_exchange_True(self):
        agent_id = []
        self.lane_index = self.road.network.get_lane
        for agent in self.ego_vehicles:
            if agent.lane_index in [("nx", "nxs", 0), ("nxs", "mid", 0), ("mid", "nxr", 0), ("ner", "nes", 0),
                                    ("nes", "ne", 0)]:
                agent_id.append(agent.id)
        return agent_id


    def exchangeStateShifter(self, agentId):
        for agent in self.ego_vehicles:
            if int(agent.id) == agentId:
                print(agent.id, str(agent.exchange_done))
                agent.exchange_done = True
                print(agent.id, str(agent.exchange_done))

    def close(self):
        """
            Close the environment.

            Will close the environment viewer if it exists.
        """
        self.done = True
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = None
        plt.close('all')
        #return self.done
        #self.close()


    def get_available_actions(self):
        """
            Get the list of currently available actions.

            Lane changes are not available on the boundary of the road, and velocity changes are not available at
            maximal or minimal velocity.

        :return: the list of available actions
        """
        actions = [self.ACTIONS_INDEXES['IDLE']]
        if self.vehicle.velocity_index < self.vehicle.SPEED_COUNT - 1:
            actions.append(self.ACTIONS_INDEXES['FASTER'])
        if self.vehicle.velocity_index > 0:
            actions.append(self.ACTIONS_INDEXES['SLOWER'])
        actions.append(self.ACTIONS_INDEXES['STOP'])
        return actions

        """ 
            vehicle collections:
            all cars including ego-vehicles: RoundaboutEnv.road.vehicles
            ego vehicles: RoundaboutEnv.vehicle
        """
    def _make_road(self):
        # Circle lanes: (s)outh/(e)ast/(n)orth/(w)est (e)ntry/e(x)it.
        center = variables.center_south  # [m]
        centerNorth = variables.center_north  # [m]
        radius = variables.raradius  # [m]
        alpha = 20  # [deg]

        net = RoadNetwork()
        radii = radius+4
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [c, c]

        """  
            Creation of south Roundabout
        """
        net.add_lane("se", "ex", CircularLane(center, radii, rad(90-alpha), rad(alpha), clockwise=False, line_types=line))
        net.add_lane("ex", "ee", CircularLane(center, radii, rad(alpha), rad(-alpha), clockwise=False, line_types=line))
        net.add_lane("ee", "nx", CircularLane(center, radii, rad(-alpha), rad(-90+alpha), clockwise=False, line_types=line))
        net.add_lane("nx", "ne", CircularLane(center, radii, rad(-90+alpha), rad(-90-alpha), clockwise=False, line_types=line))
        net.add_lane("ne", "wx", CircularLane(center, radii, rad(-90-alpha), rad(-180+alpha), clockwise=False, line_types=line))
        net.add_lane("wx", "we", CircularLane(center, radii, rad(-180+alpha), rad(-180-alpha), clockwise=False, line_types=line))
        net.add_lane("we", "sx", CircularLane(center, radii, rad(180-alpha), rad(90+alpha), clockwise=False, line_types=line))
        net.add_lane("sx", "se", CircularLane(center, radii, rad(90+alpha), rad(90-alpha), clockwise=False, line_types=line))

        """ 
            Creation of North Roundabout
        """

        # Circle lanes: (s)outh/(e)ast/(n)orth/(w)est + (e)ntry/e(x)it + n.
        net.add_lane("sen", "exn", CircularLane(centerNorth, radii, rad(90-alpha), rad(alpha), clockwise=False, line_types=line))
        net.add_lane("exn", "een", CircularLane(centerNorth, radii, rad(alpha), rad(-alpha), clockwise=False, line_types=line))
        net.add_lane("een", "nxn", CircularLane(centerNorth, radii, rad(-alpha), rad(-90+alpha), clockwise=False, line_types=line))
        net.add_lane("nxn", "nen", CircularLane(centerNorth, radii, rad(-90+alpha), rad(-90-alpha), clockwise=False, line_types=line))
        net.add_lane("nen", "wxn", CircularLane(centerNorth, radii, rad(-90-alpha), rad(-180+alpha), clockwise=False, line_types=line))
        net.add_lane("wxn", "wen", CircularLane(centerNorth, radii, rad(-180+alpha), rad(-180-alpha), clockwise=False, line_types=line))
        net.add_lane("wen", "sxn", CircularLane(centerNorth, radii, rad(180-alpha), rad(90+alpha), clockwise=False, line_types=line))
        net.add_lane("sxn", "sen", CircularLane(centerNorth, radii, rad(90+alpha), rad(90-alpha), clockwise=False, line_types=line))


        # Access lanes: (r)oad/(s)ine
        access = 150  # [m]
        dev = 120  # [m]
        a = 5  # [m]
        delta_st = 0.20*dev  # [m]

        delta_en = dev-delta_st
        w = 2*np.pi/dev

        #Lanes to create the start of the ego_cars
        net.add_lane("start", "east", StraightLane([access, -access/2], [70, -access/2], width=60, line_types=line, speed_limit=50))
        net.add_lane("east", "easte", StraightLane([70, -access/2], [30,-access/2], line_types=line))
        net.add_lane("easte", "mid", SineLane([30, -access/2-a], [4, -access/2-a], a, w, -np.pi / 2, line_types=line))
        net.add_lane("none", "none", StraightLane([70, -access/2-2], [70,-105], width=0, line_types=line))
        net.add_lane("none", "none", StraightLane([70, -73], [70,-45], width=0, line_types=line))

        #Lines to go to the north roundabout
        net.add_lane("ner", "nes", StraightLane([-2, -access], [-2, -dev / 2], line_types=[s, c]))
        net.add_lane("nes", "ne", SineLane([-2 - a, -dev / 2], [-2 - a, -dev / 2 + delta_st], a, w, -np.pi / 2, line_types=line))

        #Lines to go from south to north, mid split the straightline so the cars can enter from the pool
        net.add_lane("nx", "nxs", SineLane([2 + a, dev / 2 - delta_en], [2 + a, -dev / 2], a, w, -np.pi / 2 + w * delta_en, line_types=line))
        net.add_lane("nxs", "mid", StraightLane([2, -dev / 2], [2, -access/2], line_types=[n, c]))
        net.add_lane("mid", "nxr", StraightLane([2, -access/2], [2, -access], line_types=[n, c]))

        #Lines for enter and exit the north round about
        net.add_lane("sxn", "ner", SineLane([-2-a, -access-delta_st], [-2-a, -access], a, w, -np.pi/2+w*delta_en, line_types=line))
        net.add_lane("nxr", "sen", SineLane([2+a, -access], [2+a, -access-delta_st], a, w, -np.pi/2, line_types=line))
        #net.add_lane("sem", "sen", SineLane([2+a, -access-delta_st+5], [2+a, -access-delta_st], a, w, -np.pi/2+w*delta_en, line_types=line))
        road = Road(network=net, np_random=self.np_random)
        self.road = road

    def _make_other_vehicles(self):
        """
            Populate the roundabouts with dummy vehicles (Blue vehicles)
        """
        position_deviation = 2

        # Other vehicles

        other_vehicles_type = methods.class_from_path(self.config["other_vehicles_type"])

        #Roundabout south
        for i in range(1, variables.num_other_cars_south+1):
            if i == 1:
                vehicle = other_vehicles_type.make_on_lane(self.road,
                                                       ("ne", "wx", 0),
                                                       longitudinal=20 * i + self.np_random.randn() * position_deviation,
                                                       velocity=variables.MAX_VELOCITY)
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)
                self.other_vehicles.append(vehicle)
                #vehicle.setDistanceWanted(30)

            else:
                vehicle = other_vehicles_type.make_on_lane(self.road,
                                                       ("ne", "wx", 0),
                                                       longitudinal=20*i + self.np_random.randn() * position_deviation,
                                                       velocity=variables.MAX_VELOCITY)
                vehicle.randomize_behavior()
                vehicle.setDistanceWanted(29 * np.random.random_sample() + 1)
                self.road.vehicles.append(vehicle)
                self.other_vehicles.append(vehicle)
                #vehicle.setDistanceWanted(0)

        #Roundabout north
        for i in range(1, variables.num_other_cars_north+1):
            if i == 1 :
                vehicle = other_vehicles_type.make_on_lane(self.road,
                                                       ("wen", "sxn", 0),
                                                       longitudinal=20 * i + self.np_random.randn() * position_deviation,
                                                       velocity=variables.MAX_VELOCITY)
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)
                self.other_vehicles.append(vehicle)


            else:
                vehicle = other_vehicles_type.make_on_lane(self.road,
                                                       ("wen", "sxn", 0),
                                                       longitudinal=20 * i + self.np_random.randn() * position_deviation,
                                                       velocity=variables.MAX_VELOCITY)
                vehicle.randomize_behavior()
                vehicle.setDistanceWanted(29 * np.random.random_sample() + 1) #self.np_random.randn() * 29
                self.road.vehicles.append(vehicle)
                self.other_vehicles.append(vehicle)


    def _make_ego_vehicles(self):

        """ 
            Ego-vehicle
            Creates Ego-vehicle on entrance lane to roundabout 
        """
        ego_lane = self.road.network.get_lane(variables.START_POS)

        # ego_lane.position is where the car will appear, and heading at is the direction

        if (self.steps%self.TIME_DIFFERENCE == 0 and len(self.ego_vehicles) < variables.num_ego_vehicles):
            longitud = random.randint(0, 50)
            lateral = random.randint(-35, 35)
            ego_vehicle = AgentVehicle(self.road,
                                     ego_lane.position(longitud,lateral),
                                     velocity=variables.MAX_VELOCITY,
                                     heading=ego_lane.heading_at(3),
                                     id=str(len(self.ego_vehicles))).plan_route_to("nxr")
            self.road.vehicles.append(ego_vehicle)
            self.ego_vehicles.append(ego_vehicle)

        #AgentVehicle.SPEED_MIN = 0
        #AgentVehicle.SPEED_MAX = 15
        #AgentVehicle.SPEED_COUNT = 4

    #Define the parameter settings for the run
    def set_configuration(self, risk_tol,threshold_tol,hv_tol,sigma):
        self.risk_tol = risk_tol
        self.threshold_tol = threshold_tol
        self.hv_tol = hv_tol
        self.sigma = sigma

def rad(deg):
        return deg*np.pi/180


register(
        id='roundaboutTraining-v1',
        entry_point='highwayEnv.environment:RoundaboutEnv',
    )
