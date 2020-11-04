from __future__ import division, print_function, absolute_import
import pandas
from gym import spaces
import numpy as np

from highwayEnv.Vars_and_Methods import methods
from highwayEnv.road.lane import AbstractLane
from highwayEnv.vehicles.control import AgentVehicle


class KinematicObservation(object):
    """
        Observe the kinematics of nearby vehicles.
    """
    FEATURES = ['presence', 'x', 'y', 'vx', 'vy']

    def __init__(self, env,
                 features=FEATURES,
                 vehicles_count=5,
                 features_range=None,
                 absolute=False,
                 flatten=False,
                 **kwargs):
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        """
        self.env = env
        self.features = features
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        self.absolute = absolute
        self.flatten = flatten

    def space(self):
        shape = (self.vehicles_count * len(self.features),) if self.flatten \
            else (self.vehicles_count, len(self.features))
        return spaces.Box(shape=shape, low=-1, high=1, dtype=np.float32)

    def normalize(self, df,car_number):
        """
            Normalize the observation values.

            For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            side_lanes = self.env.road.network.all_side_lanes(self.env.ego_vehicles[car_number].lane_index)
            self.features_range = {
                "x": [-5.0 * AgentVehicle.SPEED_MAX, 5.0 * AgentVehicle.SPEED_MAX],
                "y": [-AbstractLane.DEFAULT_WIDTH * len(side_lanes), AbstractLane.DEFAULT_WIDTH * len(side_lanes)],
                "vx": [-2*AgentVehicle.SPEED_MAX, 2*AgentVehicle.SPEED_MAX],
                "vy": [-2*AgentVehicle.SPEED_MAX, 2*AgentVehicle.SPEED_MAX]
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = methods.remap(df[feature], [f_range[0], f_range[1]], [-1, 1])
        return df

    def observe(self):
        if len(self.env.ego_vehicles) >0:
            # Add ego-vehicle
            df = pandas.DataFrame.from_records([self.env.ego_vehicles[0].to_dict()])[self.features]
            for i in range(len(self.env.ego_vehicles)):
                # Add nearby traffic
                close_vehicles = self.env.road.close_vehicles_to(self.env.ego_vehicles[i],
                                                                 self.env.PERCEPTION_DISTANCE,
                                                                 self.vehicles_count - 1)
                if close_vehicles:
                    origin = self.env.ego_vehicles[i] if not self.absolute else None
                    df = df.append(pandas.DataFrame.from_records(
                        [v.to_dict(origin)
                         for v in close_vehicles[-self.vehicles_count + 1:]])[self.features],
                                   ignore_index=True)
                # Normalize
                df = self.normalize(df,i)
                # Fill missing rows
                if df.shape[0] < self.vehicles_count:
                    rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
                    df = df.append(pandas.DataFrame(data=rows, columns=self.features), ignore_index=True)
            # Reorder
            df = df[self.features]
            # Clip
            obs = np.clip(df.values, -1, 1)
            # Flatten
            if self.flatten:
                obs = np.ravel(obs)
            return obs

def observation_factory(env, config):
    if config["type"] == "TimeToCollision":
        return TimeToCollisionObservation(env, **config)
    elif config["type"] == "Kinematics":
        return KinematicObservation(env, **config)
    elif config["type"] == "OccupancyGrid":
        return OccupancyGridObservation(env, **config)
    elif config["type"] == "KinematicsGoal":
        return KinematicsGoalObservation(env, **config)
    elif config["type"] == "GrayscaleObservation":
        return GrayscaleObservation(env, **config)
    else:
        raise ValueError("Unknown observation type")
