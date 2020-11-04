from highwayEnv.environment import roundabout
import matplotlib.pyplot as plt
import math
from highwayEnv.Vars_and_Methods import variables as var
import sys
from scipy.spatial import distance
import pickle
import random
from clusterOfKnowledge import *


class Archive(object):

    def __init__(self, config=None):


        self.archive = {'hypervolume': [], 'first_risk': [], 'counter': [], 'threshold': [], 'cumWaitTime': [],
               'cumCrashes': [], 'avgFitness': [], 'time_nd': [], 'risk_nd': [], 'case': [], 'id': []}


    def hypervolume(self,risks, waiting_times):
        """Returns the value of the hypervolume. 

        This function takes 2 lists: risks and a waiting times for a given front
        calculate the hypervolume using the point (First waiting time - 0.5 , 0) as reference point
        and return the value.
        (Usually the first waiting time is 0)
        """

        x = waiting_times[0]-0.5
        hv = 0.0

        for i in range(len(risks)):
            hv += (waiting_times[i] - x) * risks[i]
            x = waiting_times[i]
        return hv

    def search_similar_states(self, agent, risk_tol, hv_tol):
        """Search in the archive for similar states within the tolerances."""
        similar_states = []

        for index, archive_hv in enumerate(self.archive['hypervolume']):
            # if hv and risk difference are already in the archive put it in the similar states list
            if (archive_hv - hv_tol <= agent.hypervolume <= archive_hv + hv_tol and self.archive['first_risk'][index] - risk_tol
                    <= agent.non_dominated_risk[0] <= self.archive['first_risk'][index] + risk_tol and agent.id == self.archive['id'][index]):
                similar_states.append(index)
        return similar_states

    def closest_states(self,agent, similar_states):
        """Identify the nearest similar state in the archive to the current state based on euclidean distance."""
        selected_states = []
        fitness_list = []
        smallest_distance = sys.float_info.max

        for state in similar_states:
            min_value = distance.euclidean(
                (self.archive['hypervolume'][state], self.archive['first_risk'][state]), (agent.hypervolume, agent.non_dominated_risk[0]))
            if min_value < smallest_distance:
                smallest_distance = min_value
                selected_states.clear()
                fitness_list.clear()
                selected_states.append(state)
                fitness_list.append(self.calculate_caf(self.archive['counter'][state], self.archive['avgFitness'][state]))
            elif min_value == smallest_distance:
                selected_states.append(state)
                fitness_list.append(self.calculate_caf(self.archive['counter'][state], self.archive['avgFitness'][state]))
        return selected_states, fitness_list

    def search_test_archive(self,agent, risk_tol, hv_tol, distances_to_solution):
        """Set the value of the agent's risk threshold. 

        This function takes 4 parameters: 
            1. agent: the agent calling this function
            2. risk_tol: the tolerance of the risk used for comparing states on the archive
            3. hv_tol: the tolerance of hypervolume used for comparing states on the archive
            4. distances_to_solution: a list that stores the distance to the closest state (only 
                if we don't find a similar state)

        For the TEST ENVIRONMENT...
        Search in the archive for a similar state (using the eucleadian distance with hypervolume 
        and first risk) and search for the best solution depending on the fitness, then take the 
        threshold of that and set it as the agent's risk threshold.

        If we don't find a similar state then take the closest similar state.
        """
        agent.archive_index = -1
        best_fitness = sys.float_info.max
        similar_states = []
        smallest_distance = sys.float_info.max
        smallest_distance_index = 0

        for index, archive_hv in enumerate(self.archive['hypervolume']):

            # if hv and risk difference are alrady in the archive put it in the similar states list
            if (archive_hv - hv_tol <= agent.hypervolume <= archive_hv + hv_tol and self.archive['first_risk'][index] - risk_tol
                    <= agent.non_dominated_risk[0] <= self.archive['first_risk'][index] + risk_tol):
                similar_states.append(index)

            # calculating minimum distance to use in case we don't find a similar state
            min_value = distance.euclidean(
                (archive_hv, self.archive['first_risk'][index]), (agent.hypervolume, agent.non_dominated_risk[0]))
            if (min_value < smallest_distance):
                smallest_distance = min_value
                smallest_distance_index = index

        # if there are no similar states
        if len(similar_states) == 0:
            agent.risk_threshold = self.archive['threshold'][smallest_distance_index]
            distances_to_solution.append(smallest_distance)

        else:
            for state in similar_states:
                if self.archive['avgFitness'][state] < best_fitness:
                    best_fitness = self.archive['avgFitness'][state]
                    agent.archive_index = state
            agent.risk_threshold = self.archive['threshold'][agent.archive_index]


    def search_archive(self,agent, risk_tol, threshold_tol, hv_tol):
        """Set the value of the agent's threshold. 

        This function takes 4 parameters: 
        1. agent: the agent calling this function
        2. risk_tol: the tolerance of the risk used for comparing states on the archive
        3. threshold_tol: the tolerance of threshold used for comparing similar thresholds
        4. hv_tol: the tolerance of hypervolume used for comparing states on the archive


        For the TRAINING ENVIRONMENT...
        Search in the archive for the state and choose a risk threshold to test, if the current 
        state is not in the archive then keep the current risk threshold.
        """
        agent.archive_index = -1
        similar_states = self.search_similar_states(agent, risk_tol, hv_tol)
        if len(similar_states) != 0:
            selected_states, fitness_list = self.closest_states(agent, similar_states)
            # Probability of select a new threshold to test, or to use caf to select the next threshold to try
            new_threshold_probability = (
                min(fitness_list) + 100)/110 if (min(fitness_list) + 100)/110 <= 1 else 1
            if random.random() <= new_threshold_probability:
                return
            else:
                self.select_threshold(agent, selected_states)

        else:
            return


    def select_threshold(self,agent, selected_states):
        caf = sys.float_info.max
        # Don't create new threshold, then select with caf
        for state_index in selected_states:
            # use the UCB to select a threshold that should be evaluated
            caf_temp = self.calculate_caf(
                self.archive['counter'][state_index], self.archive['avgFitness'][state_index])
            if caf_temp < caf:
                caf = caf_temp
                agent.archive_index = state_index
        agent.risk_threshold = self.archive['threshold'][agent.archive_index]


    def update_archive(self, agent, risk_tol, threshold_tol, hv_tol, archive_index, hypervolume, non_dominated_risk, case):
        """Update the archive with the new values. 

        This function takes 7 parameters: 
        1. agent: the agent calling this function
        2. risk_tol: the tolerance of the risk used for comparing states on the archive
        3. threshold_tol: the tolerance of threshold used for comparing similar thresholds
        4. hv_tol: the tolerance of hypervolume used for comparing states on the archive
        5. archive_index: the index in the archive to be modified
        6. hypervolume: the hypervolume to be taken into account
        7. non_dominated_risk: the list of non dominated risks to be taken into account

        For the TRAINING ENVIRONMENT...
        Update the archive depending on the solution selected by the agent, If the agent selected a solution
        from an existing state then update that state, If the agent selected a solution from a new state
        then add that state to the archive.
        """

        # already existing solution found and archive not yet updated
        if archive_index != -1 and agent.archive_updated == False:
            self.archive['counter'][archive_index] += 1
            self.archive['cumWaitTime'][archive_index] += agent.waiting_time
            self.archive['cumCrashes'][archive_index] += int(agent.crashed)
            self.archive['avgFitness'][archive_index] = (self.archive['cumWaitTime'][archive_index] +
                                                    self.archive['cumCrashes'][archive_index]*var.CRASH_PENALTY)/self.archive['counter'][archive_index]
            self.archive['case'][archive_index].append(case)
            agent.archive_updated = True

        # Else if we selected a new state and tried it and archive not yet updated
        elif archive_index == -1 and agent.archive_updated == False:

            # Check again the archive because it might be that we have new states while we were in the roundabout
            similar_states = self.search_similar_states(agent, risk_tol, hv_tol)
            selected_states, fitness_list = self.closest_states(agent, similar_states)
            closest_threshold = -1
            for state in selected_states:
                smallest_difference = sys.float_info.max
                if (self.archive['threshold'][state]-threshold_tol
                        <= agent.risk_threshold <= self.archive['threshold'][state] + threshold_tol):
                    threshold_diff = abs(self.archive['threshold'][state] - agent.risk_threshold)
                    if threshold_diff < smallest_difference:
                        smallest_difference = threshold_diff
                        closest_threshold = state

            # Case similar threshold            
            if closest_threshold != -1:
                self.archive['counter'][closest_threshold] += 1
                self.archive['cumWaitTime'][closest_threshold] += agent.waiting_time
                self.archive['cumCrashes'][closest_threshold] += int(agent.crashed)
                self.archive['avgFitness'][closest_threshold] = (self.archive['cumWaitTime'][closest_threshold] +
                                                self.archive['cumCrashes'][closest_threshold]*var.CRASH_PENALTY) / self.archive['counter'][closest_threshold]
                self.archive['case'][closest_threshold].append(case)
                agent.archive_updated = True

            # Case new Threshold for an existing state
            elif agent.archive_updated == False and len(selected_states) > 0:
                self.archive['hypervolume'].append(
                    self.archive['hypervolume'][selected_states[0]])
                self.archive['first_risk'].append(
                    self.archive['first_risk'][selected_states[0]])
                self.archive['counter'].append(1)
                self.archive['threshold'].append(agent.risk_threshold)
                self.archive['cumWaitTime'].append(agent.waiting_time)
                self.archive['cumCrashes'].append(int(agent.crashed))
                self.archive['avgFitness'].append(
                    agent.waiting_time + int(agent.crashed)*var.CRASH_PENALTY)
                self.archive['time_nd'].append([agent.non_dominated_time])
                self.archive['risk_nd'].append([non_dominated_risk])
                self.archive['case'].append([case])
                self.archive['id'].append(agent.id)
                agent.archive_updated = True

            # Case completely new state
            elif agent.archive_updated == False:
                self.archive['hypervolume'].append(hypervolume)
                self.archive['first_risk'].append(non_dominated_risk[0])
                self.archive['counter'].append(1)
                self.archive['threshold'].append(agent.risk_threshold)
                self.archive['cumWaitTime'].append(agent.waiting_time)
                self.archive['cumCrashes'].append(int(agent.crashed))
                self.archive['avgFitness'].append(
                    agent.waiting_time + int(agent.crashed) * var.CRASH_PENALTY)
                self.archive['time_nd'].append([agent.non_dominated_time])
                self.archive['risk_nd'].append([non_dominated_risk])
                self.archive['case'].append([case])
                self.archive['id'].append(agent.id)
                agent.archive_updated = True


    def calculate_caf(self,counter, fitness):
        return fitness - var.CAF_FACTOR / counter**2


    def get_archive(self):
        return self.archive

    def get_archive2(self):
        dic = statesById2(7, self.archive)
        return dic


    def set_archive(self,risk_tol, threshold_tol, hv_tol, sigma):
        self.archive = pickle.load(open(
            var.path + '/Archives/{0}_{1}_{2}_{3}.pkl'.format(risk_tol, threshold_tol, hv_tol, sigma), 'rb'))

    def idValues (self):
        id1, id2 = agentsExchanging(self.archive)
        return id1, id2

    def clusterUpdateArchive(self, id1, id2):
        exchangeDirector(self.archive, id1, id2)

