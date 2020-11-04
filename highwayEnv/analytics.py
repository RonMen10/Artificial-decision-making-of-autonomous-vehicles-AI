import matplotlib.pyplot as plt
import pandas as pd
from highwayEnv.vehicles.control import AgentVehicle


threshold_list = []
xlist = []


def collect_thresholds(agent_list):
    """Return the list of all the thresholds from the agents 
    """
    threshold_list = []
    for agent in agent_list:
        threshold_list.append(agent.risk_threshold)
    return threshold_list


def build_boxplot(agent_list):
    """Plot of all the thresholds 
    """
    threshold_list.append(collect_thresholds(agent_list))
    fig1, ax1 = plt.subplots()
    ax1.set_title('Boxplot of agent thresholds')
    ax1.boxplot(threshold_list)
    plt.show()


def build_scatterplot(self, agent_list):
    """Build a scatterplot for the agent's thresholds
    """
    self.plot_counter += 1

    threshold_list.extend(collect_thresholds(agent_list))

    for i in range(len(agent_list)):
        xlist.append(self.plot_counter)

    fig1, ax1 = plt.subplots()
    ax1.set_title('Boxplot of agent thresholds')
    ax1.scatter(x=xlist, y=threshold_list)

    plt.show()


def calculate_statistics(archive, cumulated_crashes, cumulated_time, risk_tol, threshold_tol, hv_tol, sigma, attempts, steps, selected_seed, distance_to_solution):
    """Return a dataframe with all the data used for analysis 

    Data frame includes the following data:
        'Parameters of the Run', 'Number of States learned', 'Average Best Fitness', 'Cumulated Waiting Time', 
        'Cumulated Crashes', 'Success Rate', 'number of simulation steps', 'Distances to solution list'.
    """

    archive_df = pd.DataFrame(archive)

    # Training data:
    # number of states
    number_states = len(archive_df.groupby(
        ['hypervolume', 'first_risk']).size())

    # average of best fitness for each stage
    temp_df = archive_df.loc[:, ['hypervolume', 'first_risk', 'avgFitness']]
    avg_best_fitness = temp_df.groupby(
        ['hypervolume', 'first_risk']).min().mean()

    # Test data:
    success_rate = 1. - (cumulated_crashes / attempts)

    # create DF and save it
    result_df = pd.DataFrame({'Run': '{}_{}_{}_{}_{}'.format(risk_tol, threshold_tol,
                                                             hv_tol, sigma, selected_seed),
                              'Number of States': number_states,
                              'Average Best Fitness': avg_best_fitness,
                              'Cumulated Waiting Time': cumulated_time,
                              'Cumulated Crashes': cumulated_crashes,
                              'Success Rate': success_rate,
                              'Steps': steps,
                              'Distances list': [distance_to_solution]})
    return result_df
