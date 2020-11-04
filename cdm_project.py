import gym

# Import the modules we need from the folder, so we can modify the files and update them with github
import highwayEnv
from highwayEnv.Vars_and_Methods import methods, variables
from multiprocessing import Pool
import pandas as pd


def roundabout_project(risk_tol, threshold_tol, hv_tol, sigma, seeds_list):
    """Returns a dictionary with the data of the tested agents.

    This function gets the hyper parameters to run and performs 2 tasks:

    1.- Training phase, where all the agents run for certain time in the Roundabout Environment
        learning how to enter the roundabout with a seed of "1" for the random parameters.

    2.- Test phase, where the agents get the collected knowledge gained in the training face and
        run in a different Roundabout Environment according to the seed we are providing, at the
        end we return a dictionary with data about how good the agents performed.

        Dictionary includes the following data:
        'Parameters of the Run', 'Number of States learned', 'Average Best Fitness', 'Cumulated Waiting Time', 
        'Cumulated Crashes', 'Success Rate', 'Distances to solution list'.
    """

    # Training phase
    # Load the roundabout.py environment
    env = gym.make("roundaboutTraining-v1")
    env.set_configuration(risk_tol, threshold_tol, hv_tol, sigma)
    done = False
    print("Training...")
    while not done:
        action = 0  # First action is Idle
        done = env.step(action)
        methods.render(env)
    env.close()

    # Test phase
    # Load the roundaboutTest.py environment
    # Run for different seeds
    for seed in seeds_list:
        env = gym.make("roundaboutTest-v1")
        env.set_configuration(risk_tol, threshold_tol, hv_tol, sigma, seed)
        done = False
        print("Testing...")
        while not done:
            action = 0  # First action is Idle
            done, this_dictionary = env.step(action)
            methods.render(env)
        env.close()

        data_per_run = pd.DataFrame(this_dictionary)
        data_per_run.to_csv(variables.path + '/TestStatistics/Stats_{0}_{1}_{2}_{3}_{4}.csv'.format(
            risk_tol, threshold_tol, hv_tol, sigma, seed), index=None, header=True)


if __name__ == '__main__':
    """ Performs the parallelization part.

        Here we run the script several times at the same time depending on the cores we have on the
        computer, for each core or process we run one combination of parameters.

        Give the parameters list that to test in the form: [(risk_tolerance, threshold_tolerance, hv_tolerance, sigma, seed_list)]
        Use starmap to give more attributes to a function in parallelized mode.

        At the end of everything it will save the data collected in a csv file
    """

    parameter_list = []
    seeds_list = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    #seeds_list = [3]
    for parameters in [(0.05, 0.05, 0.05, 0.05), (0.05, 0.05, 0.05, 0.1), (0.05, 0.05, 0.05, 0.2), (0.05, 0.05, 0.1, 0.05), (0.05, 0.05, 0.1, 0.1), (0.05, 0.05, 0.1, 0.2), (0.05, 0.05, 0.2, 0.05), (0.05, 0.05, 0.2, 0.1), (0.05, 0.05, 0.2, 0.2), (0.05, 0.025, 0.05, 0.05), (0.05, 0.025, 0.05, 0.1), (0.05, 0.025, 0.05, 0.2), (0.05, 0.025, 0.1, 0.05), (0.05, 0.025, 0.1, 0.1), (0.05, 0.025, 0.1, 0.2), (0.05, 0.025, 0.2, 0.05), (0.05, 0.025, 0.2, 0.1), (0.05, 0.025, 0.2, 0.2), (0.05, 0.075, 0.05, 0.05), (0.05, 0.075, 0.05, 0.1), (0.05, 0.075, 0.05, 0.2), (0.05, 0.075, 0.1, 0.05), (0.05, 0.075, 0.1, 0.1), (0.05, 0.075, 0.1, 0.2), (0.05, 0.075, 0.2, 0.05), (0.05, 0.075, 0.2, 0.1), (0.05, 0.075, 0.2, 0.2), (0.1, 0.05, 0.05, 0.05), (0.1, 0.05, 0.05, 0.1), (0.1, 0.05, 0.05, 0.2), (0.1, 0.05, 0.1, 0.05), (0.1, 0.05, 0.1, 0.1), (0.1, 0.05, 0.1, 0.2), (0.1, 0.05, 0.2, 0.05), (0.1, 0.05, 0.2, 0.1), (0.1, 0.05, 0.2, 0.2), (0.1, 0.025, 0.05, 0.05), (0.1, 0.025, 0.05, 0.1), (0.1, 0.025, 0.05, 0.2), (0.1, 0.025, 0.1, 0.05), (0.1, 0.025, 0.1, 0.1), (0.1, 0.025, 0.1, 0.2), (0.1, 0.025, 0.2, 0.05), (0.1, 0.025, 0.2, 0.1), (0.1, 0.025, 0.2, 0.2), (0.1, 0.075, 0.05, 0.05), (0.1, 0.075, 0.05, 0.1), (0.1, 0.075, 0.05, 0.2), (0.1, 0.075, 0.1, 0.05), (0.1, 0.075, 0.1, 0.1), (0.1, 0.075, 0.1, 0.2), (0.1, 0.075, 0.2, 0.05), (0.1, 0.075, 0.2, 0.1), (0.1, 0.075, 0.2, 0.2), (0.2, 0.05, 0.05, 0.05), (0.2, 0.05, 0.05, 0.1), (0.2, 0.05, 0.05, 0.2), (0.2, 0.05, 0.1, 0.05), (0.2, 0.05, 0.1, 0.1), (0.2, 0.05, 0.1, 0.2), (0.2, 0.05, 0.2, 0.05), (0.2, 0.05, 0.2, 0.1), (0.2, 0.05, 0.2, 0.2), (0.2, 0.025, 0.05, 0.05), (0.2, 0.025, 0.05, 0.1), (0.2, 0.025, 0.05, 0.2), (0.2, 0.025, 0.1, 0.05), (0.2, 0.025, 0.1, 0.1), (0.2, 0.025, 0.1, 0.2), (0.2, 0.025, 0.2, 0.05), (0.2, 0.025, 0.2, 0.1), (0.2, 0.025, 0.2, 0.2), (0.2, 0.075, 0.05, 0.05), (0.2, 0.075, 0.05, 0.1), (0.2, 0.075, 0.05, 0.2), (0.2, 0.075, 0.1, 0.05), (0.2, 0.075, 0.1, 0.1), (0.2, 0.075, 0.1, 0.2), (0.2, 0.075, 0.2, 0.05), (0.2, 0.075, 0.2, 0.1), (0.2, 0.075, 0.2, 0.2)]:
    #for parameters in [(0.05, 0.05, 0.05, 0.05)]:
        parameter_list.append((parameters + (seeds_list,)))

    pool = Pool(processes=30)  # Number of processes we want to paralellize
    pool.starmap(roundabout_project, parameter_list)
    pool.close()