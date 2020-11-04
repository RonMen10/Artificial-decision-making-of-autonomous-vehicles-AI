import pickle

from highwayEnv.Vars_and_Methods import variables
import pandas as pd

#pickle_in = open(variables.path + "/Archives/tests.pkl", "rb")
pickle_in = open(variables.path + "/Archives/0.05_0.05_0.05_0.05.pkl", "rb")
example_dict = pickle.load(pickle_in)
data_per_run = pd.DataFrame(example_dict)
data_per_run.to_csv(variables.path + '/Ronald.csv')