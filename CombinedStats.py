import pandas as pd
import numpy as np
import math


seeds = [(3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,), (12,), (13,), (14,), (15,), (16,), (17,), (18,), (19,), (20,), (21,), (22,)]
parameter_list = [(0.05, 0.05, 0.05, 0.05), (0.05, 0.05, 0.05, 0.1), (0.05, 0.05, 0.05, 0.2), (0.05, 0.05, 0.1, 0.05), (0.05, 0.05, 0.1, 0.1), (0.05, 0.05, 0.1, 0.2), (0.05, 0.05, 0.2, 0.05), (0.05, 0.05, 0.2, 0.1), (0.05, 0.05, 0.2, 0.2), (0.05, 0.025, 0.05, 0.05), (0.05, 0.025, 0.05, 0.1), (0.05, 0.025, 0.05, 0.2), (0.05, 0.025, 0.1, 0.05), (0.05, 0.025, 0.1, 0.1), (0.05, 0.025, 0.1, 0.2), (0.05, 0.025, 0.2, 0.05), (0.05, 0.025, 0.2, 0.1), (0.05, 0.025, 0.2, 0.2), (0.05, 0.075, 0.05, 0.05), (0.05, 0.075, 0.05, 0.1), (0.05, 0.075, 0.05, 0.2), (0.05, 0.075, 0.1, 0.05), (0.05, 0.075, 0.1, 0.1), (0.05, 0.075, 0.1, 0.2), (0.05, 0.075, 0.2, 0.05), (0.05, 0.075, 0.2, 0.1), (0.05, 0.075, 0.2, 0.2), (0.1, 0.05, 0.05, 0.05), (0.1, 0.05, 0.05, 0.1), (0.1, 0.05, 0.05, 0.2), (0.1, 0.05, 0.1, 0.05), (0.1, 0.05, 0.1, 0.1), (0.1, 0.05, 0.1, 0.2), (0.1, 0.05, 0.2, 0.05), (0.1, 0.05, 0.2, 0.1), (0.1, 0.05, 0.2, 0.2), (0.1, 0.025, 0.05, 0.05), (0.1, 0.025, 0.05, 0.1), (0.1, 0.025, 0.05, 0.2), (0.1, 0.025, 0.1, 0.05), (0.1, 0.025, 0.1, 0.1), (0.1, 0.025, 0.1, 0.2), (0.1, 0.025, 0.2, 0.05), (0.1, 0.025, 0.2, 0.1), (0.1, 0.025, 0.2, 0.2), (0.1, 0.075, 0.05, 0.05), (0.1, 0.075, 0.05, 0.1), (0.1, 0.075, 0.05, 0.2), (0.1, 0.075, 0.1, 0.05), (0.1, 0.075, 0.1, 0.1), (0.1, 0.075, 0.1, 0.2), (0.1, 0.075, 0.2, 0.05), (0.1, 0.075, 0.2, 0.1), (0.1, 0.075, 0.2, 0.2), (0.2, 0.05, 0.05, 0.05), (0.2, 0.05, 0.05, 0.1), (0.2, 0.05, 0.05, 0.2), (0.2, 0.05, 0.1, 0.05), (0.2, 0.05, 0.1, 0.1), (0.2, 0.05, 0.1, 0.2), (0.2, 0.05, 0.2, 0.05), (0.2, 0.05, 0.2, 0.1), (0.2, 0.05, 0.2, 0.2), (0.2, 0.025, 0.05, 0.05), (0.2, 0.025, 0.05, 0.1), (0.2, 0.025, 0.05, 0.2), (0.2, 0.025, 0.1, 0.05), (0.2, 0.025, 0.1, 0.1), (0.2, 0.025, 0.1, 0.2), (0.2, 0.025, 0.2, 0.05), (0.2, 0.025, 0.2, 0.1), (0.2, 0.025, 0.2, 0.2), (0.2, 0.075, 0.05, 0.05), (0.2, 0.075, 0.05, 0.1), (0.2, 0.075, 0.05, 0.2), (0.2, 0.075, 0.1, 0.05), (0.2, 0.075, 0.1, 0.1), (0.2, 0.075, 0.1, 0.2), (0.2, 0.075, 0.2, 0.05), (0.2, 0.075, 0.2, 0.1), (0.2, 0.075, 0.2, 0.2)] 

df=pd.DataFrame()
for parameters in parameter_list:
    df_all_seeds=pd.DataFrame(columns=['Run','Number of States','Average Best Fitness','Cumulated Waiting Time','Cumulated Crashes','Success Rate','Max Distance','States not found','Waiting Time per Try'])
    for seed in seeds:
        try:
            df_temp=pd.read_csv('C:/GitHub/Collective-Decision-Making/test/TestStatistics/Stats_{}_{}_{}_{}_{}.csv'.format(parameters[0],parameters[1],parameters[2],parameters[3],seed[0]))
            try:
                df_temp['Max Distance'] = max(list(map(float,df_temp['Distances list'][0][1:-1].split(', '))))
                df_temp['States not found']=len(list(map(float,df_temp['Distances list'][0][1:-1].split(', '))))
            except:
                df_temp['Max Distance'] = 0
                df_temp['States not found'] = 0
            df_temp['Attempts'] = df_temp['Cumulated Crashes']/(1.0-df_temp['Success Rate'])
            df_temp['Waiting Time per Try'] = df_temp['Cumulated Waiting Time']/(df_temp['Attempts'])

            df_temp=df_temp.drop(['Distances list'],axis=1)
            df_all_seeds=df_all_seeds.append(df_temp)

        except:
            print('Stats_{}_{}_{}_{}_{}'.format(parameters[0],parameters[1],parameters[2],parameters[3],seed[0]))

    df=df.append({'Run':'{}_{}_{}_{}'.format(parameters[0],parameters[1],parameters[2],parameters[3]),
        'Avg Number of States':df_all_seeds['Number of States'].mean(),
        'Avg Best Fitness':df_all_seeds['Average Best Fitness'].mean(),
        'Avg Cumulated Waiting Time':df_all_seeds['Cumulated Waiting Time'].mean(),
        'Avg Cumulated Crashes':df_all_seeds['Cumulated Crashes'].mean(),
        'Avg Success Rate':df_all_seeds['Success Rate'].mean(),
        'Avg Max Distance':df_all_seeds['Max Distance'].mean(),
        'Avg States not found':df_all_seeds['States not found'].mean(),
        'Avg Waiting Time per Try':df_all_seeds['Waiting Time per Try'].mean(),
        'StdError Number of States':0,
        'StdError Best Fitness':0,
        'StdError Cumulated Waiting Time':df_all_seeds['Cumulated Waiting Time'].std()/math.sqrt(20),
        'StdError Cumulated Crashes':df_all_seeds['Cumulated Crashes'].std()/math.sqrt(20),
        'StdError Success Rate':df_all_seeds['Success Rate'].std()/math.sqrt(20),
        'StdError Max Distance':df_all_seeds['Max Distance'].std()/math.sqrt(20),
        'StdError States not found':df_all_seeds['States not found'].std()/math.sqrt(20),
        'StdError Waiting Time per Try':df_all_seeds['Waiting Time per Try'].std()/math.sqrt(20)
        }, ignore_index=True)
    

df=df[['Run','Avg Number of States', 'StdError Number of States','Avg Best Fitness', 'StdError Best Fitness','Avg Cumulated Waiting Time','StdError Cumulated Waiting Time','Avg Cumulated Crashes', 'StdError Cumulated Crashes','Avg Success Rate','StdError Success Rate','Avg Max Distance','StdError Max Distance','Avg States not found','StdError States not found','Avg Waiting Time per Try','StdError Waiting Time per Try']]
df.to_csv('C:/GitHub/Collective-Decision-Making/test/TestStatistics/mean_stde.csv')