import numpy as np 
import pandas as pd 

df=pd.read_csv('C:/GitHub/Collective-Decision-Making/test/TestStatistics/mean_stde.csv',sep=',')
print(df.columns)
df=df[df['Avg Success Rate']>= 0.85]
df=df[df['StdError Success Rate']<=0.015]
df=df[df['Avg Cumulated Waiting Time']<= 2000]
print(len(df))


def getPOParameters(time, success, stdetime, stdesuccess, parameters):
    POparameters=[]
    dominatedIndices=[]
    nonDominatedIndices=[]
    for i in range(0,len(time)):
        for j in range(0, len(time)):
                if ((time[i]<=time[j] and success[i]>=success [j] and stdetime[i] <= stdetime[j] and stdesuccess[i]<stdesuccess[j]) or
                (time[i]<time[j] and success[i]>=success [j] and stdetime[i] <= stdetime[j] and stdesuccess[i]<=stdesuccess[j]) or
                (time[i]<=time[j] and success[i]>success [j] and stdetime[i] <= stdetime[j] and stdesuccess[i]<=stdesuccess[j]) or
                (time[i]<=time[j] and success[i]>=success [j] and stdetime[i] < stdetime[j] and stdesuccess[i]<=stdesuccess[j])):
                    if j not in dominatedIndices:
                        dominatedIndices.append(j)

    nonDominatedIndices = Diff(list(range(0, len(time))), dominatedIndices)
    for index in nonDominatedIndices:
        POparameters.append(parameters[index])

    return list(set(POparameters))


def Diff(li1, li2): 
        li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2] 
        return li_dif 

time=df['Avg Cumulated Waiting Time'].values
success=df['Avg Success Rate'].values
stdetime=df['StdError Cumulated Waiting Time'].values
stdesuccess=df['StdError Success Rate'].values
parameters=df['Run'].values
print(parameters)

POParameters = getPOParameters(time,success,stdetime,stdesuccess,parameters)

for i in POParameters:
    dft=df[df['Run']==i]
    print(i, dft['Avg Number of States'].values)

                