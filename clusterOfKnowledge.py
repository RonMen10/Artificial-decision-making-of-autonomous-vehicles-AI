import pickle
import numpy as np
import random
from highwayEnv.Vars_and_Methods import variables
from highwayEnv.environment import roundabout
from rouletteWheel import *



def findCenter(id, dictionary):
    sumHyperVolume = 0
    sumFirstRisk = 0
    count = 0
    for i in range(0, len(dictionary['id'])):
        if int(dictionary['id'][i]) == id:
            sumHyperVolume += float(dictionary['hypervolume'][i])
            sumFirstRisk += float(dictionary['first_risk'][i])
            count += 1
    centerFirstRisk = sumFirstRisk / count
    centerHyperVolume = sumHyperVolume / count
    center = [centerHyperVolume, centerFirstRisk]
    return center


def findFardest(id1, id2, dictionary):
    "Information exchange from the id2 to the id1"
    center = findCenter(id1, dictionary)
    stateList = []
    distanceList = []
    index = 0
    for i in range(0, len(dictionary['id'])):
        if int(dictionary['id'][i]) == id2:
            distance = np.sqrt(
                (center[1] - dictionary['first_risk'][i]) ** 2 + (center[0] - dictionary['hypervolume'][i]) ** 2)
            distanceList.append(distance)
            stateList.append([dictionary['hypervolume'][i],
                              dictionary['first_risk'][i],
                              dictionary['counter'][i],
                              dictionary['threshold'][i],
                              dictionary['cumWaitTime'][i],
                              dictionary['cumCrashes'][i],
                              dictionary['avgFitness'][i],
                              dictionary['time_nd'][i],
                              dictionary['risk_nd'][i],
                              dictionary['case'][i],
                              dictionary['id'][i]])
    index = distanceList.index(max(distanceList))
    fardestState = stateList[index]
    return fardestState


def findClosest(id, dictionary):
    "Distances between each and every point inside a cluster. Returns matrix of distances"
    array3D = []
    distanceList = []
    newDictionary = statesById2(id=id, dictionary=dictionary)
    for i in range(0, len(newDictionary['id'])):
        distanceCompareList = []
        for j in range(0, len(newDictionary['id'])):
            dataList = []
            if newDictionary['first_risk'][i] == newDictionary['first_risk'][j] and newDictionary['hypervolume'][i] == \
                    newDictionary['hypervolume'][j]:
                distance = 1000
            else:
                distance = np.sqrt((newDictionary['first_risk'][i] - newDictionary['first_risk'][j]) ** 2 + (
                            newDictionary['hypervolume'][i] - newDictionary['hypervolume'][j]) ** 2)

            dataList = [distance, newDictionary['hypervolume'][i], newDictionary['first_risk'][i],
                        newDictionary['hypervolume'][j], newDictionary['first_risk'][j]]
            distanceList.append(distance)
            distanceCompareList.append(dataList)

        array3D.append(distanceCompareList)
    minDistance = min(distanceList)

    for i in range(0, len(newDictionary['id'])):
        for j in range(0, len(newDictionary['id'])):
            if minDistance == array3D[i][j][0]:
                hypervolume1 = array3D[i][j][1]
                first_risk1 = array3D[i][j][2]
                hypervolume2 = array3D[i][j][3]
                first_risk2 = array3D[i][j][4]
                break

    state1 = findBy(hypervolume1, first_risk1, newDictionary)
    state2 = findBy(hypervolume2, first_risk2, newDictionary)

    return state1


def statesById2(id, dictionary):
    dictionaryById = {'hypervolume': [], 'first_risk': [], 'counter': [], 'threshold': [], 'cumWaitTime': [],
                      'cumCrashes': [], 'avgFitness': [], 'time_nd': [], 'risk_nd': [], 'case': [], 'id': []}

    for i in range(0, len(dictionary['id'])):
        if int(dictionary['id'][i]) == id:
            dictionaryById['hypervolume'].append(dictionary['hypervolume'][i])
            dictionaryById['first_risk'].append(dictionary['first_risk'][i])
            dictionaryById['counter'].append(dictionary['counter'][i])
            dictionaryById['threshold'].append(dictionary['threshold'][i])
            dictionaryById['cumWaitTime'].append(dictionary['cumWaitTime'][i])
            dictionaryById['cumCrashes'].append(dictionary['cumCrashes'][i])
            dictionaryById['avgFitness'].append(dictionary['avgFitness'][i])
            dictionaryById['time_nd'].append(dictionary['time_nd'][i])
            dictionaryById['risk_nd'].append(dictionary['risk_nd'][i])
            dictionaryById['case'].append(dictionary['case'][i])
            dictionaryById['id'].append(dictionary['id'][i])
    return dictionaryById


def findBy(hypervolume, first_risk, dictionary):
    itemsList = []
    for i in range(0, len(dictionary['id'])):
        if float(dictionary['hypervolume'][i]) == hypervolume and float(dictionary['first_risk'][i]) == first_risk:
            itemsList.append(dictionary['hypervolume'][i])
            itemsList.append(dictionary['first_risk'][i])
            itemsList.append(dictionary['counter'][i])
            itemsList.append(dictionary['threshold'][i])
            itemsList.append(dictionary['cumWaitTime'][i])
            itemsList.append(dictionary['cumCrashes'][i])
            itemsList.append(dictionary['avgFitness'][i])
            itemsList.append(dictionary['time_nd'][i])
            itemsList.append(dictionary['risk_nd'][i])
            itemsList.append(dictionary['case'][i])
            itemsList.append(dictionary['id'][i])
    return itemsList


def infoExchange(dictionary, id1, id2):
    incomingState = findFardest(id1, id2, dictionary)
    outgoingState = findClosest(id1, dictionary)
    stateExchange(dictionary, incomingState, outgoingState)


def stateExchange(dictionary, incomingState, outgoingState):
    if len(incomingState) > 0:
        for i in range(0, len(dictionary['id'])):
            if dictionary['hypervolume'][i] == outgoingState[0] and dictionary['first_risk'][i] == outgoingState[1] and \
                    dictionary['id'][i] == outgoingState[10]:
                dictionary['hypervolume'][i] = incomingState[0]
                dictionary['first_risk'][i] = incomingState[1]
                dictionary['counter'][i] = incomingState[2]
                dictionary['threshold'][i] = incomingState[3]
                dictionary['cumWaitTime'][i] = incomingState[4]
                dictionary['cumCrashes'][i] = incomingState[5]
                dictionary['avgFitness'][i] = incomingState[6]
                dictionary['time_nd'][i] = incomingState[7]
                dictionary['risk_nd'][i] = incomingState[8]
                dictionary['case'][i] = incomingState[9]
    return print("clustered")

def findIds(dictionary, idList):
    idf = '-1'
    while idf not in dictionary['id']:
        idd = rouletteWheelMeth(idList)
        #id = random.choice(idList)
        idf = str(idd)
    return idd

def _is_terminal(dictionary):
    pickle.dump(dictionary, open(
        variables.path + '/Archives/tests.pkl', 'wb'))
    return True


def findBiggestGap(id, dictionary):
    newDictionary = statesById2(id, dictionary)
    newList = newDictionary['hypervolume']
    dif = 0
    difference = 0
    value1 = 0
    value2 = 0
    v1 = 0
    v2 = 0
    for i in range(len(newList)-1):
        if difference > dif:
            dif = difference
            v1 = value1
            v2 = value2

        difference = abs(newList[i] - newList[i+1])
        value1 = newList[i]
        value2 = newList[i+1]

    return v1, v2

def fillGap(id1, id2, dictionary):
    v1, v2 = findBiggestGap(id1, dictionary)
    outgoingDictionary = statesById2(id2, dictionary)
    incomingDictionary = statesById2(id1, dictionary)
    oldlist = incomingDictionary['hypervolume']
    outgoingList = outgoingDictionary['hypervolume']
    dif = 1000
    inValues = []
    newVal = -1
    newRisk = -1
    avg = (v1 + v2)/2
    maxVal = max(v1, v2)
    minVal = min(v1, v2)


    for i in range(len(outgoingList)):
        if minVal < outgoingList[i] and outgoingList[i] < maxVal:
            inValues.append(outgoingList[i])

    if len(inValues) > 0:
        for i in range(len(inValues)):
            distance = abs(avg - inValues[i])
            if distance < dif:
                dif = distance
                newVal = inValues[i]

    for i in range (len(outgoingDictionary['hypervolume'])):
        if newVal == -1:
            break
        elif outgoingDictionary['hypervolume'][i] == newVal:
            newRisk = outgoingDictionary['first_risk'][i]
            break
    state = findBy(newVal, newRisk, outgoingDictionary)
    return state

def gapExchange(id1, id2, dictionary):
    incomingState = fillGap(id1, id2, dictionary)
    outgoingState = findClosest(id1, dictionary)
    stateExchange(dictionary, incomingState, outgoingState)

def typeChange(aList):
    outList = []
    for i in aList:
        outList.append(int(i))
    return outList

def agentsExchanging(dictionary):
    listOfCarIDs = typeChange(roundabout.theList[1])
    id1 = findIds(dictionary, listOfCarIDs)
    listOfCarIDs = typeChange(roundabout.theList[2])
    listOfCarIDs.remove(id1)
    id2 = findIds(dictionary, listOfCarIDs)
    return id1, id2

def exchangeDirector(dictionary, id1, id2):
    listOfCarIDs = typeChange(roundabout.theList[1])
    if len(listOfCarIDs) >= 2:
        print(typeChange(roundabout.theList[1]))
        "Info exchange in both directions"
        infoExchange(dictionary, id1, id2)
        #infoExchange(dictionary, id2, id1)
        gapExchange(id1, id2, dictionary)
    else:
        print("No enough cars to exchange")


