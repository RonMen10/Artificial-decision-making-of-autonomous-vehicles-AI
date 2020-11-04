import random
"Sets of individuals:"

"3 taxi drivers"
tp0 = 0.20

"3 old guys"
tp1 = 0.001

"4 common drivers"
tp2 = 0.09925


def individualProbability (id):
    if id == 0 or id == 1 or id == 2:
        tp = tp0
    elif id == 3 or id == 4 or id == 5:
        tp = tp1
    else:
        tp = tp2
    return tp

def rouletteWheelMeth(idList):
    tpSum = 0
    tpList = [0]
    for i in range(0, len(idList)):
        tpSum += individualProbability(int(idList[i]))
        tpList.append(tpSum)

    pick = random.uniform(0, tpSum)
    index = 1
    while tpList[index] <= pick:
        index += 1
    return idList[index - 1]
