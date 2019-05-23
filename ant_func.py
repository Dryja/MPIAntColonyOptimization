import math

def findPheromonsPath (pheromonsPath, bestPath, pheromons, nCities):
    previousCity = 0
    nextCity = 0
    for i in range(1, nCities):
        previousCity = bestPath[i - 1]
        nextCity = bestPath[i]
        pheromonsPath[i-1] = pheromons[previousCity, nextCity]
    pheromonsPath[nCities - 1] = pheromons[nextCity, bestPath[0]]

def computeProbabilities(currentCity, path, map, nCities, pheromons, alpha, beta):
    probabilities = [0]*nCities
    total = 0
    for i in range(nCities):  
        if (path[i] != -1 or i == currentCity):
            probabilities[i] = 0
        else:
            p = math.pow(1.0 / map[currentCity,i],alpha) * math.pow(pheromons[currentCity,i], beta)
            probabilities[i] = p
            total += p

    if (total == 0):
        for i in range(nCities):
            if (path[i] == -1 or i != currentCity):
                probabilities[i] = 1
                total += 1

    for i in range(nCities):
        probabilities[i] = probabilities[i] / total
    return probabilities

def countAnts(rank, size, antsNum):
    nAntsPerNode = []
    nAntsBeforeMe = 0

    antsPerNode = antsNum // size
    restAnts = antsNum - antsPerNode * size
    
    for i in range(size):
        nAntsPerNode.append(antsPerNode)
        if restAnts > i: 
            nAntsPerNode[i] += 1

    nAnts = nAntsPerNode[rank]

    for i in range(size):
        if i < rank:
            nAntsBeforeMe += nAntsPerNode[i]
        else: 
            break
    
    return (nAnts, nAntsBeforeMe)

def computeNextCity(currentCity, path, map, nCities, pheromons, alpha, beta, random):
    probabilities = computeProbabilities(currentCity, path, map, nCities, pheromons, alpha, beta)
    value = (random % 100) + 1
    sum = 0
    for i in range(nCities):
        sum += math.ceil(probabilities[i] * 100)
        if (sum >= value):
            return i
    return -1

def updatePheromons(pheromons, path, cost, nCities):
    orderedCities = [0] * nCities
    for i in range(nCities):
        order = path[i]
        orderedCities[order] = i

    for i in range(nCities-1):
        pheromons[orderedCities[i],orderedCities[i + 1]] += 1.0/cost
        pheromons[orderedCities[i + 1],orderedCities[i]] += 1.0/cost
        if (pheromons[orderedCities[i],orderedCities[i + 1]] > 1):
            pheromons[orderedCities[i],orderedCities[i + 1]] = 1.0
            pheromons[orderedCities[i + 1],orderedCities[i]] = 1.0

    pheromons[orderedCities[nCities - 1],orderedCities[0]] += 1.0/cost
    pheromons[orderedCities[0],orderedCities[nCities - 1]] += 1.0/cost
    if (pheromons[orderedCities[nCities - 1],orderedCities[0]] > 1.0):
        pheromons[orderedCities[nCities - 1],orderedCities[0]] = 1.0
        pheromons[orderedCities[0],orderedCities[nCities - 1]] = 1.0

def computeCost(bestCost, bestPath, currentPath, map, nCities):
    currentCost = 0
    orderedCities = [0] * nCities
    for i in range(nCities):
        orderedCities[currentPath[i]] = i

    for i in range(nCities-1):
        currentCost += map[orderedCities[i],orderedCities[i + 1]]
  
    currentCost += map[orderedCities[nCities - 1], orderedCities[0]]

    if (bestCost > currentCost):
        return currentCost
    else:
        return bestCost