import sys, random, math
from mpi4py import MPI
import numpy as np
from timeit import default_timer as timer
# run command mpiexec python .\ant_colony.py map 1 1 1 0.4 0.3 0.2

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

if len(sys.argv) != 8:
    if rank == 0:
        print(
            "Use:  map_file antsNum externalIterNum onNodeIterNum alpha beta evaporationCoeff".format(
                sys.argv[0]
            )
        )
    exit()

comm = MPI.COMM_WORLD
status = MPI.Status 
rank = comm.Get_rank()
size = comm.Get_size()
randomNumbers = []
nRandomNumbers = 200
start = None
nAntsPerNode = []
bestCost = math.inf
antsNum = externalIterNum = onNodeIterNum = nCities = 0
randomNumbers = []
alpha = beta = evaporationCoeff = 0.0
random_counter = 0
terminationCondition = 0
otherTerminationCondition = 0
otherBestCost = 0


if rank == 0:
    start = timer()
    print("Number of nodes:", size)
    map = np.loadtxt(sys.argv[1], skiprows=1, dtype=int)
    print(map.shape)
    antsNum = int(sys.argv[2])
    externalIterNum = int(sys.argv[3])
    onNodeIterNum = int(sys.argv[4])
    alpha = float(sys.argv[5])
    beta = float(sys.argv[6])
    evaporationCoeff = float(sys.argv[7])
    print("Iterations:", externalIterNum * onNodeIterNum)
    nCities = int(np.asscalar(np.loadtxt(sys.argv[1], max_rows=1)))

    for i in range(nRandomNumbers):
        randomNumbers.append(random.randint(1, 100000))

randomNumbers = comm.bcast(randomNumbers, root=0)
antsNum = comm.bcast(antsNum, root=0) 
onNodeIterNum = comm.bcast(onNodeIterNum, root=0)
externalIterNum = comm.bcast(externalIterNum, root=0) 
alpha = comm.bcast(alpha, root=0)
beta = comm.bcast(beta, root=0)
evaporationCoeff = comm.bcast(evaporationCoeff, root=0)
nCities = comm.bcast(nCities, root=0) 
map = comm.bcast(map, root=0) 

pheromons = np.full((nCities, nCities), 0.1)
pheromonsUpdate = np.full((nCities, nCities), 1)
localPheromonsPath = otherPheromonsPath = [0.0] * nCities
bestPath = currentPath = otherBestPath = [-1] * nCities

antsPerNode = antsNum // size
restAnts = antsNum - antsPerNode * size

for i in range(size):
    nAntsPerNode.append(antsPerNode)
    if restAnts > i: 
        nAntsPerNode[i] += 1

nAnts = nAntsPerNode[rank]

nAntsBeforeMe = 0
for i in range(size):
    if i < rank:
        nAntsBeforeMe += nAntsPerNode[i]
    else: 
        break

random_counter = (random_counter + (onNodeIterNum * nAntsBeforeMe * nCities)) % nRandomNumbers

antsBestCost = math.inf

external_loop_counter = 0

while external_loop_counter < externalIterNum:
    loop_counter = 0
    while loop_counter < onNodeIterNum:
        for ant_counter in range(nAnts):
            for i in range(nCities):
                currentPath[i] = -1

            rand = randomNumbers[random_counter]
            currentCity = rand % nCities
            random_counter = (random_counter + 1) % nRandomNumbers
            currentPath[currentCity] = 0
            for cities_counter in range(1, nCities):
                rand = randomNumbers[random_counter]
                random_counter = (random_counter + 1) % nRandomNumbers
                currentCity = computeNextCity(currentCity, currentPath, map, nCities, pheromons, alpha, beta, rand)
                if currentCity == -1:
                    print("There is an error choosing the next city in iteration {} for ant {} on node {}\n".format(loop_counter, ant_counter, rank))
                    MPI.Finalize()
                    exit(-1)
    
                currentPath[currentCity] = cities_counter
            oldCost = bestCost
            bestCost = computeCost(bestCost, bestPath, currentPath, map, nCities)
    
            if oldCost > bestCost:
                bestPath = currentPath[:]
        
        if (bestCost < antsBestCost):
            antsBestCost = bestCost
            terminationCondition = 0
        else:
            terminationCondition += 1

        with np.nditer(pheromons, op_flags=['readwrite']) as pheromon:
            for x in pheromon:
                x *= evaporationCoeff


        updatePheromons(pheromons, bestPath, bestCost, nCities)
    
        loop_counter += 1
    
    findPheromonsPath(localPheromonsPath, bestPath, pheromons, nCities)

    tempBestCost = bestCost
    tempBestPath = [0]*nCities
    tempPheromonsPath = [0]*nCities
    tempTerminationCondition = terminationCondition
    tempBestPath = bestPath[:]

    for i in range(size):
        if (rank == i):
            otherBestPath = bestPath[:]
            otherPheromonsPath = localPheromonsPath[:]
            tempPheromonsPath = localPheromonsPath[:]
            otherTerminationCondition = terminationCondition
            otherBestCost = bestCost
    
        otherBestPath = comm.bcast(otherBestPath, root=0) 
        otherPheromonsPath = comm.bcast(otherPheromonsPath, root=0) 
        otherTerminationCondition = comm.bcast(otherTerminationCondition, root=0)
        otherBestCost =  comm.bcast(otherBestCost, root=0)

        if (rank != i):
            if (otherBestCost < tempBestCost):
                tempTerminationCondition = otherTerminationCondition
                tempBestCost = otherBestCost
                tempBestPath = otherBestPath[:]
                tempPheromonsPath = otherPheromonsPath[:]
            elif (otherBestCost == tempBestCost):
                tempTerminationCondition += otherTerminationCondition

    for j in range(nCities-1):
        pheromonsUpdate[tempBestPath[j],tempBestPath[j+1]] += 1.0
        pheromonsUpdate[tempBestPath[j+1],tempBestPath[j]] += 1.0
        pheromons[tempBestPath[j],tempBestPath[j+1]] += tempPheromonsPath[j]
        pheromons[tempBestPath[j+1],tempBestPath[j]] += tempPheromonsPath[j]
    
    pheromonsUpdate[tempBestPath[nCities-1],tempBestPath[0]] += 1.0
    pheromonsUpdate[tempBestPath[0],tempBestPath[nCities-1]] += 1.0
    pheromons[tempBestPath[nCities-1],tempBestPath[0]] += tempPheromonsPath[nCities - 1]
    pheromons[tempBestPath[0],tempBestPath[nCities-1]] += tempPheromonsPath[nCities - 1]

    pheromons /= pheromonsUpdate

    bestCost = tempBestCost
    bestPath = tempBestPath[:]
    terminationCondition = tempTerminationCondition


    external_loop_counter += 1

    random_counter = (random_counter + (onNodeIterNum * (antsNum - nAnts) * nCities)) % nRandomNumbers

if (rank == 0):
    for i in range(1, size):
        otherBestPath = comm.recv(source=i)
        bestCost = computeCost(bestCost, bestPath, otherBestPath, map, nCities)

        if (oldCost > bestCost):
            bestPath = otherBestPath[:]
else:
    comm.send(bestPath, dest=0)

if (rank == 0):
    print("best cost : {}\n".format(bestCost))

    end = timer()
    print("TotalTime {}\n".format(end - start))

MPI.Finalize()
