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
    probabilities = []
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
    probabilities = computeProbabilities(currentCity, path, map, nCities, pheromons, alpha, beta);
    
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

comm = MPI.COMM_WORLD
status = MPI.Status 
rank = comm.Get_rank()
size = comm.Get_size()
randomNumbers = []
nRandomNumbers = 200
start = None
map = []
nAntsPerNode = []
bestCost = math.inf

if len(sys.argv) != 8:
    if rank == 0:
        print(
            "Use:  map_file antsNum externalIterNum onNodeIterNum alpha beta evaporationCoeff".format(
                sys.argv[0]
            )
        )
    exit()

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
    nCities = map.size**1/2

    for i in range(1, nRandomNumbers):
        randomNumbers.append(random.randint(1, 100000))

if MPI.Bcast(nRandomNumbers, 1, MPI.LONG, 0, comm) != MPI.SUCCESS:
    print("Node  : Error in Broadcast of nRandomNumbers".format(rank))
    MPI.Finalize()
    exit(-1)

if MPI.Bcast(randomNumbers, nRandomNumbers, MPI.LONG, 0, comm) != MPI.SUCCESS:
    print("Node  : Error in Broadcast of RandomNumbers".format(rank))
    MPI.Finalize()
    exit(-1)

if MPI.Bcast(antsNum, 1, MPI.INT, 0, comm) != MPI.SUCCESS: 
    print("Node {} : Error in Broadcast of antsNum".format(rank))
    MPI.Finalize()  
    exit(-1)

if MPI.BCAST(onNodeIterNum, 1, MPI.LONG, 0, comm) != MPI.SUCCESS: 
    print("Node {} : Error in Broadcast of onNodeIterNum".format(rank))
    MPI.Finalize()  
    exit(-1)  

if MPI.BCAST(externalIterNum, 1, MPI.LONG, 0, comm) != MPI.SUCCESS:
    print("Node {} : Error in Broadcast of externalIterNum".format(rank))
    MPI.Finalize()  
    exit(-1)

if MPI.BCAST(alpha, 1, MPI.DOUBLE, 0, comm) != MPI.SUCCESS: 
    print("Node {} : Error in Broadcast of alpha".format(rank))
    MPI.Finalize()  
    exit(-1)

if MPI.BCAST(beta, 1, MPI.DOUBLE, 0, comm) != MPI.SUCCESS: 
    print("Node {} : Error in Broadcast of beta".format(rank))
    MPI.Finalize()  
    exit(-1)

if MPI.BCAST(evaporationCoeff, 1, MPI.DOUBLE, 0, comm) != MPI.SUCCESS: 
    print("Node {} : Error in Broadcast of evaporationCoeff".format(rank))
    MPI.Finalize()  
    exit(-1)

if MPI.BCAST(nCities, 1, MPI.INT, 0, comm) != MPI.SUCCESS: 
    print("Node {} : Error in Broadcast of nCities".format(rank))
    MPI.Finalize()  
    exit(-1)

if MPI.Bcast(map, nCities*nCities, MPI.INT, 0, comm) != MPI.SUCCESS:
    print("Node {} : Error in Broadcast of map".format(rank))
    MPI.Finalize()  
    exit(-1)

pheromons = []
pheromonsUpdate = []
localPheromonsPath = []
otherBestPath = []
otherPheromonsPath = []
bestPath = []
currentPath = []

for i in range(nCities):
    otherBestPath.append(-1)
    currentPath.append(-1)
    bestPath.append(-1)

for i in range(nCities * nCities):
    pheromons.append(0.1)

antsPerNode = antsNum / size;
restAnts = antsNum - antsPerNode * size;

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
    
        for j in range(nCities*nCities):
            pheromons[j] *= evaporationCoeff

        updatePheromons(pheromons, bestPath, bestCost, nCities);
    
        loop_counter += 1
    
    findPheromonsPath(localPheromonsPath, bestPath, pheromons, nCities);

    for j in range(nCities*nCities):
        pheromonsUpdate[j] = 1.0

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
    
        if (MPI.Bcast(otherBestPath, nCities, MPI.INT, i, comm) != MPI.SUCCESS):
            print("Node {} : Error in Broadcast of otherBestPath".format(rank));
            MPI.Finalize()
            exit(-1)
        if (MPI.Bcast(otherPheromonsPath, nCities, MPI.DOUBLE, i, comm) != MPI.SUCCESS):
            print("Node {} : Error in Broadcast of otherPheromonsPath".format(rank))
            MPI.Finalize()
            exit(-1)
        if (MPI.Bcast(otherTerminationCondition, 1, MPI.LONG, i, comm) != MPI.SUCCESS):
            print("Node {} : Error in Broadcast of otherTerminationCondition".format(rank))
            MPI.Finalize()
            exit(-1)
        if (MPI.Bcast(otherBestCost, 1, MPI.LONG, i, comm) != MPI.SUCCESS):
            print("Node {} : Error in Broadcast of otherBestCost".format(rank))
            MPI.Finalize()
            exit(-1)

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

    for j in range(nCities*nCities):
        pheromons[j] = pheromons[j] / pheromonsUpdate[j]

    bestCost = tempBestCost
    bestPath = tempBestPath[:]
    terminationCondition = tempTerminationCondition


    external_loop_counter += 1

    random_counter = (random_counter + (onNodeIterNum * (antsNum - nAnts) * nCities)) % nRandomNumbers;

if (rank == 0):
    for i in range(1, size):
        if (MPI.Recv(otherBestPath, nCities, MPI.INT, i, MPI.ANY_TAG, comm, status) != MPI.SUCCESS):
            print("Node {} : Error in Recv of otherBestPath".format(rank))
            MPI.Finalize()
            exit(-1)
        oldCost = bestCost
        bestCost = computeCost(bestCost, bestPath, otherBestPath, map, nCities);

        if (oldCost > bestCost):
            bestPath = otherBestPath[:]
else:
    if (MPI.Send(bestPath, nCities, MPI.INT, 0, 0, comm) != MPI.SUCCESS): 
        print("Node {} : Error in Send of bestPath".format(rank))
        MPI.Finalize()
        exit(-1)

if (rank == 0):
    print("best cost : {}\n".format(bestCost))

    end = timer()
    print("TotalTime {}\n".format(end - start))

MPI.Finalize()
