import sys, random, math
from mpi4py import MPI
import numpy as np
from timeit import default_timer as timer
from ant_func import *

# run command mpiexec python .\ant_colony.py map.txt random.txt 1 4 1 0.4 0.3 0.2

comm = MPI.COMM_WORLD
status = MPI.Status 
rank = comm.Get_rank()
size = comm.Get_size()

randomNumbers = []
start = None
bestCost = math.inf
nRandomNumbers = antsNum = externalIterNum = onNodeIterNum = nCities = 0
alpha = beta = evaporationCoeff = 0.0
terminationCondition = otherTerminationCondition = 0
otherBestCost = external_loop_counter = 0
antsBestCost = math.inf

if len(sys.argv) != 9:
    if rank == 0:
        print(
            "Use:  map_file nRandomNumbers antsNum externalIterNum onNodeIterNum alpha beta evaporationCoeff".format(
                sys.argv[0]
            )
        )
    exit()

if rank == 0:
    print("Number of nodes:", size)
    map = np.loadtxt(sys.argv[1], skiprows=1, dtype=int)
    print(map.shape)
    randomNumbers = np.loadtxt(sys.argv[2], skiprows=1, dtype=int)
    antsNum = int(sys.argv[3])
    externalIterNum = int(sys.argv[4])
    onNodeIterNum = int(sys.argv[5])
    alpha = float(sys.argv[6])
    beta = float(sys.argv[7])
    evaporationCoeff = float(sys.argv[8])
    print("Iterations:", externalIterNum * onNodeIterNum)
    nCities = len(map)
    nRandomNumbers = len(randomNumbers)
    start = timer()

nRandomNumbers = comm.bcast(nRandomNumbers, root=0)
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

nAnts, nAntsBeforeMe = countAnts(rank, size, antsNum)

random_counter = (onNodeIterNum * nAntsBeforeMe * nCities) % nRandomNumbers

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
