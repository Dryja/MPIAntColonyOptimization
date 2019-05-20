import sys
from mpi4py import MPI
import numpy as np

# run command mpiexec python .\ant_colony.py map 1 1 1 0.4 0.3 0.2

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if len(sys.argv) != 8:
    if rank == 0:
        print(
            "Use: {} map_file antsNum externalIterNum onNodeIterNum alpha beta evaporationCoeff".format(
                sys.argv[0]
            )
        )
    exit()

if rank == 0:
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

    # TODO cast

# TODO logic

