import sys
import random

import numpy as np


def random_map(n):
    return np.random.randint(1, 1000, size=(n, n))

def random_numbers(nNumber):
    return np.random.randint(1, np.iinfo(np.int32).max, size=nNumber)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Use: {} file_output_map number_of_cites file_output_random number_of_randoms".format(sys.argv[0]))
    else:
        n = int(sys.argv[2])
        nNumber = int(sys.argv[4])
        mat = random_map(n)
        np.savetxt(sys.argv[1], mat, fmt="%d", header=str(n), comments="")
        np.savetxt(sys.argv[3], random_numbers(nNumber), fmt="%d", header=str(nNumber), comments="")
