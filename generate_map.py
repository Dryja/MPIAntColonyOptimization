import sys
import random

import numpy as np


def random_map(n):
    return np.random.randint(1, 1000, size=(n, n))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Use: {} file_output number_of_cites".format(sys.argv[0]))
    else:
        n = int(sys.argv[2])
        mat = random_map(n)
        np.savetxt(sys.argv[1], mat, fmt="%d", header=str(n), comments="")
