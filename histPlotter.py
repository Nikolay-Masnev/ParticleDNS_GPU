import numpy as np
import sys
import matplotlib.pyplot as plt

if __name__=="__main__":
    data = np.loadtxt(sys.argv[1], dtype=float).flatten() 
    BoxSize = float(sys.argv[2])
    nBins = data.size
    x = np.arange(0, BoxSize, BoxSize/nBins)

    data = data * nBins / BoxSize

    plt.plot(x, data)
    plt.xlabel('r')
    plt.ylabel('n(r)')
    ymax = max(data)
    plt.ylim(0, 1.05 * ymax)
    plt.savefig(sys.argv[3], format="pdf", bbox_inches="tight")