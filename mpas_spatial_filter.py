
import numpy as np
from netCDF4 import Dataset
from scipy.sparse import csr_matrix

import matplotlib.pylab as plt


def topo_filt(mesh, ring=1):
    """
    Build a topology-based filter for an MPAS mesh, assembling a
    sparse matrix representing a 'tophat' operator for each cell
    in the mesh.

    Attributes
    ----------
    mesh : netCDF dataset
        The MPAS-O mesh data structure.

    ring : integer
        The number of topological 'rings' in filters.

    """
    # Darren Engwirda

#-- form the 1-ring adj. graph for cells as a sparse matrix

    xvec = np.array([], dtype=np.int32)
    ivec = np.array([], dtype=np.int32)
    jvec = np.array([], dtype=np.int32)

    nEdgesOnCell = np.asarray(mesh["nEdgesOnCell"][:])
    cellsOnCell = np.asarray(mesh["cellsOnCell"][:])

    for edge in range(np.max(nEdgesOnCell)):

        have = nEdgesOnCell > edge

        cidx = np.argwhere(have).ravel()

        okay = cellsOnCell[have, edge] - 1 >= 0

        cidx = cidx[okay]

        cadj = cellsOnCell[cidx, edge] - 1

        ivec = np.hstack((ivec, cidx))
        jvec = np.hstack((jvec, cadj))
        xvec = np.hstack((
            xvec, np.ones(cadj.size, dtype=np.int32)))

    ivec = np.hstack((
        ivec, np.arange(0, cellsOnCell.shape[0])))
    jvec = np.hstack((
        jvec, np.arange(0, cellsOnCell.shape[0])))
    xvec = np.hstack((xvec, np.ones(
        cellsOnCell.shape[0], dtype=np.int32)))

#-- expand to adj.-of-adj. through matrix multiplication

    filt = csr_matrix((xvec, (ivec, jvec))) ** ring

#-- reset all nz values to one, for un-weighted averages

    filt.data[:] = 1

    return filt


if (__name__ == "__main__"):

    mesh = Dataset("initial_state.nc", "r")

    xc = np.asarray(mesh["lonCell"][:])
    yc = np.asarray(mesh["latCell"][:])
    
    zb = np.asarray(mesh["bottomDepth"][:])
    zb = np.reshape(zb, (xc.size, 1))

#-- apply filtering:
#-- xx_filt = (F * xx) / |cells-per-filter|

    print("filtering (ring=2)")

    filt = topo_filt(mesh, ring=2)

    z2_filt = (filt * zb) / (filt * np.ones((filt.shape[0], 1)))

    print("drawing...")

    fig = plt.figure()
    plt.scatter(xc, yc, c=z2_filt, alpha=0.5, s=1)
    plt.axis("equal")
    plt.savefig('bottomDepth (filt=2).png')
    plt.close(fig)

    print("filtering (ring=4)")

    filt = topo_filt(mesh, ring=4)

    z4_filt = (filt * zb) / (filt * np.ones((filt.shape[0], 1)))

    print("drawing...")

    fig = plt.figure()
    plt.scatter(xc, yc, c=z4_filt, alpha=0.5, s=1)
    plt.axis("equal")
    plt.savefig('bottomDepth (filt=4).png')
    plt.close(fig)

    print("filtering (ring=8)")

    filt = topo_filt(mesh, ring=8)

    z8_filt = (filt * zb) / (filt * np.ones((filt.shape[0], 1)))

    print("drawing...")

    fig = plt.figure()
    plt.scatter(xc, yc, c=z8_filt, alpha=0.5, s=1)
    plt.axis("equal")
    plt.savefig('bottomDepth (filt=8).png')
    plt.close(fig)
    
#-- check filter topology:
#-- nonzeros per row represent adj. cells in tophat 

   #cell = 7
   #iadj = filt.indices[filt.indptr[cell]:filt.indptr[cell+1]]
   #print(iadj)

   #plt.scatter(xc, yc, c="k", alpha=0.5)
   #plt.scatter(xc[iadj], yc[iadj], c="r", alpha=0.5)
   #plt.scatter(xc[cell], yc[cell], c="b", alpha=0.5)
   #plt.axis("equal")
   #plt.show()

    mesh.close()
