from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def master():
    print(rank)
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

def worker():
    print(rank)
    data = np.zeros((3,3))

def main():
    print(size)
    if rank == 0:
        master()
    else:
        worker

if __name__ == '__main__':
    main()