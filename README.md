# Citing article

A. De Rango, et al., "Partitioned Reduction for Heterogeneous Environments," in 2024 32nd Euromicro International Conference on Parallel, Distributed and Network-Based Processing (PDP), Dublin, Ireland, 2024 pp. 285-289.
doi: 10.1109/PDP62718.2024.00047

# Build Library and Test Code

The number of cores per socket and the number of cores per node must be set in the code in the constants MAX_CPUS_PER_SOCKET and MAX_CPUS_PER_NODE respectively.

The test is prepared for integer type messages.
 
The software can be built having the compilers in $PATH

make

# Running on local machine

The script qexec provides an example of execution with the needed setting of environment variables:

- GROUP_SIZE: Number of partitions of a message 
- CALCULUS: Amount of calculation
- MAX_LOOPS: Number of repetitions of the collective (cache re-use is avoided)
- NDATA: size in bytes of the message (minimum = sizeof(int) * GROUP_SIZE)

```bash
export GROUP_SIZE=20
export CALCULUS=20
export MAX_LOOPS=10
export NDATA=80
mpirun -np 40 ./testPR
```
