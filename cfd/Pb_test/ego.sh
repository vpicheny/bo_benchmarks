# Submission script for the helloWorld program
#
# Comments starting with #OAR are used by the 
# resource manager
#
#OAR -l /nodes=1/core=8,walltime=48:00:00
#OAR -p cluster='dellc6420'
#
#OAR -q default
# The job is submitted to the default queue

module load mpi/openmpi-1.10.7-gcc

Rscript ego.r


