#PBS -q dirac_reg
#PBS -l nodes=1:ppn=1:fermi
#PBS -l walltime=00:10:00
#PBS -A gpgpu
#PBS -N run08k-n1_6th
#PBS -e run-008k-n01_6th.err
#PBS -o run-008k-n01_6th.out
#PBS -V

module unload pgi
module unload gcc
module unload openmpi
module load   openmpi-gnu/1.4.2
module unload cuda/3.1
module load   cuda/3.0
cd $PBS_O_WORKDIR
mpirun -np 1 ./gpu-6th
