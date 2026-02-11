#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J HD102365_no_espresso_3p_dynesty
### -- ask for number of cores (default: 1) -- 
#BSUB -n 16
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=4GB]"
### -- specify that we want the job to get killed if it exceeds 5GB per core/slot -- 
#BSUB -M 5GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 72:00
### -- set the email address -- 
#BSUB -u jzhao@space.dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
#BSUB -o /work2/lbuc/jzhao/PyORBIT_ESSP/HD102365/out_HD102365_dynesty/Output_HD102365_no_espresso_3p_dynesty.out

# Change to configuration directory
cd /work2/lbuc/jzhao/PyORBIT_ESSP/HD102365/results_HD102365_dynesty_test/no_espresso/3p/HD102365_no_espresso_3p_dynesty

# Clean up previous runs
rm -f configuration_file_dynesty_run_HD102365_no_espresso_3p_dynesty.log

# Activate PyORBIT environment
# source ~/anaconda3/etc/profile.d/conda.sh
source /zhome/9d/b/207249/anaconda3/etc/profile.d/conda.sh
# source /work2/lbuc/iara/anaconda3/etc/profile.d/conda.sh
conda activate pyorbit

# Set CPU affinity and threading environment variables
export OMP_NUM_THREADS=1              # Prevent nested parallelism
export MKL_NUM_THREADS=1              # Intel MKL threading
export OPENBLAS_NUM_THREADS=1         # OpenBLAS threading
export NUMEXPR_NUM_THREADS=1          # NumExpr threading
export OMP_PROC_BIND=true             # Bind threads to cores
export OMP_PLACES=cores               # Use physical cores

# Run PyORBIT analysis with dynesty
pyorbit_run dynesty HD102365_no_espresso_3p_dynesty.yaml > configuration_file_dynesty_run_HD102365_no_espresso_3p_dynesty.log
pyorbit_results dynesty HD102365_no_espresso_3p_dynesty.yaml -all >> configuration_file_dynesty_run_HD102365_no_espresso_3p_dynesty.log

# Create results directory and copy files
# mkdir -p /work2/lbuc/jzhao/PyORBIT_ESSP/HD102365/results_HD102365_dynesty_test/no_espresso/3p/HD102365_no_espresso_3p_dynesty/HD102365_no_espresso_3p_dynesty
# cp HD102365_no_espresso_3p_dynesty.yaml /work2/lbuc/jzhao/PyORBIT_ESSP/HD102365/results_HD102365_dynesty_test/no_espresso/3p/HD102365_no_espresso_3p_dynesty/HD102365_no_espresso_3p_dynesty/
# cp configuration_file_dynesty_run_HD102365_no_espresso_3p_dynesty.log /work2/lbuc/jzhao/PyORBIT_ESSP/HD102365/results_HD102365_dynesty_test/no_espresso/3p/HD102365_no_espresso_3p_dynesty/HD102365_no_espresso_3p_dynesty/

# Deactivate environment
conda deactivate

echo "Job HD102365_no_espresso_3p_dynesty completed at: $(date)"
