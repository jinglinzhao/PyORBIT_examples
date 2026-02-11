#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J HD102365_all_instr_0p_emcee
### -- ask for number of cores (default: 1) -- 
#BSUB -n 16
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 8GB of memory per core/slot -- 
#BSUB -R "rusage[mem=8GB]"
### -- specify that we want the job to get killed if it exceeds 9GB per core/slot -- 
#BSUB -M 9GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 48:00
### -- set the email address -- 
#BSUB -u jzhao@space.dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
#BSUB -o /work2/lbuc/jzhao/PyORBIT_ESSP/HD102365/out_HD102365_emcee/Output_HD102365_all_instr_0p_emcee.out

# Change to configuration directory
cd /work2/lbuc/jzhao/PyORBIT_ESSP/HD102365/results_HD102365_emcee/all_instr/0p/HD102365_all_instr_0p_emcee

# Clean up previous runs
rm -f configuration_file_emcee_run_HD102365_all_instr_0p_emcee.log

# Activate PyORBIT environment
source /work2/lbuc/iara/anaconda3/etc/profile.d/conda.sh
conda activate pyorbit

# Run PyORBIT analysis with emcee
pyorbit_run emcee HD102365_all_instr_0p_emcee.yaml > configuration_file_emcee_run_HD102365_all_instr_0p_emcee.log
pyorbit_results emcee HD102365_all_instr_0p_emcee.yaml -all >> configuration_file_emcee_run_HD102365_all_instr_0p_emcee.log

# Create results directory and copy files
mkdir -p /work2/lbuc/jzhao/PyORBIT_ESSP/HD102365/results_HD102365_emcee/all_instr/0p/HD102365_all_instr_0p_emcee/HD102365_all_instr_0p_emcee
cp HD102365_all_instr_0p_emcee.yaml /work2/lbuc/jzhao/PyORBIT_ESSP/HD102365/results_HD102365_emcee/all_instr/0p/HD102365_all_instr_0p_emcee/HD102365_all_instr_0p_emcee/
cp configuration_file_emcee_run_HD102365_all_instr_0p_emcee.log /work2/lbuc/jzhao/PyORBIT_ESSP/HD102365/results_HD102365_emcee/all_instr/0p/HD102365_all_instr_0p_emcee/HD102365_all_instr_0p_emcee/

# Deactivate environment
conda deactivate

echo "Job HD102365_all_instr_0p_emcee completed at: $(date)"
